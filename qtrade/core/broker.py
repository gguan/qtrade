# components/broker.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .trade import Trade
from .order import Order
from .position import Position
from .commission import Commission


# Mapping of common case-insensitive column aliases to canonical capitalized names.
_COMMON_NAMES = {
    "date": "Date",
    "time": "Time",
    "timestamp": "Timestamp",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj_close": "Adj_Close",
    "volume": "Volume",
}


def _canonicalize_columns(df: pd.DataFrame) -> None:
    df.rename(columns=lambda x: _COMMON_NAMES[x.lower()] if x.lower() in _COMMON_NAMES else x, inplace=True)


class Broker:
    """
    The Broker class is responsible for executing orders and managing positions.

    Internally Broker tracks positions and OHLCV data per asset (keyed by string symbol);
    a single-asset DataFrame passed to the constructor is wrapped under the key
    ``"default"`` for backwards compatibility, so existing single-asset workflows
    continue to use the same external API (``broker.data``, ``broker.position``).

    Attributes:
        cash (float): Current shared cash balance across all assets.
        commission (Optional[Commission]): Instance for calculating trade commissions. If None, no commission is applied.
        margin_ratio (float): Margin ratio (0 < margin_ratio ≤ 1).
        trade_on_close (bool): If True, orders are filled at the current close price. Otherwise, at the next open price.
        current_time (pd.Timestamp): Timestamp of the current bar.
        _pending_orders (List[Order]): Pending orders (e.g., stop/limit orders) awaiting execution.
        _executing_orders (List[Order]): Orders to be executed at the next bar's open price if trade_on_close is False.
        _filled_orders (List[Order]): List of filled orders.
        _closed_orders (List[Order]): List of rejected/canceled orders.
        _equity_history (pd.Series): Portfolio-level historical record of account equity.
    """

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        cash: float,
        commission: Commission | None,
        margin_ratio: float | dict[str, float],
        trade_on_close: bool,
        contract_multiplier: float | dict[str, float] | None = None,
    ):
        """
        Initialize the Broker with market data and account settings.

        Args:
            data (pd.DataFrame | dict[str, pd.DataFrame]): Market data with
                ['Open', 'High', 'Low', 'Close'] columns. Pass a single DataFrame for
                a single-asset backtest (wrapped internally as ``{"default": data}``)
                or a dict of DataFrames keyed by asset symbol for multi-asset.
            cash (float): Initial cash balance. Must be positive.
            commission (Optional[Commission]): Commission calculator instance.
            margin_ratio (float | dict[str, float]): Margin requirement. Pass a scalar
                to apply uniformly, or a dict keyed by asset for per-asset settings
                (e.g. 1.0 for stocks, 0.05 for futures). Each value must be in (0, 1].
            trade_on_close (bool): Execution mode for orders.
            contract_multiplier (float | dict[str, float] | None, optional): Contract
                size for futures-style instruments. ``100`` for COMEX gold (1 price tick
                = $100 P&L per contract), ``50`` for E-mini S&P. Pass a dict keyed by
                asset for portfolios mixing stocks and futures of different categories.
                Defaults to ``None`` (multiplier of 1.0 — stock semantics).

        Raises:
            AssertionError: If cash is not positive or any margin_ratio is out of bounds.
            ValueError: If a per-asset dict is missing keys or has unknown keys.
        """
        assert cash > 0, "Initial cash must be positive."

        # Normalize to dict[str, DataFrame]; canonicalize column names per asset.
        if isinstance(data, pd.DataFrame):
            data_by_asset = {"default": data}
        else:
            assert len(data) > 0, "Multi-asset data dict cannot be empty."
            data_by_asset = data
        for df in data_by_asset.values():
            _canonicalize_columns(df)

        self._data_by_asset: dict[str, pd.DataFrame] = data_by_asset

        # Per-asset margin ratios.
        self._margin_ratio_by_asset = self._normalize_per_asset(
            margin_ratio, data_by_asset, name="margin_ratio", default=None,
        )
        for asset, mr in self._margin_ratio_by_asset.items():
            assert 0 < mr <= 1, f"margin_ratio for '{asset}' must be in (0, 1], got {mr}"

        # Per-asset contract multipliers (default 1.0 = stock semantics).
        self._multiplier_by_asset = self._normalize_per_asset(
            contract_multiplier, data_by_asset, name="contract_multiplier", default=1.0,
        )
        for asset, mult in self._multiplier_by_asset.items():
            assert mult > 0, f"contract_multiplier for '{asset}' must be positive, got {mult}"

        self.cash = cash
        self.commission = commission
        self.trade_on_close = trade_on_close
        self._positions: dict[str, Position] = {a: Position() for a in data_by_asset}

        # Use the first asset's index as the timeline; multi-asset support assumes
        # aligned indexes (validated in P2).
        first_index = next(iter(data_by_asset.values())).index
        self.current_time = first_index[0]

        self._pending_orders: list[Order] = []
        self._executing_orders: list[Order] = []

        self._filled_orders: list[Order] = []
        self._closed_orders: list[Order] = []  # Rejected and canceled orders

        # Initialize equity_history with NaN — process_bar fills each entry as
        # the simulation advances. The starting bar gets cash so consumers
        # like calculate_stats can use iloc[0] as the initial value. Reading
        # equity_history mid-backtest (e.g. from inside a strategy) sees NaN
        # for un-processed future bars instead of a misleading flat-cash line.
        self._equity_history = pd.Series(data=np.nan, index=first_index, dtype='float64')
        self._equity_history.iloc[0] = self.cash

    @staticmethod
    def _normalize_per_asset(
        value, data_by_asset, *, name: str, default,
    ) -> dict[str, float]:
        """Coerce a scalar / dict / None into a per-asset dict.

        - ``None`` and ``default is not None`` → uniform fill with ``default``.
        - scalar → uniform fill with that scalar.
        - dict → keys must exactly match ``data_by_asset``.
        """
        if value is None:
            if default is None:
                raise TypeError(f"{name} is required (no default).")
            return {a: float(default) for a in data_by_asset}
        if isinstance(value, dict):
            missing = set(data_by_asset) - set(value)
            extra = set(value) - set(data_by_asset)
            if missing or extra:
                raise ValueError(
                    f"{name} dict keys must exactly match data assets. "
                    f"Missing: {sorted(missing)}, extra: {sorted(extra)}."
                )
            return {a: float(value[a]) for a in data_by_asset}
        return {a: float(value) for a in data_by_asset}

    # ------------------------------------------------------------------
    # Backwards-compatible single-asset accessors.
    # ------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Single-asset access. For multi-asset, use :attr:`data_by_asset`."""
        if len(self._data_by_asset) > 1:
            raise AttributeError(
                "Broker has multiple assets; use broker.data_by_asset[symbol] instead of broker.data"
            )
        return next(iter(self._data_by_asset.values()))

    @property
    def position(self) -> Position:
        """Single-asset access. For multi-asset, use :attr:`positions`."""
        if len(self._positions) > 1:
            raise AttributeError(
                "Broker has multiple assets; use broker.positions[symbol] instead of broker.position"
            )
        return next(iter(self._positions.values()))

    @property
    def data_by_asset(self) -> dict[str, pd.DataFrame]:
        """Per-asset OHLCV data."""
        return self._data_by_asset

    @property
    def positions(self) -> dict[str, Position]:
        """Per-asset Position objects."""
        return self._positions

    @property
    def assets(self) -> list[str]:
        """List of asset symbols this broker tracks."""
        return list(self._data_by_asset.keys())

    @property
    def margin_ratio(self) -> float:
        """Single-asset margin ratio. For multi-asset, use :attr:`margin_ratio_by_asset`."""
        if len(self._margin_ratio_by_asset) > 1:
            raise AttributeError(
                "Broker has multiple assets; use broker.margin_ratio_by_asset[symbol]"
            )
        return next(iter(self._margin_ratio_by_asset.values()))

    @property
    def margin_ratio_by_asset(self) -> dict[str, float]:
        """Per-asset margin ratios."""
        return dict(self._margin_ratio_by_asset)

    @property
    def multiplier_by_asset(self) -> dict[str, float]:
        """Per-asset contract multipliers (1.0 = stock semantics)."""
        return dict(self._multiplier_by_asset)

    # ------------------------------------------------------------------
    # Aggregate properties (portfolio-level for multi-asset, identical to
    # single-asset behavior when only "default" is present).
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        """Calculate the current equity of the account (cash + total unrealized P&L)."""
        return self.cash + self.unrealized_pnl

    @property
    def cumulative_returns(self) -> float:
        """Cumulative returns (equity / initial equity)."""
        return self.equity / self._equity_history.iloc[0]

    @property
    def available_margin(self) -> float:
        """Available margin for new trades, summed across all assets.

        Per-asset margin = ``abs(size) × multiplier × current_price × margin_ratio[asset]``.
        """
        used_margin = 0.0
        for asset, position in self._positions.items():
            current_price = self._data_by_asset[asset].loc[self.current_time, 'Close']
            mr = self._margin_ratio_by_asset[asset]
            used_margin += sum(
                abs(trade.size) * trade.multiplier * current_price * mr
                for trade in position.active_trades
            )
        return max(0, self.equity - used_margin)

    @property
    def unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all active trades on every asset.

        Per-trade P&L = ``size × multiplier × (current_price − entry_price)``.
        """
        total = 0.0
        for asset, position in self._positions.items():
            current_price = self._data_by_asset[asset].loc[self.current_time, 'Close']
            for trade in position.active_trades:
                total += trade.size * trade.multiplier * (current_price - trade.entry_price)
        return total

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as a percentage of total initial margin (across all assets)."""
        total_initial_margin = 0.0
        for asset, position in self._positions.items():
            mr = self._margin_ratio_by_asset[asset]
            total_initial_margin += sum(
                abs(trade.size) * trade.multiplier * trade.entry_price * mr
                for trade in position.active_trades
            )
        return self.unrealized_pnl / total_initial_margin * 100 if total_initial_margin != 0 else 0

    @property
    def realized_pnl(self) -> float:
        """Sum of realized P&L from all closed trades across all assets."""
        total = 0.0
        for position in self._positions.values():
            for trade in position.closed_trades:
                if trade.profit is not None:
                    total += trade.profit
        return total

    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """Closed trades across all assets (concatenated; per-asset access via :attr:`positions`)."""
        result: list[Trade] = []
        for position in self._positions.values():
            result.extend(position.closed_trades)
        return tuple(result)

    @property
    def filled_orders(self) -> tuple[Order, ...]:
        """Get a tuple of all filled orders."""
        return tuple(self._filled_orders)

    @property
    def closed_orders(self) -> tuple[Order, ...]:
        """Get a tuple of all closed orders."""
        return tuple(self._closed_orders)

    def get_trade_history(self) -> pd.DataFrame:
        """Trade-by-trade DataFrame across all assets."""
        trades = self.closed_trades
        return pd.DataFrame({
            'Asset': [t.asset for t in trades],
            'Type': ['Long' if t.is_long else 'Short' for t in trades],
            'Size': [t.size for t in trades],
            'Entry Price': [t.entry_price for t in trades],
            'Exit Price': [t.exit_price for t in trades],
            'Entry Time': [t.entry_date for t in trades],
            'Exit Date': [t.exit_date for t in trades],
            'Profit': [t.profit for t in trades],
            'Tag': [t.tag for t in trades],
            'Exit Reason': [t.exit_reason for t in trades],
            'Duration': [t.exit_date - t.entry_date if t.exit_date else None for t in trades],
        })

    @property
    def equity_history(self) -> pd.Series:
        """Get a copy of the equity history (portfolio-level)."""
        return self._equity_history.copy()

    # ------------------------------------------------------------------
    # Order placement and bar processing.
    # ------------------------------------------------------------------

    def place_orders(self, orders: Order | list[Order]) -> None:
        """
        Submit one or multiple orders.

        Args:
            orders (Union[Order, List[Order]]): A single order or a list of orders.

        Raises:
            TypeError: If orders is neither an Order instance nor a list of Orders.
        """
        if isinstance(orders, list):
            if not all(isinstance(order, Order) for order in orders):
                raise TypeError("All elements must be instances of Order.")
            new_orders = orders
        elif isinstance(orders, Order):
            new_orders = [orders]
        else:
            raise TypeError("orders must be an Order instance or a list of Orders.")

        for order in new_orders:
            if order._stop or order._limit:
                self._pending_orders.append(order)
            else:
                if self.trade_on_close:
                    fill_date = self.current_time
                    fill_price = self._data_by_asset[order.asset].loc[fill_date, 'Close']
                    self.__process_order(order, fill_price, fill_date)
                else:
                    self._executing_orders.append(order)
        self.__update_account_value_history()

    def process_bar(self, current_time: pd.Timestamp) -> None:
        """Process the trading logic for the current bar."""
        self.current_time = current_time
        self.__remove_closed_orders()
        self.__process_executing_orders()
        self.__check_sl_tp()
        # Trail update happens AFTER SL/TP check so the tightened stop takes
        # effect on the NEXT bar. Within a single OHLC bar we don't know
        # whether the low (long) / high (short) came before or after the
        # extreme — assuming the unfavorable extreme came first is the
        # conservative choice (matches the bar-level convention used
        # elsewhere in the engine).
        self.__update_trailing_stops()
        self.__process_pending_orders()
        self.__update_account_value_history()

    def __update_account_value_history(self) -> None:
        self._equity_history.loc[self.current_time] = self.equity

    def __remove_closed_orders(self) -> None:
        """
        Drop canceled/rejected orders from the pending and executing queues
        before they reach a fill attempt.
        """
        self._closed_orders.extend(order for order in self._pending_orders if order.is_closed)
        self._pending_orders = [order for order in self._pending_orders if not order.is_closed]
        self._closed_orders.extend(order for order in self._executing_orders if order.is_closed)
        self._executing_orders = [order for order in self._executing_orders if not order.is_closed]

    def __process_executing_orders(self) -> None:
        """Execute orders that were queued for fill at the next bar's open price."""
        for order in self._executing_orders:
            fill_date = self.current_time
            fill_price = self._data_by_asset[order.asset].loc[fill_date, 'Open']
            self.__process_order(order, fill_price, fill_date)
        self._executing_orders.clear()

    def __process_pending_orders(self) -> None:
        """Process pending orders, including stop and limit orders."""
        orders_to_remove = []
        for order in self._pending_orders:
            df = self._data_by_asset[order.asset]
            bar_open = df.loc[self.current_time, 'Open']
            high = df.loc[self.current_time, 'High']
            low = df.loc[self.current_time, 'Low']

            # Remember the stop level before we clear it — used as the fill
            # price for pure-stop orders that just triggered.
            stop_price = order._stop

            # Check stop conditions
            if order._stop:
                is_stop_triggered = high >= order._stop if order.is_long else low <= order._stop
                if is_stop_triggered:
                    order._stop = None  # Reset stop to prevent multiple triggers
                else:
                    continue  # Stop not triggered, skip to next order

            # Check limit conditions
            if order._limit:
                is_limit_triggered = low < order._limit if order.is_long else high > order._limit
                if is_limit_triggered:
                    fill_date = self.current_time
                    fill_price = order._limit
                    self.__process_order(order, fill_price, fill_date)
                    orders_to_remove.append(order)
                else:
                    continue  # Limit not triggered, skip to next order
            else:
                # Pure stop order just triggered (no limit). Conservatively
                # fill at the trigger price — but if the market gapped past
                # the stop on the open, use the open price so the simulation
                # reflects the worse fill that real execution would see:
                #   long stop  → max(stop, open) (higher = pays more)
                #   short stop → min(stop, open) (lower  = receives less)
                # Add a SlippageCommission on top to model further slippage.
                if self.trade_on_close:
                    fill_date = self.current_time
                    if order.is_long:
                        fill_price = max(stop_price, bar_open)
                    else:
                        fill_price = min(stop_price, bar_open)
                    self.__process_order(order, fill_price, fill_date)
                    orders_to_remove.append(order)
                else:
                    # trade_on_close=False — defer to next bar's open like a
                    # plain market order.
                    self._executing_orders.append(order)

        for order in orders_to_remove:
            self._pending_orders.remove(order)

    def __process_order(self, order: Order, fill_price: float, fill_date: pd.Timestamp) -> None:
        """Handle the execution of a filled order against the order's asset."""
        if not self.__is_margin_sufficient(order, fill_price):
            order._close(reason="Insufficient margin")
            logging.info(f"Order rejected: {order._close_reason}")
            self._closed_orders.append(order)
            return

        position = self._positions[order.asset]
        remaining_order_size = order.size
        commission_cost = self.commission.calculate_commission(order.size, fill_price) if self.commission else 0
        self.cash -= commission_cost

        for trade in position.active_trades:
            if trade.is_long == order.is_long:
                continue  # Skip trades on the same side

            if abs(remaining_order_size) >= abs(trade.size):
                # Fully close the trade
                closed_trade = self._close_trade(
                    trade=trade,
                    exit_price=fill_price,
                    exit_date=fill_date,
                    exit_reason='signal',
                )
                self.cash += closed_trade.profit
                remaining_order_size += closed_trade.size  # Adjust remaining size
            else:
                # Partially close the trade
                closed_trade = self._close_trade(
                    trade=trade,
                    close_size=-remaining_order_size,
                    exit_price=fill_price,
                    exit_date=fill_date,
                    exit_reason='signal',
                )
                self.cash += closed_trade.profit
                remaining_order_size = 0  # Order fully filled

            if remaining_order_size == 0:
                break

        # Drop fully-closed trades
        position._active_trades = [t for t in position.active_trades if t.size != 0]

        if remaining_order_size != 0:
            self._open_trade(
                entry_price=fill_price,
                entry_date=fill_date,
                size=remaining_order_size,
                sl=order._sl,
                tp=order._tp,
                trail_percent=order._trail_percent,
                trail_amount=order._trail_amount,
                tag=order.tag,
                asset=order.asset,
            )

        order._fill(fill_price, fill_date)
        self._filled_orders.append(order)

    def __is_margin_sufficient(self, order: Order, fill_price: float) -> bool:
        """Check if there's enough portfolio margin to take on the order on its asset."""
        order_mult = self._multiplier_by_asset[order.asset]
        order_mr = self._margin_ratio_by_asset[order.asset]
        position = self._positions[order.asset]
        new_position_size = position.size + order.size
        new_margin = abs(new_position_size) * order_mult * fill_price * order_mr

        # Account value uses the prospective fill_price for the order's asset
        # and current Close for everyone else.
        unrealized_pnl = 0.0
        for asset, pos in self._positions.items():
            price = fill_price if asset == order.asset else self._data_by_asset[asset].loc[self.current_time, 'Close']
            for trade in pos.active_trades:
                unrealized_pnl += trade.size * trade.multiplier * (price - trade.entry_price)
        account_value = self.cash + unrealized_pnl

        # Other assets' used margin counts against the available pool.
        other_used_margin = 0.0
        for asset, pos in self._positions.items():
            if asset == order.asset:
                continue
            price = self._data_by_asset[asset].loc[self.current_time, 'Close']
            mr = self._margin_ratio_by_asset[asset]
            other_used_margin += sum(
                abs(t.size) * t.multiplier * price * mr for t in pos.active_trades
            )

        return account_value >= new_margin + other_used_margin

    def __update_trailing_stops(self) -> None:
        """Bump trailing-stop SLs for every open trade based on this bar's range."""
        for asset, position in self._positions.items():
            df = self._data_by_asset[asset]
            high = df.loc[self.current_time, 'High']
            low = df.loc[self.current_time, 'Low']
            for trade in position.active_trades:
                trade._update_trailing_stop(high, low)

    def __check_sl_tp(self) -> None:
        """Check and apply stop loss / take profit conditions across every asset."""
        for asset, position in self._positions.items():
            df = self._data_by_asset[asset]
            high = df.loc[self.current_time, 'High']
            low = df.loc[self.current_time, 'Low']

            for trade in position.active_trades:
                if not trade.sl and not trade.tp:
                    continue

                sl = trade.sl
                tp = trade.tp

                if trade.is_long:
                    if sl is not None and low <= sl:
                        self.__execute_trade_exit(trade, sl, 'sl')
                    elif tp is not None and high >= tp:
                        self.__execute_trade_exit(trade, tp, 'tp')
                else:
                    if sl is not None and high >= sl:
                        self.__execute_trade_exit(trade, sl, 'sl')
                    elif tp is not None and low <= tp:
                        self.__execute_trade_exit(trade, tp, 'tp')

            position._active_trades = [t for t in position.active_trades if t.size != 0]

    def __execute_trade_exit(self, trade: Trade, exit_price: float, exit_reason: str) -> None:
        """Execute the exit of a trade due to SL or TP."""
        commission_cost = self.commission.calculate_commission(trade.size, exit_price) if self.commission else 0
        self.cash -= commission_cost

        closed_trade = self._close_trade(
            trade=trade,
            exit_price=exit_price,
            exit_date=self.current_time,
            exit_reason=exit_reason
        )
        # The exit is in the opposite direction of the trade itself
        # (closing a long is a sell; closing a short is a buy).
        sl_tp_order = Order(size=-closed_trade.size, tag=exit_reason, asset=trade.asset)
        sl_tp_order._fill(exit_price, self.current_time)
        self._filled_orders.append(sl_tp_order)
        self.cash += closed_trade.profit

    def close_all_positions(self) -> None:
        """Close all open positions across every asset at the current bar's close."""
        for asset, position in self._positions.items():
            price = self._data_by_asset[asset].loc[self.current_time, 'Close']
            for trade in position.active_trades:
                commission_cost = self.commission.calculate_commission(
                    abs(trade.size), price
                ) if self.commission else 0
                self.cash -= commission_cost

                closed_trade = self._close_trade(
                    trade=trade,
                    exit_price=price,
                    exit_date=self.current_time,
                    exit_reason='end'
                )
                self.cash += closed_trade.profit

            position._active_trades = [t for t in position.active_trades if t.size != 0]
        self.__update_account_value_history()

    def _open_trade(
            self,
            entry_price: float,
            entry_date: pd.Timestamp,
            size: int,
            sl: float | None = None,
            tp: float | None = None,
            trail_percent: float | None = None,
            trail_amount: float | None = None,
            tag: object | None = None,
            asset: str = "default",
        ) -> None:
        """Open a new trade position on the given asset."""
        df = self._data_by_asset[asset]
        new_trade = Trade(
            entry_price=entry_price,
            entry_date=entry_date,
            entry_index=df.index.get_loc(entry_date),
            size=size,
            sl=sl,
            tp=tp,
            trail_percent=trail_percent,
            trail_amount=trail_amount,
            tag=tag,
            asset=asset,
            multiplier=self._multiplier_by_asset[asset],
        )
        self._positions[asset]._active_trades.append(new_trade)

    def _close_trade(
            self,
            trade: Trade,
            exit_price: float,
            exit_date: pd.Timestamp,
            exit_reason: str,
            close_size: int | None = None
        ) -> Trade:
        """Close an active trade and append it to the closed-trades list of its asset."""
        df = self._data_by_asset[trade.asset]
        closed_trade = trade.close(
            size=close_size,
            exit_price=exit_price,
            exit_date=exit_date,
            exit_index=df.index.get_loc(exit_date),
            exit_reason=exit_reason
        )
        self._positions[trade.asset]._closed_trades.append(closed_trade)
        return closed_trade
