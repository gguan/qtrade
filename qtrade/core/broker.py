# components/broker.py

import logging
from typing import Optional, Union

import pandas as pd

from .trade import Trade
from .order import Order
from .position import Position
from .commission import Commission


class Broker:
    """
    The Broker class is responsible for executing orders and managing positions.

    Attributes:
        data (pd.DataFrame): DataFrame containing market data with columns ['Open', 'High', 'Low', 'Close'].
        cash (float): Current cash balance in the account.
        commission (Optional[Commission]): Instance for calculating trade commissions. If None, no commission is applied.
        margin_ratio (float): Margin ratio (0 < margin_ratio ≤ 1).
        trade_on_close (bool): If True, orders are filled at the current close price. Otherwise, at the next open price.
        position (Position): Current position information.
        current_time (pd.Timestamp): Timestamp of the current bar.
        _new_orders (List[Order]): Orders submitted in the current bar.
        _pending_orders (List[Order]): Pending orders (e.g., stop/limit orders) awaiting execution.
        _executing_orders (List[Order]): Orders to be executed at the next bar's open price if trade_on_close is False.
        _filled_orders (List[Order]): List of filled orders.
        _rejected_orders (List[Order]): List of rejected orders.
        _equity_history (pd.Series): Historical record of account equity.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cash: float,
        commission: Optional[Commission],
        margin_ratio: float,
        trade_on_close: bool
    ):
        """
        Initialize the Broker with market data and account settings.

        Args:
            data (pd.DataFrame): Market data with ['open', 'high', 'low', 'close'] columns.
            cash (float): Initial cash balance. Must be positive.
            commission (Optional[Commission]): Commission calculator instance.
            margin_ratio (float): Margin ratio (0 < margin_ratio ≤ 1).
            trade_on_close (bool): Execution mode for orders.

        Raises:
            AssertionError: If cash is not positive or margin_ratio is out of bounds.
        """
        assert cash > 0, "Initial cash must be positive."
        assert 0 < margin_ratio <= 1, "Margin ratio must be between 0 and 1."
        
        common_names = {
            "Date": "date",
            "Time": "time",
            "Timestamp": "timestamp",
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        data.rename(columns=common_names, errors="ignore", inplace=True)

        self.data = data
        self.cash = cash
        self.commission = commission
        self.margin_ratio = margin_ratio
        self.trade_on_close = trade_on_close
        self.position = Position()

        self.current_time = data.index[0]

        self._new_orders: list[Order] = []
        self._pending_orders: list[Order] = []
        self._executing_orders: list[Order] = []

        self._filled_orders: list[Order] = []
        self._closed_orders: list[Order] = []  # Rejected and canceled orders

        self._equity_history = pd.Series(data=self.cash, index=data.index).astype('float64')

    @property
    def equity(self) -> float:
        """
        Calculate the current equity of the account.

        Returns:
            float: Current equity (cash + unrealized P&L).
        """
        return self.cash + self.unrealized_pnl

    @property
    def cumulative_returns(self) -> float:
        """
        Calculate the cumulative returns of the account.

        Returns:
            float: Cumulative returns (equity / initial equity).
        """
        return self.equity / self._equity_history.iloc[0]

    @property
    def available_margin(self) -> float:
        """
        Calculate the available margin for new trades.

        Returns:
            float: Available margin, minimum 0.
        """
        current_price = self.data.loc[self.current_time, 'close']
        used_margin = sum(
            abs(trade.size) * current_price * self.margin_ratio
            for trade in self.position.active_trades
        )
        return max(0, self.equity - used_margin)

    @property
    def unrealized_pnl(self) -> float:
        """
        Calculate the unrealized profit and loss.

        Returns:
            float: Sum of unrealized P&L from all active trades.
        """
        current_price = self.data.loc[self.current_time, 'close']
        return sum(
            trade.size * (current_price - trade.entry_price) for trade in self.position.active_trades
        )

    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """
        Get a tuple of all closed trades.

        Returns:
            Tuple[Trade, ...]: Closed trades.
        """
        return self.position.closed_trades

    @property
    def filled_orders(self) -> tuple[Order, ...]:
        """
        Get a tuple of all filled orders.

        Returns:
            Tuple[Order, ...]: Filled orders.
        """
        return tuple(self._filled_orders)

    @property
    def closed_orders(self) -> tuple[Order, ...]:
        """
        Get a tuple of all closed orders.

        Returns:
            Tuple[Order, ...]: Closed orders.
        """
        return tuple(self._closed_orders)

    @property
    def equity_history(self) -> pd.Series:
        """
        Get a copy of the equity history.

        Returns:
            pd.Series: Historical equity values.
        """
        return self._equity_history.copy()

    def place_orders(self, orders: Union[Order, list[Order]]) -> None:
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
                    fill_price = self.data.loc[fill_date, 'close']
                    self.__process_order(order, fill_price, fill_date)
                else:
                    self._executing_orders.append(order)
        self.__update_account_value_history()

    def process_bar(self, current_time: pd.Timestamp) -> None:
        """
        Process the trading logic for the current bar.

        Args:
            current_time (pd.Timestamp): Timestamp of the current bar.
        """
        self.current_time = current_time
        self.__remove_closed_orders()
        self.__process_executing_orders()
        self.__check_sl_tp()
        self.__process_pending_orders()
        self.__update_account_value_history()

    def __update_account_value_history(self) -> None:
        """
        Update the historical record of account equity.
        """
        self._equity_history.loc[self.current_time] = self.equity

    def __remove_closed_orders(self) -> None:
        """
        Remove invalid orders (rejected and canceled) from the new orders list.
        """
        self._closed_orders.extend(order for order in self._pending_orders if order.is_closed)
        self._pending_orders = [order for order in self._pending_orders if not order.is_closed]

    def __process_executing_orders(self) -> None:
        """
        Execute orders that are set to be filled at the next bar's open price.
        """
        for order in self._executing_orders:
            fill_date = self.current_time
            fill_price = self.data.loc[fill_date, 'open']
            self.__process_order(order, fill_price, fill_date)
        self._executing_orders.clear()

    def __process_pending_orders(self) -> None:
        """
        Process pending orders, including stop and limit orders.
        """
        high = self.data.loc[self.current_time, 'high']
        low = self.data.loc[self.current_time, 'low']

        orders_to_remove = []
        for order in self._pending_orders:
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
                # Market order
                if self.trade_on_close:
                    fill_date = self.current_time
                    fill_price = self.data.loc[fill_date, 'close']
                    self.__process_order(order, fill_price, fill_date)
                    orders_to_remove.append(order)
                else:
                    self._executing_orders.append(order)

        # Remove processed orders from pending_orders
        for order in orders_to_remove:
            self._pending_orders.remove(order)

    def __process_order(self, order: Order, fill_price: float, fill_date: pd.Timestamp) -> None:
        """
        Handle the execution of a filled order.

        Args:
            order (Order): The order to process.
            fill_price (float): The price at which the order was filled.
            fill_date (pd.Timestamp): The date when the order was filled.
        """
        if not self.__is_margin_sufficient(order, fill_price):
            # Insufficient margin, reject the order
            order._close(reason="Insufficient margin")
            logging.info(f"Order rejected: {order._close_reason}")
            self._closed_orders.append(order)
            return

        remaining_order_size = order.size
        commission_cost = self.commission.calculate_commission(order.size, fill_price) if self.commission else 0
        self.cash -= commission_cost

        for trade in self.position.active_trades:
            if trade.is_long == order.is_long:
                continue  # Skip trades on the same side

            if abs(remaining_order_size) >= abs(trade.size):
                # Fully close the trade
                closed_trade = self._close_trade(
                    trade = trade, 
                    exit_price = fill_price,
                    exit_date = fill_date,
                    exit_reason='signal',
                )
                self.cash += closed_trade.profit
                remaining_order_size += closed_trade.size  # Adjust remaining size
            else:
                # Partially close the trade
                closed_trade = self._close_trade(
                    trade = trade,
                    close_size = -remaining_order_size,
                    exit_price = fill_price,
                    exit_date = fill_date,
                    exit_reason='signal',
                )
                self.cash += closed_trade.profit
                remaining_order_size = 0  # Order fully filled

            if remaining_order_size == 0:
                break  # Order fully filled

        # Remove trades with zero size
        self.position._active_trades = [
            trade for trade in self.position.active_trades if trade.size != 0
        ]

        if remaining_order_size != 0:
            # Open a new position with the remaining order size
            self._open_trade(
                entry_price = fill_price,
                entry_date = fill_date,
                size = remaining_order_size,
                sl = order._sl,
                tp = order._tp,
                tag = order.tag
            )

        # Record the filled order
        order._fill(fill_price, fill_date)
        self._filled_orders.append(order)

    def __is_margin_sufficient(self, order: Order, fill_price: float) -> bool:
        """
        Check if there is sufficient margin to execute the order.

        Args:
            order (Order): The order to check.
            fill_price (float): The price at which the order will be filled.

        Returns:
            bool: True if there is sufficient margin, False otherwise.
        """
        new_position_size = self.position.size + order.size
        new_margin = abs(new_position_size) * fill_price * self.margin_ratio

        # Calculate unrealized P&L considering the fill price
        unrealized_pnl = sum(
            trade.size * (fill_price - trade.entry_price) 
            for trade in self.position.active_trades
        )
        account_value = self.cash + unrealized_pnl

        return account_value >= new_margin

    def __check_sl_tp(self) -> None:
        """
        Check and apply stop loss (SL) and take profit (TP) conditions for all active trades.
        """
        high = self.data.loc[self.current_time, 'high']
        low = self.data.loc[self.current_time, 'low']

        for trade in self.position.active_trades:
            if not trade.sl and not trade.tp:
                continue  # No SL/TP set for this trade

            sl = trade.sl
            tp = trade.tp

            if trade.is_long:
                # For long positions
                if sl is not None and low <= sl:
                    # Stop loss triggered
                    self.__execute_trade_exit(trade, sl, 'sl')
                elif tp is not None and high >= tp:
                    # Take profit triggered
                    self.__execute_trade_exit(trade, tp, 'tp')
            else:
                # For short positions
                if sl is not None and high >= sl:
                    # Stop loss triggered
                    self.__execute_trade_exit(trade, sl, 'sl')
                elif tp is not None and low <= tp:
                    # Take profit triggered
                    self.__execute_trade_exit(trade, tp, 'tp')

        # Remove trades with zero size
        self.position._active_trades = [
            trade for trade in self.position.active_trades if trade.size != 0
        ]

    def __execute_trade_exit(self, trade: Trade, exit_price: float, exit_reason: str) -> None:
        """
        Execute the exit of a trade due to SL or TP.

        Args:
            trade (Trade): The trade to exit.
            exit_price (float): The price at which to exit the trade.
            exit_reason (str): Reason for exit ('sl' or 'tp').
        """
        commission_cost = self.commission.calculate_commission(trade.size, exit_price) if self.commission else 0
        self.cash -= commission_cost

        closed_trade = self._close_trade(
            trade=trade,
            exit_price=exit_price,
            exit_date=self.current_time,
            exit_reason=exit_reason
        )
        sl_tp_order = Order(size=closed_trade.size, tag=exit_reason)
        sl_tp_order._fill(exit_price, self.current_time)
        self._filled_orders.append(sl_tp_order)
        self.cash += closed_trade.profit
        
    def close_all_positions(self) -> None:
        """
        Close all open positions at the end of the trading period.
        """
        price = self.data.loc[self.current_time, 'close']
        for trade in self.position.active_trades:
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

        # Remove trades with zero size
        self.position._active_trades = [
            trade for trade in self.position.active_trades if trade.size != 0
        ]
        self.__update_account_value_history()
        
    def _open_trade(
            self, 
            entry_price: float, 
            entry_date: pd.Timestamp, 
            size: int, 
            sl: Optional[float] = None, 
            tp: Optional[float] = None, 
            tag: Optional[object] = None
        ) -> None:
        """
        Open a new trade position with the specified parameters.

        Args:
            size (int): Trade size.
            entry_price (float): Entry price.
            sl (Optional[float]): Stop loss price.
            tp (Optional[float]): Take profit price.
            tag (object): Custom tag for the trade.
        """
        """Opens a new trade position with the specified parameters."""
        new_trade = Trade(
            entry_price=entry_price,
            entry_date=entry_date,
            size=size,
            sl=sl,
            tp=tp,
            tag=tag
        )
        self.position._active_trades.append(new_trade)

    def _close_trade(
            self, 
            trade: Trade, 
            exit_price: float, 
            exit_date: pd.Timestamp, 
            exit_reason: str, 
            close_size: Optional[int] = None
        ) -> Trade:
        """
        Close an active trade and move it to the closed trades list.

        Args:
            trade (Trade): Trade to close.
            close_size (int): Size to close.
            exit_price (float): Price at which the trade is closed.
            exit_date (pd.Timestamp): Date when the trade is closed.
            exit_reason (str): Reason for closing ('signal', 'sl', 'tp', 'end').
        """
        closed_trade = trade.close(
            size=close_size,
            exit_price=exit_price,
            exit_date=exit_date,
            exit_reason=exit_reason
        )
        self.position._closed_trades.append(closed_trade)
        return closed_trade