
# Strategy base class

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

from qtrade.core import Order, Trade, Position


class Strategy(ABC):
    """Base class for trading strategies.

    Single-asset strategies use the legacy attribute-style API
    (``self.data``, ``self.position``, ``self.buy(size=10)``); multi-asset
    strategies use ``self.data_by_asset[asset]``, ``self.positions[asset]``,
    and pass an asset symbol to ``self.buy("AAPL", size=10)``.
    """

    def __init__(self, broker, data, params):
        """
        Initialize the strategy.

        Args:
            broker: The Broker instance.
            data: Either a single DataFrame (single-asset, kept under the key
                ``"default"``) or a dict[str, DataFrame] (multi-asset).
            params: Strategy parameters dict; entries are also bound as attributes.
        """
        self._broker = broker
        if isinstance(data, pd.DataFrame):
            self._data: dict[str, pd.DataFrame] = {"default": data.copy(deep=True)}
        else:
            self._data = {asset: df.copy(deep=True) for asset, df in data.items()}
        self._params = params
        for key, value in params.items():
            setattr(self, key, value)

    @abstractmethod
    def prepare(self):
        """Initialize the strategy (e.g., declare indicators)."""
        pass

    @abstractmethod
    def on_bar_close(self):
        """Called on each bar (time step) to generate trading signals."""
        pass

    # ------------------------------------------------------------------
    # Order placement.
    # ------------------------------------------------------------------

    def buy(self,
            asset: str | None = None,
            *,
            size: int | None = None,
            limit: float | None = None,
            stop: float | None = None,
            sl: float | None = None,
            tp: float | None = None,
            tag: object = None):
        """
        Place a buy order.

        Args:
            asset: Asset symbol (positional). Required when the strategy is
                running on more than one asset; optional otherwise.
            size: Order size. If omitted, defaults to the maximum size that
                fits in current available margin at the most recent close.
            limit / stop / sl / tp / tag: standard order fields.
        """
        asset = self._resolve_asset(asset)
        if size is None:
            size = self._broker.available_margin // self._data[asset]['Close'].loc[self._broker.current_time]
        else:
            size = abs(size)  # buy() always opens long, treat user-provided size as magnitude
        order = Order(size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag, asset=asset)
        self._broker.place_orders(order)

    def sell(self,
             asset: str | None = None,
             *,
             size: int | None = None,
             limit: float | None = None,
             stop: float | None = None,
             sl: float | None = None,
             tp: float | None = None,
             tag: object = None):
        """
        Place a sell order.

        Args:
            asset: Asset symbol (positional). Required when the strategy is
                running on more than one asset; optional otherwise.
            size: Order size. If omitted, defaults to the current position
                size on that asset (used for closing longs).
            limit / stop / sl / tp / tag: standard order fields.
        """
        asset = self._resolve_asset(asset)
        if size is None:
            size = self._broker.positions[asset].size
        else:
            size = abs(size)  # sell() always opens short, treat user-provided size as magnitude
        order = Order(-size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag, asset=asset)
        self._broker.place_orders(order)

    def close(self, asset: str | None = None):
        """
        Close all open positions.

        Args:
            asset: If provided, close only that asset's position. Otherwise
                close every asset's position.
        """
        if asset is None:
            for a in self.assets:
                self._close_one(a)
        else:
            self._close_one(asset)

    def _close_one(self, asset: str) -> None:
        size = self._broker.positions[asset].size
        if size > 0:
            self.sell(asset, size=size, tag='close')
        elif size < 0:
            self.buy(asset, size=-size, tag='close')

    def _resolve_asset(self, asset: str | None) -> str:
        if asset is not None:
            return asset
        assets = self.assets
        if len(assets) > 1:
            raise ValueError(
                f"Strategy is running on multiple assets {assets}; "
                "pass the asset symbol explicitly, e.g., self.buy('AAPL', size=10)."
            )
        return assets[0]

    # ------------------------------------------------------------------
    # Read accessors.
    # ------------------------------------------------------------------

    @property
    def assets(self) -> list[str]:
        """List of asset symbols this strategy is running on."""
        return list(self._data.keys())

    @property
    def data(self) -> pd.DataFrame:
        """Single-asset access: market data truncated to the current time.

        Raises AttributeError in multi-asset mode; use :attr:`data_by_asset`.
        """
        if len(self._data) > 1:
            raise AttributeError(
                "Strategy has multiple assets; use self.data_by_asset[symbol] instead of self.data"
            )
        single = next(iter(self._data.values()))
        return single[:self._broker.current_time]

    @property
    def data_by_asset(self) -> dict[str, pd.DataFrame]:
        """Per-asset market data, each truncated to the current time."""
        return {a: df[:self._broker.current_time] for a, df in self._data.items()}

    @property
    def equity(self) -> float:
        """Current portfolio equity."""
        return self._broker.equity

    @property
    def unrealized_pnl(self) -> float:
        """Current portfolio-level unrealized profit/loss."""
        return self._broker.unrealized_pnl

    @property
    def active_trades(self) -> tuple[Trade, ...]:
        """Active trades across all assets."""
        result: list[Trade] = []
        for pos in self._broker.positions.values():
            result.extend(pos.active_trades)
        return tuple(result)

    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """Closed trades across all assets."""
        return self._broker.closed_trades

    @property
    def pending_orders(self) -> tuple[Order, ...]:
        """Pending orders (across all assets)."""
        return tuple(self._broker._pending_orders)

    @property
    def position(self) -> Position:
        """Single-asset Position. In multi-asset mode use :attr:`positions`."""
        return self._broker.position  # broker.position raises AttributeError in multi-asset mode

    @property
    def positions(self) -> dict[str, Position]:
        """Per-asset Position objects."""
        return self._broker.positions

    def __str__(self):
        params = ''
        if self._params:
            params = '(' + ', '.join(f'{k}={v}' for k, v in self._params.items()) + ')'
        return f'{self.__class__.__name__}{params}'
