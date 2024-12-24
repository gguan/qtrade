
# Strategy base class

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from qtrade.core import Broker, Order, Trade, Position


class Strategy(ABC):
    def __init__(self, broker: Broker, data: pd.DataFrame):
        """
        Initialize the strategy.

        :param data: DataFrame containing market data
        """
        self.raw_data = data
        self._broker = broker

    @abstractmethod
    def init(self):
        """
        Initialize the strategy (e.g., declare indicators).
        """
        pass

    @abstractmethod
    def on_bar_close(self):
        """
        Called on each bar (time step) to generate trading signals.
        """
        pass

    def buy(self, *,
            size: Optional[int] = None,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None):
        """
        Place a buy order.

        :param size: Order size, if not set, the size will be calculated based on the available margin
        :param limit: Limit price
        :param stop: Stop price
        :param sl: Stop loss price
        :param tp: Take profit price
        :param tag: Order tag
        """
        if size is None:
            size = self._broker.available_margin // self.data['close'].iloc[-1]
        if size == 0:
            print(self._broker.available_margin)
        order = Order(size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag)
        self._broker.place_orders(order)

    def sell(self, *,
             size: Optional[int] = None,
             limit: Optional[float] = None,
             stop: Optional[float] = None,
             sl: Optional[float] = None,
             tp: Optional[float] = None,
             tag: object = None):
        """
        Place a sell order.

        :param size: Order size
        :param limit: Limit price
        :param stop: Stop price
        :param sl: Stop loss price
        :param tp: Take profit price
        :param tag: Order tag
        """
        if size is None:
            size = self.position.size
        order = Order(-size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag)
        self._broker.place_orders(order)

    def close(self):
        """
        Close all open positions.
        """
        if self.position.size > 0:
            self.sell(size=self.position.size, tag='close')
        elif self.position.size < 0:
            self.buy(size=-self.position.size, tag='close')

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the market data, can only see data up to the current index.

        """
        return self.raw_data[:self._broker.current_time]

    @property
    def equity(self) -> float:
        """
        Get the current account value.

        """
        return self._broker.equity
    
    @property
    def unrealized_pnl(self) -> float:
        """
        Get the current unrealized profit/loss.

        """
        return self._broker.unrealized_pnl

    @property
    def active_trades(self) -> tuple[Trade, ...]:
        """
        Get the active trades.

        """
        return tuple(self._broker.position.active_trades())
    
    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """
        Get the closed trades.

        """
        return tuple(self._broker.position.closed_trades())
    
    @property
    def pending_orders(self) -> tuple[Order, ...]:
        """
        Get the pending orders.
        """
        return tuple(self._broker._pending_orders)

    @property
    def position(self) -> Position:
        """
        Get the current position.
        """
        return self._broker.position