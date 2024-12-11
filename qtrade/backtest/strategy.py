
# Strategy base class
from abc import ABC, abstractmethod
from typing import Optional, Tuple
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
    def next(self):
        """
        Called on each bar (time step) to generate trading signals.
        """
        pass

    def buy(self, *,
            size: int = 1,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None):
        """
        Place a buy order.

        :param size: Order size
        :param limit: Limit price
        :param stop: Stop price
        :param sl: Stop loss price
        :param tp: Take profit price
        :param tag: Order tag
        """
        order = Order(size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag)
        self._broker.new_order(order)

    def sell(self, *,
             size: int = 1,
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
        order = Order(-size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag)
        self._broker.new_order(order)

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
    def account_value(self) -> float:
        """
        Get the current account value.

        """
        return self._broker.account_value
    
    @property
    def unrealized_pnl(self) -> float:
        """
        Get the current unrealized profit/loss.

        """
        return self._broker.unrealized_pnl

    @property
    def active_trades(self) -> Tuple[Trade, ...]:
        """
        Get the active trades.

        """
        return tuple(self._broker.position.active_trades())
    
    @property
    def closed_trades(self) -> Tuple[Trade, ...]:
        """
        Get the closed trades.

        """
        return tuple(self._broker.position.closed_trades())
    
    @property
    def pending_orders(self) -> Tuple[Order, ...]:
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