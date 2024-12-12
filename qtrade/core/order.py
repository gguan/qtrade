# components/order.py

from typing import Optional
import pandas as pd


class Order:
    def __init__(self,
                 size: int,
                 limit: Optional[float] = None,
                 stop: Optional[float] = None,
                 sl: Optional[float] = None,
                 tp: Optional[float] = None,
                 tag: Optional[object] = None):
        """
        Initialize an order.

        :param size: Order size (positive for buy, negative for sell)
        :param limit: Limit price for limit orders
        :param stop: Stop price for stop orders
        :param sl: Stop loss price
        :param tp: Take profit price
        :param tag: Order tag for identification
        """
        assert size != 0, 'Order size cannot be zero'
        self._size = size
        self._limit = limit
        self._stop = stop
        self._sl = sl
        self._tp = tp
        self._tag = tag
        self._is_filled = False
        self._fill_price: Optional[float] = None
        self._fill_date: Optional[pd.Timestamp] = None
        self._reject_reason: Optional[str] = None

    def fill(self, fill_price: float, fill_date: pd.Timestamp):
        """
        Fill the order.

        :param fill_price: Price at which the order is filled
        :param fill_date: Date when the order is filled
        """
        if self._is_filled:
            raise ValueError("Order is already filled.")
        self._is_filled = True
        self._fill_price = fill_price
        self._fill_date = fill_date

    def reject(self, reason: str):
        self._reject_reason = reason

    @property
    def size(self) -> int:
        """Return the order size."""
        return self._size

    @property
    def limit(self) -> Optional[float]:
        """Return the limit price."""
        return self._limit

    @property
    def stop(self) -> Optional[float]:
        """Return the stop price."""
        return self._stop

    @property
    def sl(self) -> Optional[float]:
        """Return the stop loss price."""
        return self._sl

    @property
    def tp(self) -> Optional[float]:
        """Return the take profit price."""
        return self._tp

    @property
    def tag(self) -> Optional[object]:
        """Return the order tag."""
        return self._tag

    @property
    def is_long(self) -> bool:
        """True if the order is a long position (size > 0)."""
        return self._size > 0

    @property
    def is_short(self) -> bool:
        """True if the order is a short position (size < 0)."""
        return self._size < 0
    
    @property
    def is_filled(self) -> bool:
        """True if the order is filled."""
        return self._is_filled
    
    @property
    def fill_price(self) -> Optional[float]:
        """Return the fill price."""
        return self._fill_price
    
    @property
    def fill_date(self) -> Optional[pd.Timestamp]:
        """Return the fill date."""
        return self._fill_date

    def __repr__(self) -> str:
        """
        Return a string representation of the order.

        :return: String representation of the order
        """
        params = (
            ('Size', self._size),
            ('Limit', self._limit),
            ('Stop', self._stop),
            ('Sl', self._sl),
            ('Tp', self._tp),
            ('Tag', self.tag),
        )
        param_str = ', '.join(f'{param}={value}' for param, value in params if value is not None)
        return f'<Order {param_str}>'
