# components/order.py

from typing import Optional
import pandas as pd


class Order:
    """
    Represents a trading order with optional limit, stop, stop loss, and take profit prices.

    Attributes:
        size (int): Order size (positive for buy, negative for sell).
        limit (Optional[float]): Limit price for limit orders.
        stop (Optional[float]): Stop price for stop orders.
        sl (Optional[float]): Stop loss price.
        tp (Optional[float]): Take profit price.
        tag (Optional[object]): Tag for identifying the order.
        is_filled (bool): Indicates if the order has been filled.
        fill_price (Optional[float]): Price at which the order was filled.
        fill_date (Optional[pd.Timestamp]): Date when the order was filled.
        reject_reason (Optional[str]): Reason for order rejection, if any.
    """

    def __init__(
        self,
        size: int,
        limit: Optional[float] = None,
        stop: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        tag: Optional[object] = None
    ):
        """
        Initialize an Order instance.

        Args:
            size (int): Order size (positive for buy, negative for sell).
            limit (Optional[float], optional): Limit price for limit orders. Defaults to None.
            stop (Optional[float], optional): Stop price for stop orders. Defaults to None.
            sl (Optional[float], optional): Stop loss price. Defaults to None.
            tp (Optional[float], optional): Take profit price. Defaults to None.
            tag (Optional[object], optional): Tag for identification. Defaults to None.

        Raises:
            AssertionError: If the order size is zero.
        """
        assert size != 0, 'Order size cannot be zero.'

        self._size: int = size
        self._limit: Optional[float] = limit
        self._stop: Optional[float] = stop
        self._sl: Optional[float] = sl
        self._tp: Optional[float] = tp
        self._tag: Optional[object] = tag

        self._is_filled: bool = False
        self._fill_price: Optional[float] = None
        self._fill_date: Optional[pd.Timestamp] = None
        self._close_reason: Optional[str] = None

    def _fill(self, fill_price: float, fill_date: pd.Timestamp) -> None:
        """
        Mark the order as filled with the given price and date.

        Args:
            fill_price (float): Price at which the order was filled.
            fill_date (pd.Timestamp): Date when the order was filled.

        Raises:
            ValueError: If the order already filled.
        """
        if self._is_filled:
            raise ValueError("Order already filled.")
        
        if self.is_closed:
            raise ValueError("Order already closed.")

        self._is_filled = True
        self._fill_price = fill_price
        self._fill_date = fill_date

    def _close(self, reason: str) -> None:
        """
        Close the order with a given reason.

        Args:
            reason (str): Reason for rejection.
        """
        if self._is_filled:
            raise ValueError("Order already filled.")
        if self.is_closed:
            raise ValueError("Order already closed.")
        self._close_reason = reason

    def cancel(self) -> None:
        """Cancel the order."""
        self._close("Order canceled.")

    @property
    def size(self) -> int:
        """int: Order size."""
        return self._size

    @property
    def limit(self) -> Optional[float]:
        """Optional[float]: Limit price."""
        return self._limit

    @property
    def stop(self) -> Optional[float]:
        """Optional[float]: Stop price."""
        return self._stop

    @property
    def sl(self) -> Optional[float]:
        """Optional[float]: Stop loss price."""
        return self._sl

    @property
    def tp(self) -> Optional[float]:
        """Optional[float]: Take profit price."""
        return self._tp

    @property
    def tag(self) -> Optional[object]:
        """Optional[object]: Order tag."""
        return self._tag

    @property
    def is_long(self) -> bool:
        """bool: True if the order is a long position."""
        return self._size > 0

    @property
    def is_short(self) -> bool:
        """bool: True if the order is a short position."""
        return self._size < 0

    @property
    def is_filled(self) -> bool:
        """bool: Indicates if the order is filled."""
        return self._is_filled

    @property
    def fill_price(self) -> Optional[float]:
        """Optional[float]: Fill price."""
        return self._fill_price

    @property
    def fill_date(self) -> Optional[pd.Timestamp]:
        """Optional[pd.Timestamp]: Fill date."""
        return self._fill_date
    
    @property
    def is_closed(self) -> bool:
        """bool: Indicates if the order is canceled or rejected."""
        return self._close_reason is not None

    def __repr__(self) -> str:
        params = (
            ('Size', self._size),
            ('Limit', self._limit),
            ('Stop', self._stop),
            ('Sl', self._sl),
            ('Tp', self._tp),
            ('Tag', self.tag),
        )
        param_str = ', '.join(f'{name}={value}' for name, value in params if value is not None)
        return f'<Order {param_str}>'