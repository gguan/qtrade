# components/order.py

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
        is_closed (bool): Indicates if the order has been canceled or rejected.
        fill_price (Optional[float]): Price at which the order was filled.
        fill_date (Optional[pd.Timestamp]): Date when the order was filled.
    """

    def __init__(
        self,
        size: int,
        limit: float | None = None,
        stop: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: object | None = None,
        asset: str = "default",
        *,
        trail_percent: float | None = None,
        trail_amount: float | None = None,
    ):
        """
        Initialize an Order instance.

        Args:
            size (int): Order size (positive for buy, negative for sell).
            limit (Optional[float], optional): Limit price for limit orders. Defaults to None.
            stop (Optional[float], optional): Stop price for stop orders. Defaults to None.
            sl (Optional[float], optional): Stop loss price. Defaults to None.
            tp (Optional[float], optional): Take profit price. Defaults to None.
            trail_percent (Optional[float], optional): Trailing-stop distance as a
                fraction of the high-water mark (e.g. ``0.05`` = 5%). Mutually
                exclusive with ``trail_amount``. The Broker auto-bumps the SL
                each bar so it ratchets in the trade's favor only.
            trail_amount (Optional[float], optional): Trailing-stop distance as
                an absolute price gap (account currency). Mutually exclusive
                with ``trail_percent``.
            tag (Optional[object], optional): Tag for identification. Defaults to None.
            asset (str, optional): Asset symbol this order targets. Defaults to "default" for
                single-asset backtests. Multi-asset support is layered on this field.

        Raises:
            AssertionError: If the order size is zero.
            ValueError: If both ``trail_percent`` and ``trail_amount`` are set,
                or either is non-positive.
        """
        assert size != 0, 'Order size cannot be zero.'

        if trail_percent is not None and trail_amount is not None:
            raise ValueError("trail_percent and trail_amount are mutually exclusive.")
        if trail_percent is not None and trail_percent <= 0:
            raise ValueError(f"trail_percent must be > 0, got {trail_percent}.")
        if trail_amount is not None and trail_amount <= 0:
            raise ValueError(f"trail_amount must be > 0, got {trail_amount}.")

        self._size: int = size
        self._limit: float | None = limit
        self._stop: float | None = stop
        self._sl: float | None = sl
        self._tp: float | None = tp
        self._trail_percent: float | None = trail_percent
        self._trail_amount: float | None = trail_amount
        self._tag: object | None = tag
        self._asset: str = asset

        self._is_filled: bool = False
        self._fill_price: float | None = None
        self._fill_date: pd.Timestamp | None = None
        self._close_reason: str | None = None

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
    def limit(self) -> float | None:
        """Optional[float]: Limit price."""
        return self._limit

    @property
    def stop(self) -> float | None:
        """Optional[float]: Stop price."""
        return self._stop

    @property
    def sl(self) -> float | None:
        """Optional[float]: Stop loss price."""
        return self._sl

    @property
    def tp(self) -> float | None:
        """Optional[float]: Take profit price."""
        return self._tp

    @property
    def trail_percent(self) -> float | None:
        """Optional[float]: Trailing-stop fraction (e.g. 0.05 for 5%)."""
        return self._trail_percent

    @property
    def trail_amount(self) -> float | None:
        """Optional[float]: Trailing-stop absolute distance."""
        return self._trail_amount

    @property
    def tag(self) -> object | None:
        """Optional[object]: Order tag."""
        return self._tag

    @property
    def asset(self) -> str:
        """str: Asset symbol this order targets."""
        return self._asset

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
    def fill_price(self) -> float | None:
        """Optional[float]: Fill price."""
        return self._fill_price

    @property
    def fill_date(self) -> pd.Timestamp | None:
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
            ('TrailPct', self._trail_percent),
            ('TrailAmt', self._trail_amount),
            ('Tag', self.tag),
        )
        param_str = ', '.join(f'{name}={value}' for name, value in params if value is not None)
        return f'<Order {param_str}>'