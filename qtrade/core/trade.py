# components/trade.py

import pandas as pd


class Trade:
    """
    Represents an individual trade with entry and exit details, profit calculations, and trade status.

    Attributes:
        _size (int): Trade size (positive for long, negative for short).
        _entry_price (float): Price at which the trade was entered.
        _entry_date (pd.Timestamp): Date when the trade was entered.
        _sl (Optional[float]): Stop loss price.
        _tp (Optional[float]): Take profit price.
        _tag (Optional[object]): Tag for identifying the trade.
        _exit_price (Optional[float]): Price at which the trade was exited.
        _exit_date (Optional[pd.Timestamp]): Date when the trade was exited.
        _profit (Optional[float]): Profit or loss from the trade.
        _exit_reason (Optional[str]): Reason for exiting the trade ('signal', 'sl', 'tp', 'end').
    """

    def __init__(
        self,
        entry_price: float,
        entry_date: pd.Timestamp,
        entry_index: int,
        size: int,
        sl: float | None = None,
        tp: float | None = None,
        tag: object | None = None,
        asset: str = "default",
        multiplier: float = 1.0,
    ):
        """
        Initialize a Trade instance.

        Args:
            entry_price (float): Price at which the trade was entered.
            entry_date (pd.Timestamp): Date when the trade was entered.
            size (int): Trade size (positive for long, negative for short).
            sl (Optional[float], optional): Stop loss price. Defaults to None.
            tp (Optional[float], optional): Take profit price. Defaults to None.
            tag (Optional[object], optional): Tag for identifying the trade. Defaults to None.
            asset (str, optional): Asset symbol this trade is on. Defaults to "default" for
                single-asset backtests. Multi-asset support is layered on this field.
            multiplier (float, optional): Contract multiplier for futures-style instruments
                (e.g. 100 for COMEX gold, 50 for E-mini S&P). 1 share = 1 unit of price for
                stocks. Profit and margin both scale with this. Defaults to 1.0.

        Raises:
            ValueError: If the trade size is zero.
        """
        assert size != 0, 'Trade size cannot be zero.'

        self._entry_price: float = entry_price
        self._entry_date: pd.Timestamp = entry_date
        self._entry_index: int = entry_index
        self._size: int = size
        self._sl: float | None = sl
        self._tp: float | None = tp
        self._tag: object | None = tag
        self._asset: str = asset
        self._multiplier: float = multiplier

        self._exit_price: float | None = None
        self._exit_date: pd.Timestamp | None = None
        self._exit_index: int | None = None
        self._profit: float | None = None
        self._exit_reason: str | None = None  # 'signal', 'sl', 'tp', 'end'

    def close(
        self,
        size: int | None,
        exit_price: float,
        exit_date: pd.Timestamp,
        exit_index: int,
        exit_reason: str
    ) -> 'Trade':
        """
        Closes a portion or full of the trade and records exit details.

        Args:
            size (int): Size to close (must not exceed current trade size).
            exit_price (float): Price at which the trade is closed.
            exit_date (pd.Timestamp): Date when the trade is closed.
            exit_reason (str): Reason for closing ('signal', 'sl', 'tp', 'end').

        Returns:
            Trade: A new Trade instance representing the closed portion.

        Raises:
            ValueError: If attempting to close more than the current trade size or if the trade is already fully closed.
        """
        if self.is_closed:
            raise ValueError("Cannot close a trade that is already fully closed.")
        if size and abs(size) > abs(self._size):
            raise ValueError("Cannot close more than the current position size.")

        size_to_close = size if size is not None else self._size

        # Calculate profit for the closed portion (scaled by contract multiplier)
        profit = (exit_price - self._entry_price) * size_to_close * self._multiplier

        # Create a new Trade object to record the closed portion
        closed_trade = Trade(
            entry_price=self._entry_price,
            entry_date=self._entry_date,
            entry_index=self._entry_index,
            size=size_to_close,
            sl=self._sl,
            tp=self._tp,
            tag=self._tag,
            asset=self._asset,
            multiplier=self._multiplier,
        )
        closed_trade._exit_price = exit_price
        closed_trade._exit_date = exit_date
        closed_trade._exit_index = exit_index
        closed_trade._profit = profit
        closed_trade._exit_reason = exit_reason

        # Update the original Trade object's size
        self._size -= size_to_close

        return closed_trade

    @property
    def is_long(self) -> bool:
        """bool: True if the trade is a long position."""
        return self._size > 0

    @property
    def is_short(self) -> bool:
        """bool: True if the trade is a short position."""
        return self._size < 0

    @property
    def size(self) -> int:
        """int: Current size of the trade."""
        return self._size

    @property
    def entry_price(self) -> float:
        """float: Price at which the trade was entered."""
        return self._entry_price

    @property
    def entry_date(self) -> pd.Timestamp:
        """pd.Timestamp: Date when the trade was entered."""
        return self._entry_date
    
    @property
    def entry_index(self) -> int:
        """int: Index when the trade was entered."""
        return self._entry_index

    @property
    def asset(self) -> str:
        """str: Asset symbol this trade is on."""
        return self._asset

    @property
    def multiplier(self) -> float:
        """float: Contract multiplier (1.0 for stocks, e.g. 100 for COMEX GC)."""
        return self._multiplier

    @property
    def sl(self) -> float | None:
        """Optional[float]: Stop loss price."""
        return self._sl

    @property
    def tp(self) -> float | None:
        """Optional[float]: Take profit price."""
        return self._tp

    @property
    def tag(self) -> object | None:
        """Optional[object]: Tag for identifying the trade."""
        return self._tag
    
    @property
    def exit_price(self) -> float | None:
        """Optional[float]: Price at which the trade was exited."""
        return self._exit_price

    @property
    def exit_date(self) -> pd.Timestamp | None:
        """Optional[pd.Timestamp]: Date when the trade was exited."""
        return self._exit_date
    
    @property
    def exit_index(self) -> int | None:
        """Optional[int]: Index when the trade was exited."""
        return self._exit_index

    @property
    def profit(self) -> float | None:
        """
        Optional[float]: Profit or loss from the trade.

        Returns:
            Optional[float]: Profit if the trade is closed, otherwise None.
        """
        return self._profit

    @property
    def exit_reason(self) -> str | None:
        """
        Optional[str]: Reason for exiting the trade.

        Returns:
            Optional[str]: Exit reason if the trade is closed, otherwise None.
        """
        return self._exit_reason

    @property
    def is_closed(self) -> bool:
        """bool: True if the trade has been fully closed."""
        return self.size == 0 or self.exit_date is not None

    def __repr__(self) -> str:
        return (f'<Trade Size: {self._size} | Time: {self._entry_date} - {self._exit_date or "N/A"} | '
                f'Price: {self._entry_price} - {self._exit_price or "N/A"} | '
                f'Profit/Loss: {self._profit or "N/A"} | '
                f'Tag: {self._tag if self._tag is not None else "N/A"} | '
                f'Reason: {self._exit_reason if self._exit_reason is not None else "N/A"}>')