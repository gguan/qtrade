# components/trade.py

import pandas as pd

# Sentinel for "argument not provided" — distinguishes "leave alone" from
# "set to None" (= clear the level) in update_exit_levels().
_UNSET: object = object()


class Trade:
    """
    Represents an individual trade with entry and exit details, profit calculations, and trade status.

    Attributes:
        _size (int): Trade size (positive for long, negative for short).
        _entry_price (float): Price at which the trade was entered.
        _entry_date (pd.Timestamp): Date when the trade was entered.
        _sl (Optional[float]): Stop loss price.
        _tp (Optional[float]): Take profit price.
        _trail_percent (Optional[float]): Trailing-stop fraction (e.g. 0.05 for 5%).
        _trail_amount (Optional[float]): Trailing-stop absolute distance.
        _trail_high (Optional[float]): High-water mark for long trailing stops.
        _trail_low (Optional[float]): Low-water mark for short trailing stops.
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
        *,
        trail_percent: float | None = None,
        trail_amount: float | None = None,
    ):
        """
        Initialize a Trade instance.

        Args:
            entry_price (float): Price at which the trade was entered.
            entry_date (pd.Timestamp): Date when the trade was entered.
            size (int): Trade size (positive for long, negative for short).
            sl (Optional[float], optional): Stop loss price. Defaults to None.
            tp (Optional[float], optional): Take profit price. Defaults to None.
            trail_percent (Optional[float], optional): Trailing-stop distance as a
                fraction of the current high-water mark (e.g. ``0.05`` = 5%).
                Mutually exclusive with ``trail_amount``. Defaults to None.
            trail_amount (Optional[float], optional): Trailing-stop distance as an
                absolute price gap (in account currency). Mutually exclusive with
                ``trail_percent``. Defaults to None.
            tag (Optional[object], optional): Tag for identifying the trade. Defaults to None.
            asset (str, optional): Asset symbol this trade is on. Defaults to "default" for
                single-asset backtests. Multi-asset support is layered on this field.
            multiplier (float, optional): Contract multiplier for futures-style instruments
                (e.g. 100 for COMEX gold, 50 for E-mini S&P). 1 share = 1 unit of price for
                stocks. Profit and margin both scale with this. Defaults to 1.0.

        Raises:
            ValueError: If the trade size is zero, or both trail_percent and trail_amount
                are set, or either trail value is non-positive.
        """
        assert size != 0, 'Trade size cannot be zero.'

        if trail_percent is not None and trail_amount is not None:
            raise ValueError("trail_percent and trail_amount are mutually exclusive.")
        if trail_percent is not None and trail_percent <= 0:
            raise ValueError(f"trail_percent must be > 0, got {trail_percent}.")
        if trail_amount is not None and trail_amount <= 0:
            raise ValueError(f"trail_amount must be > 0, got {trail_amount}.")

        self._entry_price: float = entry_price
        self._entry_date: pd.Timestamp = entry_date
        self._entry_index: int = entry_index
        self._size: int = size
        self._sl: float | None = sl
        self._tp: float | None = tp
        self._trail_percent: float | None = trail_percent
        self._trail_amount: float | None = trail_amount
        # Seed the high/low-water mark with the entry price so the first bar
        # after open already has a baseline trail-implied SL.
        self._trail_high: float | None = entry_price if (size > 0 and (trail_percent or trail_amount)) else None
        self._trail_low: float | None = entry_price if (size < 0 and (trail_percent or trail_amount)) else None
        self._tag: object | None = tag
        self._asset: str = asset
        self._multiplier: float = multiplier

        self._exit_price: float | None = None
        self._exit_date: pd.Timestamp | None = None
        self._exit_index: int | None = None
        self._profit: float | None = None
        self._exit_reason: str | None = None  # 'signal', 'sl', 'tp', 'end'

        # Apply the initial trail-implied SL so it's already in effect on bar
        # 1, even before the first __update_trailing_stops pass runs.
        if self._trail_percent is not None or self._trail_amount is not None:
            self._recompute_trail_sl()

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
        # Preserve trail metadata on the closed-trade record so post-run
        # analysis can tell which trades were trailing-stopped.
        closed_trade._trail_percent = self._trail_percent
        closed_trade._trail_amount = self._trail_amount
        closed_trade._trail_high = self._trail_high
        closed_trade._trail_low = self._trail_low

        # Update the original Trade object's size
        self._size -= size_to_close

        return closed_trade

    # ------------------------------------------------------------------
    # Mutators — change SL/TP after a trade is open.
    # ------------------------------------------------------------------

    def update_exit_levels(
        self,
        sl: float | None = _UNSET,  # type: ignore[assignment]
        tp: float | None = _UNSET,  # type: ignore[assignment]
    ) -> None:
        """Modify SL / TP on an open trade.

        Pass ``None`` to *clear* a level; omit a parameter to leave it
        unchanged. Useful for hand-rolled trailing logic, ratcheting up
        a stop after a profit threshold, or moving TP further out.

        Args:
            sl: New stop-loss price. Pass ``None`` to clear the existing SL.
            tp: New take-profit price. Pass ``None`` to clear the existing TP.

        Example:
            >>> trade.update_exit_levels(sl=110)        # set/move SL only
            >>> trade.update_exit_levels(tp=None)       # clear TP
            >>> trade.update_exit_levels(sl=110, tp=130)  # both at once
        """
        if sl is not _UNSET:
            self._sl = sl
        if tp is not _UNSET:
            self._tp = tp

    def _update_trailing_stop(self, bar_high: float, bar_low: float) -> None:
        """Bump the trailing SL toward the favorable side based on the bar.

        Long: ratchet ``_trail_high = max(_trail_high, bar_high)``, then
        recompute ``_sl`` so it never moves away from price.
        Short: symmetric using ``_trail_low`` and ``bar_low``.

        Called by Broker once per bar; no-op if neither trail_percent nor
        trail_amount was set.
        """
        if self._trail_percent is None and self._trail_amount is None:
            return

        if self.is_long:
            if self._trail_high is None or bar_high > self._trail_high:
                self._trail_high = bar_high
        else:
            if self._trail_low is None or bar_low < self._trail_low:
                self._trail_low = bar_low

        self._recompute_trail_sl()

    def _recompute_trail_sl(self) -> None:
        """Recompute SL from the current trail high/low water mark.

        Only ratchets — the SL never moves against the trader. Coexists with
        an explicit SL: whichever is tighter (better for the trader) wins.
        """
        if self.is_long and self._trail_high is not None:
            if self._trail_percent is not None:
                new_sl = self._trail_high * (1 - self._trail_percent)
            else:
                new_sl = self._trail_high - self._trail_amount  # type: ignore[operator]
            if self._sl is None or new_sl > self._sl:
                self._sl = new_sl
        elif self.is_short and self._trail_low is not None:
            if self._trail_percent is not None:
                new_sl = self._trail_low * (1 + self._trail_percent)
            else:
                new_sl = self._trail_low + self._trail_amount  # type: ignore[operator]
            if self._sl is None or new_sl < self._sl:
                self._sl = new_sl

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
        """Optional[float]: Stop loss price (modifiable)."""
        return self._sl

    @sl.setter
    def sl(self, value: float | None) -> None:
        """Set / clear the stop-loss price on an open trade."""
        self._sl = value

    @property
    def tp(self) -> float | None:
        """Optional[float]: Take profit price (modifiable)."""
        return self._tp

    @tp.setter
    def tp(self, value: float | None) -> None:
        """Set / clear the take-profit price on an open trade."""
        self._tp = value

    @property
    def trail_percent(self) -> float | None:
        """Optional[float]: Trailing-stop fraction, or None if not trailing."""
        return self._trail_percent

    @property
    def trail_amount(self) -> float | None:
        """Optional[float]: Trailing-stop absolute distance, or None if not trailing."""
        return self._trail_amount

    @property
    def trail_high(self) -> float | None:
        """Optional[float]: High-water mark for long trailing stop."""
        return self._trail_high

    @property
    def trail_low(self) -> float | None:
        """Optional[float]: Low-water mark for short trailing stop."""
        return self._trail_low

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
