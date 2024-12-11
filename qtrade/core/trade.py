# components/trade.py

from typing import Optional
import pandas as pd


class Trade:
    """
    Represents a trade with entry and exit details, profit calculations, and trade status.
    """
    def __init__(self,
                 entry_price: float,
                 entry_date: pd.Timestamp,
                 size: int,
                 sl: Optional[float] = None,
                 tp: Optional[float] = None,
                 tag: object = None):
        # Initialize trade attributes
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.size = size
        self.sl = sl  # Stop loss price
        self.tp = tp  # Take profit price
        self.tag = tag
        self.exit_price: Optional[float] = None
        self.exit_date: Optional[pd.Timestamp] = None
        self.profit: Optional[float] = None
        self.exit_reason: Optional[str] = None  # 'signal', 'sl', 'tp', 'end'


    def close(self,
              size: int,
              exit_price: float,
              exit_date: pd.Timestamp,
              exit_reason: str):
        """
        Closes a portion of the trade and records exit details.
        """
        assert abs(size) <= abs(self.size), "Cannot close more than the current position size"

        # Calculate profit for the closed portion
        profit = (exit_price - self.entry_price) * size

        # Create a new Trade object to record the closed portion
        closed_trade = Trade(
            entry_price=self.entry_price,
            entry_date=self.entry_date,
            size=size,
            sl=self.sl,
            tp=self.tp,
            tag=self.tag
        )
        closed_trade.exit_price = exit_price
        closed_trade.exit_date = exit_date
        closed_trade.profit = profit
        closed_trade.exit_reason = exit_reason

        # Update the original Trade object's size
        self.size -= size

        return closed_trade

    @property
    def is_long(self) -> bool:
        """Check if the trade is a long position."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if the trade is a short position."""
        return self.size < 0

    def __repr__(self):
        """Return a formatted string representation of the Trade object."""
        return (f'<Trade | Size: {self.size} | Time: {self.entry_date} - {self.exit_date or "N/A"} | '
                f'Price: {self.entry_price} - {self.exit_price or "N/A"} | '
                f'Profit/Loss: {self.profit or "N/A"} | '
                f'Tag: {self.tag if self.tag is not None else "N/A"} | '
                f'Reason: {self.exit_reason if self.exit_reason is not None else "N/A"}>')
