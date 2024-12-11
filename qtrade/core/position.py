from typing import List, Optional
import pandas as pd
from .trade import Trade

class Position:
    """Represents a trading position containing active and closed trades."""

    def __init__(self):
        # Initialize lists to hold active and closed trades
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []

    def __bool__(self):
        """Returns True if there are any active trades."""
        return self.is_open

    def open_position(self,
                      entry_price: float,
                      entry_date: pd.Timestamp,
                      size: int,
                      sl: Optional[float] = None,
                      tp: Optional[float] = None,
                      tag: object = None):
        """Opens a new trade position with the specified parameters."""
        new_trade = Trade(entry_price, entry_date, size, sl, tp, tag)
        self.active_trades.append(new_trade)

    def close_position(self,
                       trade: Trade,
                       close_size: int,
                       exit_price: float,
                       exit_date: pd.Timestamp,
                       exit_reason: str) -> Optional[Trade]:
        """Closes an active trade and moves it to closed trades."""
        closed_trade = trade.close(close_size, exit_price, exit_date, exit_reason)
        self.closed_trades.append(closed_trade)
        
        return closed_trade

    @property
    def is_open(self) -> bool:
        """Checks if there are any active trades."""
        return len(self.active_trades) > 0

    @property
    def size(self) -> float:
        """Calculates the total size of all active trades."""
        return sum(trade.size for trade in self.active_trades)

    def __repr__(self):
        """Return a string representation of the Position object."""
        return f'<Position: {self.size} ({len(self.active_trades)} trades)>'

