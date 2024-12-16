# components/position.py

from __future__ import annotations

from .trade import Trade

class Position:
    """Represents a trading position containing active and closed trades."""

    def __init__(self):
        """
        Initialize the Position with empty lists for active and closed trades.
        """
        self._active_trades: list[Trade] = []
        self._closed_trades: list[Trade] = []

    @property
    def active_trades(self) -> tuple[Trade, ...]:
        """
        Get a tuple of all active trades.

        Returns:
            Tuple[Trade, ...]: Active trades.
        """
        return tuple(self._active_trades)

    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """
        Get a tuple of all closed trades.

        Returns:
            Tuple[Trade, ...]: Closed trades.
        """
        return tuple(self._closed_trades)
    
    @property
    def size(self) -> int:
        """Calculates the total size of all active trades."""
        return sum(trade.size for trade in self.active_trades)

    def __bool__(self) -> bool:
        """
        bool: Returns True if there are any active trades.

        Returns:
            bool: True if there are active trades, False otherwise.
        """
        return len(self.active_trades) > 0
    
    def __repr__(self):
        """Return a string representation of the Position object."""
        return (
            f'<Position Size={self.size} | Active Trades=({len(self.active_trades)}) | '
            f'Closed Trades=({len(self.closed_trades)})>'
        )

