# tests/test_position.py

import pytest
import pandas as pd
from qtrade.core import Trade, Position


def test_position_initialization():
    position = Position()
    assert position.active_trades == ()
    assert position.closed_trades == ()
    assert position.size == 0
    assert bool(position) is False


def test_position_add_trade():
    position = Position()
    trade = Trade(100.0, pd.Timestamp('2024-01-01'), 10)
    position._active_trades.append(trade)
    assert position.active_trades == (trade,)
    assert position.size == 10
    assert bool(position) is True


def test_position_properties():
    position = Position()
    trade1 = Trade(100.0, pd.Timestamp('2024-01-01'), 10)
    trade2 = Trade(105.0, pd.Timestamp('2024-01-02'), -5)
    position._active_trades.append(trade1)
    position._active_trades.append(trade2)

    assert position.size == 5  # 10 + (-5)
    assert bool(position) is True
    assert position.active_trades == (trade1, trade2)
    assert position.closed_trades == ()
