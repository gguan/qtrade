# tests/test_trade.py

import pytest
import pandas as pd
from qtrade.core import Trade


def test_trade_initialization():
    entry_price = 100.0
    entry_date = pd.Timestamp('2024-01-01')
    entry_index = 0
    size = 10
    sl = 95.0
    tp = 105.0
    tag = "TestTrade"

    trade = Trade(entry_price, entry_date, entry_index, size, sl, tp, tag)

    assert trade.entry_price == entry_price
    assert trade.entry_date == entry_date
    assert trade.size == size
    assert trade.sl == sl
    assert trade.tp == tp
    assert trade.tag == tag
    assert trade.exit_price is None
    assert trade.exit_date is None
    assert trade.profit is None
    assert trade.exit_reason is None
    assert trade.is_long is True
    assert trade.is_short is False
    assert trade.is_closed is False


def test_trade_initialization_zero_size():
    entry_price = 100.0
    entry_date = pd.Timestamp('2024-01-01')
    entry_index = 0
    size = 0

    with pytest.raises(AssertionError, match="Trade size cannot be zero."):
        Trade(entry_price, entry_date, entry_index, size)


def test_trade_close_fully():
    entry_price = 100.0
    entry_date = pd.Timestamp('2024-01-01')
    entry_index = 0
    size = 10
    trade = Trade(entry_price, entry_date, entry_index, size)

    exit_price = 110.0
    exit_date = pd.Timestamp('2024-01-02')
    exit_index = 1
    exit_reason = 'tp'

    closed_trade = trade.close(None, exit_price, exit_date, exit_index, exit_reason)

    assert closed_trade.size == size
    assert closed_trade.exit_price == exit_price
    assert closed_trade.exit_date == exit_date
    assert closed_trade.profit == (exit_price - entry_price) * size
    assert closed_trade.exit_reason == exit_reason
    assert closed_trade.is_closed is True

    # Original trade size should be zero
    assert trade.size == 0
    assert trade.is_closed is True


def test_trade_close_partially():
    entry_price = 100.0
    entry_date = pd.Timestamp('2024-01-01')
    entry_index = 0
    size = 10
    trade = Trade(entry_price, entry_date, entry_index, size)

    close_size = 4
    exit_price = 105.0
    exit_date = pd.Timestamp('2024-01-02')
    exit_index = 1
    exit_reason = 'signal'

    closed_trade = trade.close(close_size, exit_price, exit_date, exit_index, exit_reason)

    assert closed_trade.size == close_size
    assert closed_trade.exit_price == exit_price
    assert closed_trade.exit_date == exit_date
    assert closed_trade.profit == (exit_price - entry_price) * close_size
    assert closed_trade.exit_reason == exit_reason
    assert closed_trade.is_closed is True

    # Original trade should have reduced size
    assert trade.size == size - close_size
    assert trade.exit_price is None
    assert trade.exit_date is None
    assert trade.profit is None
    assert trade.exit_reason is None
    assert trade.is_closed is False


def test_trade_close_over_size():
    entry_price = 100.0
    entry_date = pd.Timestamp('2024-01-01')
    entry_index = 0
    size = 5
    trade = Trade(entry_price, entry_date, entry_index, size)

    close_size = 6
    exit_price = 110.0
    exit_date = pd.Timestamp('2024-01-02')
    exit_index = 1
    exit_reason = 'tp'

    with pytest.raises(ValueError, match="Cannot close more than the current position size."):
        trade.close(close_size, exit_price, exit_date, exit_index, exit_reason)


def test_trade_close_already_closed():
    entry_price = 100.0
    entry_date = pd.Timestamp('2024-01-01')
    entry_index = 0
    size = 5
    trade = Trade(entry_price, entry_date, entry_index, size)

    # Fully close the trade
    trade.close(None, 110.0, pd.Timestamp('2024-01-02'), 1, 'tp')

    # Attempt to close again
    with pytest.raises(ValueError, match="Cannot close a trade that is already fully closed."):
        trade.close(1, 115.0, pd.Timestamp('2024-01-03'), 1, 'tp')


def test_trade_properties_long():
    trade = Trade(100.0, pd.Timestamp('2024-01-01'), 0, 10)
    assert trade.is_long is True
    assert trade.is_short is False


def test_trade_properties_short():
    trade = Trade(100.0, pd.Timestamp('2024-01-01'), 0, -10)
    assert trade.is_long is False
    assert trade.is_short is True


def test_trade_is_closed():
    trade = Trade(100.0, pd.Timestamp('2024-01-01'), 0, 10)
    assert trade.is_closed is False

    trade.close(10, 90.0, pd.Timestamp('2024-01-02'), 1, 'sl')
    assert trade.is_closed is True