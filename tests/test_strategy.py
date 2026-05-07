"""Tests for qtrade.backtest.strategy.Strategy."""

import pandas as pd
import pytest

from qtrade.backtest.strategy import Strategy
from qtrade.core import Broker, NoCommission, Order


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            'Open': [100, 102, 104, 106, 108],
            'High': [101, 103, 105, 107, 109],
            'Low': [99, 101, 103, 105, 107],
            'Close': [100, 102, 104, 106, 108],
        },
        index=pd.date_range('2024-01-01', periods=5, freq='D'),
    )


@pytest.fixture
def broker(data):
    return Broker(data, cash=10_000, commission=NoCommission(), margin_ratio=1.0, trade_on_close=True)


class _DummyStrategy(Strategy):
    def prepare(self):
        pass

    def on_bar_close(self):
        pass


def test_strategy_stores_params_as_attributes(broker, data):
    s = _DummyStrategy(broker, data, {'fast': 5, 'slow': 20})
    assert s.fast == 5
    assert s.slow == 20


def test_strategy_buy_with_explicit_size_places_order(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.buy(size=10)
    assert len(broker.filled_orders) == 1
    assert broker.filled_orders[0].size == 10


def test_strategy_buy_default_size_uses_available_margin(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.buy()  # default size = available_margin // close_price = 10000 // 100 = 100
    assert len(broker.filled_orders) == 1
    assert broker.filled_orders[0].size == 100


def test_strategy_sell_with_explicit_size(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.sell(size=5)
    assert len(broker.filled_orders) == 1
    assert broker.filled_orders[0].size == -5


def test_strategy_sell_default_size_uses_position(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.buy(size=10)
    assert s.position.size == 10
    s.sell()  # default = position.size = 10 → order size -10 → flat
    assert s.position.size == 0


def test_strategy_close_long_position(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.buy(size=10)
    s.close()
    assert s.position.size == 0


def test_strategy_close_short_position(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.sell(size=10)
    assert s.position.size == -10
    s.close()
    assert s.position.size == 0


def test_strategy_close_when_flat_is_noop(broker, data):
    s = _DummyStrategy(broker, data, {})
    s.close()
    assert s.position.size == 0
    assert broker.filled_orders == ()


def test_strategy_data_is_truncated_to_current_time(broker, data):
    broker.process_bar(pd.Timestamp('2024-01-03'))
    s = _DummyStrategy(broker, data, {})
    assert len(s.data) == 3


def test_strategy_equity_and_unrealized_pnl_proxy_to_broker(broker, data):
    s = _DummyStrategy(broker, data, {})
    assert s.equity == broker.equity
    assert s.unrealized_pnl == broker.unrealized_pnl


def test_strategy_pending_orders_proxy(broker, data):
    s = _DummyStrategy(broker, data, {})
    broker.place_orders(Order(size=1, limit=50))  # limit order goes to pending
    assert len(s.pending_orders) == 1


def test_strategy_str_includes_params(broker, data):
    s = _DummyStrategy(broker, data, {'n': 5, 'm': 10})
    text = str(s)
    assert '_DummyStrategy' in text
    assert 'n=5' in text and 'm=10' in text


# ---------------------------------------------------------------------------
# Regression tests for known bugs (will fail until the underlying fix lands)
# ---------------------------------------------------------------------------


def test_strategy_active_trades_returns_tuple_of_active_trades(broker, data):
    """Bug #1: Strategy.active_trades calls position.active_trades() with parens,
    but it's a @property — currently raises TypeError 'tuple object is not callable'."""
    s = _DummyStrategy(broker, data, {})
    s.buy(size=5)
    trades = s.active_trades  # currently raises TypeError
    assert isinstance(trades, tuple)
    assert len(trades) == 1
    assert trades[0].size == 5


def test_strategy_closed_trades_returns_tuple_of_closed_trades(broker, data):
    """Bug #1 (sibling): Strategy.closed_trades has the same '@property called as method' bug."""
    s = _DummyStrategy(broker, data, {})
    s.buy(size=5)
    s.sell(size=5)  # closes the position
    trades = s.closed_trades  # currently raises TypeError
    assert isinstance(trades, tuple)
    assert len(trades) == 1


def test_strategy_sell_with_negative_explicit_size_still_sells(broker, data):
    """Bug #10: passing negative size to sell() used to silently flip to a buy."""
    s = _DummyStrategy(broker, data, {})
    s.sell(size=-5)  # user-friendly: magnitude regardless of sign
    assert broker.filled_orders[0].size == -5  # actually a sell
    assert broker.filled_orders[0].is_short is True


def test_strategy_buy_with_negative_explicit_size_still_buys(broker, data):
    """Bug #10 sibling: buy() with negative size should still be a buy."""
    s = _DummyStrategy(broker, data, {})
    s.buy(size=-7)
    assert broker.filled_orders[0].size == 7
    assert broker.filled_orders[0].is_long is True


def test_strategy_prepare_can_add_indicators_via_data_iteration(broker, data):
    """Regression: post v0.3.0 ``self._data`` is a dict[str, DataFrame].
    The recommended pattern (works for single + multi-asset) is to iterate
    its values. Verify a strategy using this pattern can read its own
    indicators from on_bar_close."""

    class _SmaStrat(Strategy):
        def prepare(self):
            for df in self._data.values():
                df['ma'] = df['Close'].rolling(3).mean()

        def on_bar_close(self):
            assert 'ma' in self.data.columns

    s = _SmaStrat(broker, data, {})
    s.prepare()
    # Advance broker so self.data has at least 3 rows for the rolling mean.
    broker.process_bar(data.index[2])
    s.on_bar_close()


def test_strategy_buy_with_zero_default_size_does_not_print(data, capsys):
    """Bug #3: Strategy.buy() has a leftover debug print() when default size resolves to 0."""
    # cash=50 / close=100 → default size = 0
    broker = Broker(data, cash=50, commission=NoCommission(), margin_ratio=1.0, trade_on_close=True)
    s = _DummyStrategy(broker, data, {})
    try:
        s.buy()  # default size=None → 0 → triggers Order(0) AssertionError, but print fires first
    except AssertionError:
        pass
    captured = capsys.readouterr()
    assert captured.out == "", f"Stray debug output: {captured.out!r}"
