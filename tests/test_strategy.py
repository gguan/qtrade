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
