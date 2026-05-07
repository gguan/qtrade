"""Multi-asset internals tests.

P1 refactor exposes ``broker.data_by_asset`` / ``broker.positions`` and
threads an ``asset`` field through Trade/Order. The user-facing Strategy
API is unchanged; these tests pin down the internal wiring so P2 can
build on a verified foundation.
"""

import pandas as pd
import pytest

from qtrade.core import Broker, NoCommission, Order


def _ohlc(prices, dates):
    return pd.DataFrame(
        {'Open': prices, 'High': [p + 1 for p in prices],
         'Low': [p - 1 for p in prices], 'Close': prices},
        index=dates,
    )


@pytest.fixture
def two_asset_broker():
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    data = {
        'AAPL': _ohlc([100, 101, 102, 103, 104], dates),
        'MSFT': _ohlc([200, 201, 202, 203, 204], dates),
    }
    return Broker(data, cash=100_000, commission=NoCommission(),
                  margin_ratio=1.0, trade_on_close=True)


def test_broker_accepts_dict_of_dataframes(two_asset_broker):
    assert two_asset_broker.assets == ['AAPL', 'MSFT']
    assert set(two_asset_broker.positions.keys()) == {'AAPL', 'MSFT'}


def test_single_asset_legacy_property_still_works():
    df = _ohlc([100, 101], pd.date_range('2024-01-01', periods=2, freq='D'))
    broker = Broker(df, cash=10_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    # Single-asset path: broker.data and broker.position still return scalars.
    assert isinstance(broker.data, pd.DataFrame)
    assert broker.position is broker.positions['default']


def test_multi_asset_data_property_raises_clearly(two_asset_broker):
    with pytest.raises(AttributeError, match=r"multiple assets"):
        _ = two_asset_broker.data
    with pytest.raises(AttributeError, match=r"multiple assets"):
        _ = two_asset_broker.position


def test_orders_route_to_correct_asset_position(two_asset_broker):
    two_asset_broker.process_bar(pd.Timestamp('2024-01-01'))
    two_asset_broker.place_orders(Order(size=10, asset='AAPL'))
    two_asset_broker.place_orders(Order(size=5, asset='MSFT'))

    assert two_asset_broker.positions['AAPL'].size == 10
    assert two_asset_broker.positions['MSFT'].size == 5
    # filled_orders is a single shared list but each carries its asset
    fills = two_asset_broker.filled_orders
    assert {o.asset for o in fills} == {'AAPL', 'MSFT'}


def test_trade_carries_asset_through_close(two_asset_broker):
    two_asset_broker.process_bar(pd.Timestamp('2024-01-01'))
    two_asset_broker.place_orders(Order(size=10, asset='AAPL'))
    two_asset_broker.process_bar(pd.Timestamp('2024-01-02'))
    two_asset_broker.place_orders(Order(size=-10, asset='AAPL'))  # close

    closed = two_asset_broker.positions['AAPL'].closed_trades
    assert len(closed) == 1
    assert closed[0].asset == 'AAPL'
    # MSFT untouched
    assert two_asset_broker.positions['MSFT'].closed_trades == ()


def test_unrealized_pnl_aggregates_across_assets(two_asset_broker):
    two_asset_broker.process_bar(pd.Timestamp('2024-01-01'))
    two_asset_broker.place_orders(Order(size=10, asset='AAPL'))   # fill at 100
    two_asset_broker.place_orders(Order(size=20, asset='MSFT'))   # fill at 200

    two_asset_broker.process_bar(pd.Timestamp('2024-01-03'))
    # AAPL close=102, MSFT close=202 → +20 + +40 = 60
    assert two_asset_broker.unrealized_pnl == pytest.approx(60.0)


def test_sl_tp_only_fires_on_correct_asset_data(two_asset_broker):
    """A long SL on AAPL should not trigger from MSFT's price action."""
    two_asset_broker._open_trade(
        asset='AAPL', entry_price=100.0,
        entry_date=pd.Timestamp('2024-01-01'), size=10,
        sl=99.0,  # AAPL low on day 1 = 99 → fires
    )
    two_asset_broker._open_trade(
        asset='MSFT', entry_price=200.0,
        entry_date=pd.Timestamp('2024-01-01'), size=10,
        sl=99.0,  # MSFT low never gets near 99
    )
    two_asset_broker.process_bar(pd.Timestamp('2024-01-01'))
    two_asset_broker.process_bar(pd.Timestamp('2024-01-02'))

    assert len(two_asset_broker.positions['AAPL'].closed_trades) == 1
    assert two_asset_broker.positions['AAPL'].closed_trades[0].exit_reason == 'sl'
    assert two_asset_broker.positions['MSFT'].closed_trades == ()


def test_closed_trades_property_concatenates_across_assets(two_asset_broker):
    two_asset_broker.process_bar(pd.Timestamp('2024-01-01'))
    two_asset_broker.place_orders(Order(size=10, asset='AAPL'))
    two_asset_broker.place_orders(Order(size=5, asset='MSFT'))
    two_asset_broker.process_bar(pd.Timestamp('2024-01-02'))
    two_asset_broker.close_all_positions()

    all_closed = two_asset_broker.closed_trades
    assert len(all_closed) == 2
    assert {t.asset for t in all_closed} == {'AAPL', 'MSFT'}
