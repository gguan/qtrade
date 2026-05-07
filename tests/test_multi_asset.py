"""Multi-asset internals tests.

P1 refactor exposes ``broker.data_by_asset`` / ``broker.positions`` and
threads an ``asset`` field through Trade/Order. The user-facing Strategy
API is unchanged; these tests pin down the internal wiring so P2 can
build on a verified foundation.
"""

import pandas as pd
import pytest

from qtrade.backtest.backtest import Backtest
from qtrade.backtest.strategy import Strategy
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


# ---------------------------------------------------------------------------
# Strategy + Backtest end-to-end multi-asset (P2)
# ---------------------------------------------------------------------------


@pytest.fixture
def two_asset_data():
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    return {
        'AAPL': _ohlc(list(range(100, 120)), dates),
        'MSFT': _ohlc(list(range(200, 220)), dates),
    }


class _PortfolioBuyHold(Strategy):
    """Buys 5 of each asset on the first bar where it has a position == 0."""
    def prepare(self):
        pass

    def on_bar_close(self):
        for asset in self.assets:
            if self.positions[asset].size == 0:
                self.buy(asset, size=5)


def test_strategy_assets_lists_all_inputs(two_asset_data):
    bt = Backtest(two_asset_data, _PortfolioBuyHold, cash=100_000, trade_on_close=True)
    bt.run()
    assert sorted(bt.strategy.assets) == ['AAPL', 'MSFT']


def test_strategy_buy_with_explicit_asset_routes_correctly(two_asset_data):
    bt = Backtest(two_asset_data, _PortfolioBuyHold, cash=100_000, trade_on_close=True)
    bt.run()
    closed = bt.broker.closed_trades
    assert len([t for t in closed if t.asset == 'AAPL']) >= 1
    assert len([t for t in closed if t.asset == 'MSFT']) >= 1


def test_strategy_buy_without_asset_in_multi_raises(two_asset_data):
    class _BadStrat(Strategy):
        def prepare(self): pass
        def on_bar_close(self):
            self.buy(size=10)  # no asset → should error

    bt = Backtest(two_asset_data, _BadStrat, cash=100_000, trade_on_close=True)
    with pytest.raises(ValueError, match=r"multiple assets"):
        bt.run()


def test_strategy_data_property_raises_in_multi(two_asset_data):
    class _BadStrat(Strategy):
        def prepare(self):
            _ = self.data  # should error in multi-asset mode

        def on_bar_close(self): pass

    bt = Backtest(two_asset_data, _BadStrat, cash=100_000, trade_on_close=True)
    with pytest.raises(AttributeError, match=r"multiple assets"):
        bt.run()


def test_strategy_data_by_asset_returns_truncated_per_asset(two_asset_data):
    captured = {}

    class _CaptureStrat(Strategy):
        def prepare(self): pass
        def on_bar_close(self):
            for asset, df in self.data_by_asset.items():
                captured.setdefault(asset, []).append(len(df))

    bt = Backtest(two_asset_data, _CaptureStrat, cash=100_000, trade_on_close=True)
    bt.run()
    # Lengths grow over time, finishing at the full data length.
    assert captured['AAPL'][-1] == 20
    assert captured['MSFT'][-1] == 20
    assert captured['AAPL'] == captured['MSFT']  # synchronized


def test_strategy_close_specific_asset(two_asset_data):
    """close(asset) closes only the named asset, leaving others active until end-of-run."""
    captured: dict[str, int | None] = {'aapl_size_after_close': None, 'msft_size_after_close': None}

    class _Strat(Strategy):
        def prepare(self): pass
        def on_bar_close(self):
            now = self._broker.current_time
            if now == self._data['AAPL'].index[1]:
                self.buy('AAPL', size=10)
                self.buy('MSFT', size=10)
            elif now == self._data['AAPL'].index[5]:
                self.close('AAPL')
                captured['aapl_size_after_close'] = self.positions['AAPL'].size
                captured['msft_size_after_close'] = self.positions['MSFT'].size

    bt = Backtest(two_asset_data, _Strat, cash=100_000, trade_on_close=True)
    bt.run()
    # Right after close('AAPL'): AAPL flat, MSFT untouched.
    assert captured['aapl_size_after_close'] == 0
    assert captured['msft_size_after_close'] == 10
    # End-of-run sweep closes MSFT too — final closed_trades has both.
    closed_assets = [t.asset for t in bt.broker.closed_trades]
    assert closed_assets.count('AAPL') == 1
    assert closed_assets.count('MSFT') == 1


def test_backtest_rejects_multi_asset_with_misaligned_indexes():
    df1 = _ohlc([100, 101, 102], pd.date_range('2024-01-01', periods=3, freq='D'))
    df2 = _ohlc([200, 201, 202], pd.date_range('2024-01-02', periods=3, freq='D'))  # different start
    with pytest.raises(ValueError, match=r"different index"):
        Backtest({'AAPL': df1, 'MSFT': df2}, _PortfolioBuyHold)


def test_backtest_validates_each_asset_dataframe():
    good = _ohlc([100, 101], pd.date_range('2024-01-01', periods=2, freq='D'))
    bad_no_close = good.drop(columns=['Close'])
    with pytest.raises(ValueError, match=r"Asset 'MSFT'"):
        Backtest({'AAPL': good, 'MSFT': bad_no_close}, _PortfolioBuyHold)


def test_get_trade_history_includes_asset_column(two_asset_data):
    bt = Backtest(two_asset_data, _PortfolioBuyHold, cash=100_000, trade_on_close=True)
    bt.run()
    df = bt.get_trade_history()
    assert 'Asset' in df.columns
    assert set(df['Asset'].unique()) == {'AAPL', 'MSFT'}
