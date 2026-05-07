"""Tests for qtrade.backtest.backtest.Backtest."""

import pandas as pd
import pytest

from qtrade.backtest.backtest import Backtest
from qtrade.backtest.strategy import Strategy
from qtrade.core import NoCommission


class BuyAndHold(Strategy):
    def prepare(self):
        pass

    def on_bar_close(self):
        if self.position.size == 0:
            self.buy(size=10)


class ParamStrategy(Strategy):
    """Strategy that buys only when bar count > threshold; reads `threshold` from params."""

    def prepare(self):
        self._bars = 0

    def on_bar_close(self):
        self._bars += 1
        if self._bars > self.threshold and self.position.size == 0:
            self.buy(size=5)


def test_backtest_rejects_non_datetime_index(ohlc_data_trending):
    bad = ohlc_data_trending.reset_index(drop=True)
    with pytest.raises(ValueError, match=r"Data index must be a DatetimeIndex"):
        Backtest(bad, BuyAndHold)


def test_backtest_rejects_missing_ohlc_columns():
    df = pd.DataFrame(
        {'open': [1, 2], 'high': [1, 2], 'low': [1, 2]},  # missing 'close'
        index=pd.date_range('2024-01-01', periods=2),
    )
    with pytest.raises(ValueError, match=r"Data must contain columns"):
        Backtest(df, BuyAndHold)


def test_backtest_sorts_unsorted_index(ohlc_data_trending):
    shuffled = ohlc_data_trending.iloc[::-1]
    bt = Backtest(shuffled, BuyAndHold)
    assert bt.data.index.is_monotonic_increasing


def test_backtest_run_buy_and_hold_creates_one_trade(ohlc_data_trending):
    bt = Backtest(ohlc_data_trending, BuyAndHold, cash=10_000, commission=NoCommission())
    bt.run()
    closed = bt.broker.closed_trades
    assert len(closed) == 1
    assert closed[0].size == 10
    # Trending up from 100 to 119 → profit positive
    assert closed[0].profit > 0


def test_backtest_get_trade_history_returns_dataframe(ohlc_data_trending):
    bt = Backtest(ohlc_data_trending, BuyAndHold)
    bt.run()
    df = bt.get_trade_history()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        'Type', 'Size', 'Entry Price', 'Exit Price',
        'Entry Time', 'Exit Date', 'Profit', 'Tag', 'Exit Reason', 'Duration',
    ]
    assert len(df) == 1
    assert df.iloc[0]['Type'] == 'Long'


def test_backtest_show_stats_runs_without_error(ohlc_data_trending, capsys):
    bt = Backtest(ohlc_data_trending, BuyAndHold)
    bt.run()
    bt.show_stats()
    out = capsys.readouterr().out
    assert 'Total Return' in out
    assert bt.stats is not None


def test_backtest_optimize_finds_best_param(ohlc_data_trending):
    bt = Backtest(ohlc_data_trending, ParamStrategy, cash=10_000, commission=NoCommission())
    best_params, best_stats, all_results = bt.optimize(
        maximize='Total Return [%]', threshold=[2, 5, 10],
    )
    assert best_params is not None
    assert best_stats is not None
    assert len(all_results) == 3


def test_backtest_optimize_constraint_filters_combinations(ohlc_data_trending):
    bt = Backtest(ohlc_data_trending, ParamStrategy, cash=10_000, commission=NoCommission())
    _, _, all_results = bt.optimize(
        maximize='Total Return [%]',
        constraint=lambda p: p['threshold'] != 5,
        threshold=[2, 5, 10],
    )
    # threshold=5 should be filtered out
    assert len(all_results) == 2
    assert all(r['params']['threshold'] != 5 for r in all_results)
