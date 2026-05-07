"""Tests for qtrade.utils.stats."""

import pandas as pd
import pytest

from qtrade.core import Broker, NoCommission, PercentageCommission
from qtrade.utils.stats import calculate_stats, display_metrics


@pytest.fixture
def broker_with_winning_trade(ohlc_data_trending):
    broker = Broker(ohlc_data_trending, cash=10_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    for ts in ohlc_data_trending.index:
        broker.process_bar(ts)
    broker._open_trade(entry_price=100.0, entry_date=ohlc_data_trending.index[0], size=10)
    broker.close_all_positions()
    return broker


@pytest.fixture
def broker_with_losing_trade(ohlc_data_trending):
    broker = Broker(ohlc_data_trending, cash=10_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    for ts in ohlc_data_trending.index:
        broker.process_bar(ts)
    broker._open_trade(entry_price=200.0, entry_date=ohlc_data_trending.index[0], size=10)
    broker.close_all_positions()
    return broker


@pytest.fixture
def broker_no_trades(ohlc_data_trending):
    broker = Broker(ohlc_data_trending, cash=10_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    for ts in ohlc_data_trending.index:
        broker.process_bar(ts)
    return broker


EXPECTED_KEYS = {
    'Start', 'End', 'Duration', 'Start Value', 'End Value',
    'Total Return [%]', 'Total Commission Cost[%]', 'Buy & Hold Return [%]',
    'Return (Ann.) [%]', 'Volatility (Ann.) [%]',
    'Max Drawdown [%]', 'Max Drawdown Duration',
    'Total Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
    'Avg Winning Trade [%]', 'Avg Losing Trade [%]',
    'Avg Winning Trade Duration', 'Avg Losing Trade Duration',
    'Profit Factor', 'Expectancy', 'Sharpe Ratio', 'Sortino Ratio',
    'Calmar Ratio', 'Omega Ratio',
}


def test_calculate_stats_returns_all_expected_keys(broker_with_winning_trade):
    stats = calculate_stats(broker_with_winning_trade)
    assert EXPECTED_KEYS.issubset(stats.keys())


def test_calculate_stats_basic_metrics(broker_with_winning_trade, ohlc_data_trending):
    stats = calculate_stats(broker_with_winning_trade)
    assert stats['Start'] == ohlc_data_trending.index[0]
    assert stats['Start Value'] == 10_000
    assert stats['End Value'] > stats['Start Value']
    assert stats['Total Return [%]'] > 0


def test_calculate_stats_buy_and_hold_return(broker_with_winning_trade, ohlc_data_trending):
    stats = calculate_stats(broker_with_winning_trade)
    expected = (ohlc_data_trending['Close'].iloc[-1] - ohlc_data_trending['Close'].iloc[0]) \
        / ohlc_data_trending['Close'].iloc[0] * 100
    assert stats['Buy & Hold Return [%]'] == pytest.approx(expected)


def test_calculate_stats_winning_trade_metrics(broker_with_winning_trade):
    stats = calculate_stats(broker_with_winning_trade)
    assert stats['Total Trades'] == 1
    assert stats['Win Rate [%]'] == 100.0
    assert stats['Best Trade [%]'] > 0
    # No losing trades → avg loss 0, profit factor / expectancy non-NaN paths
    assert stats['Avg Losing Trade [%]'] == 0


def test_calculate_stats_losing_trade_metrics(broker_with_losing_trade):
    stats = calculate_stats(broker_with_losing_trade)
    assert stats['Total Trades'] == 1
    assert stats['Win Rate [%]'] == 0.0
    assert stats['Worst Trade [%]'] < 0


def test_calculate_stats_no_trades(broker_no_trades):
    stats = calculate_stats(broker_no_trades)
    assert stats['Total Trades'] == 0
    assert stats['Win Rate [%]'] == 0


def test_calculate_stats_with_commission(ohlc_data_trending):
    broker = Broker(ohlc_data_trending, cash=10_000,
                    commission=PercentageCommission(percentage=0.001),
                    margin_ratio=1.0, trade_on_close=True)
    for ts in ohlc_data_trending.index:
        broker.process_bar(ts)
    # No filled orders means total_commissions is sum over empty list = 0
    stats = calculate_stats(broker)
    assert stats['Total Commission Cost[%]'] == 0


def test_display_metrics_prints_all(capsys):
    metrics = {'Foo': 1, 'Bar': 2.5, 'Baz': 'qux'}
    display_metrics(metrics)
    out = capsys.readouterr().out
    assert 'Foo' in out and '1' in out
    assert 'Bar' in out and '2.5' in out
    assert 'Baz' in out and 'qux' in out
