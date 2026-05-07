"""Performance benchmarks (opt-in).

Not run by the default ``pytest`` invocation — run them explicitly with::

    pytest tests/benchmarks/

Use ``--benchmark-autosave`` to write per-run JSON to ``.benchmarks/`` and
``--benchmark-compare`` to compare against earlier saved runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtrade.backtest import Backtest, Strategy
from qtrade.core import Broker, NoCommission, Order


def _ohlc(n: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic n-bar OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    return pd.DataFrame(
        {
            'Open': close - 0.5,
            'High': close + 1.0,
            'Low': close - 1.0,
            'Close': close,
            'Volume': (1000 + rng.integers(0, 100, size=n)).astype(int),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Broker hot-path: process_bar with no orders.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n_bars', [1_000, 10_000])
def test_bench_process_bar_no_orders(benchmark, n_bars):
    data = _ohlc(n_bars)

    def run():
        broker = Broker(data, cash=10_000, commission=NoCommission(),
                        margin_ratio=1.0, trade_on_close=True)
        for ts in data.index:
            broker.process_bar(ts)
        return broker

    benchmark(run)


# ---------------------------------------------------------------------------
# Broker hot-path: process_bar with one persistent open trade across all bars.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n_bars', [1_000, 10_000])
def test_bench_process_bar_with_open_trade(benchmark, n_bars):
    data = _ohlc(n_bars)

    def run():
        broker = Broker(data, cash=10_000, commission=NoCommission(),
                        margin_ratio=1.0, trade_on_close=True)
        broker._open_trade(entry_price=100.0, entry_date=data.index[0], size=10)
        for ts in data.index:
            broker.process_bar(ts)
        return broker

    benchmark(run)


# ---------------------------------------------------------------------------
# End-to-end Backtest with a realistic strategy.
# ---------------------------------------------------------------------------


class _SmaCrossover(Strategy):
    n1 = 5
    n2 = 20

    def prepare(self):
        # self._data is dict[str, DataFrame]; single-asset → only one entry.
        for df in self._data.values():
            df['sma1'] = df['Close'].rolling(self.n1).mean()
            df['sma2'] = df['Close'].rolling(self.n2).mean()

    def on_bar_close(self):
        s1 = self.data['sma1']
        s2 = self.data['sma2']
        if pd.isna(s1.iloc[-2]) or pd.isna(s2.iloc[-2]):
            return
        crossed_up = s1.iloc[-2] <= s2.iloc[-2] and s1.iloc[-1] > s2.iloc[-1]
        crossed_down = s1.iloc[-2] >= s2.iloc[-2] and s1.iloc[-1] < s2.iloc[-1]
        if crossed_up and self.position.size <= 0:
            self.close()
            self.buy(size=10)
        elif crossed_down and self.position.size >= 0:
            self.close()


@pytest.mark.parametrize('n_bars', [1_000, 10_000])
def test_bench_backtest_sma_crossover(benchmark, n_bars):
    data = _ohlc(n_bars)

    def run():
        bt = Backtest(data, _SmaCrossover, cash=10_000,
                      commission=NoCommission(), trade_on_close=True)
        bt.run()
        return bt

    benchmark(run)


# ---------------------------------------------------------------------------
# Multi-asset overhead: same bar count, 4 assets.
# ---------------------------------------------------------------------------


class _PerAssetSma(Strategy):
    n1 = 5
    n2 = 20

    def prepare(self):
        for df in self._data.values():
            df['sma1'] = df['Close'].rolling(self.n1).mean()
            df['sma2'] = df['Close'].rolling(self.n2).mean()

    def on_bar_close(self):
        for asset in self.assets:
            df = self.data_by_asset[asset]
            if pd.isna(df['sma1'].iloc[-2]) or pd.isna(df['sma2'].iloc[-2]):
                continue
            crossed_up = df['sma1'].iloc[-2] <= df['sma2'].iloc[-2] and df['sma1'].iloc[-1] > df['sma2'].iloc[-1]
            crossed_down = df['sma1'].iloc[-2] >= df['sma2'].iloc[-2] and df['sma1'].iloc[-1] < df['sma2'].iloc[-1]
            pos = self.positions[asset].size
            if crossed_up and pos <= 0:
                self.close(asset)
                self.buy(asset, size=10)
            elif crossed_down and pos >= 0:
                self.close(asset)


def test_bench_backtest_4_asset_portfolio(benchmark):
    n_bars = 5_000
    data = {f"A{i}": _ohlc(n_bars, seed=10 + i) for i in range(4)}

    def run():
        bt = Backtest(data, _PerAssetSma, cash=100_000,
                      commission=NoCommission(), trade_on_close=True)
        bt.run()
        return bt

    benchmark(run)


# ---------------------------------------------------------------------------
# Order placement throughput.
# ---------------------------------------------------------------------------


def test_bench_place_orders_throughput(benchmark):
    data = _ohlc(2_000)
    broker = Broker(data, cash=1_000_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    broker.process_bar(data.index[0])

    def run():
        # Alternating buy/sell, 100 round-trips
        for i in range(100):
            broker.place_orders(Order(size=1) if i % 2 == 0 else Order(size=-1))

    benchmark(run)
