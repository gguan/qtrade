"""Multi-asset mean-reversion strategy across two tickers.

Demonstrates:
- Passing a ``dict[str, pd.DataFrame]`` to ``Backtest`` for portfolio backtests.
- Per-asset entry/exit logic via ``self.buy(asset, ...)`` / ``self.sell(asset, ...)``.
- ``calculate_stats_per_asset`` for per-asset breakdowns.
- Walk-forward optimization over a parameter grid.

Run it with:

    pip install yfinance
    python examples/portfolio_strategy.py
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from qtrade.backtest import Backtest, Strategy
from qtrade.core import PercentageCommission
from qtrade.utils.stats import calculate_stats_per_asset


class PortfolioMeanReversion(Strategy):
    """Z-score mean reversion, applied independently per asset."""

    def prepare(self):
        for df in self._data.values():
            df['ma'] = df['Close'].rolling(self.window).mean()
            df['std'] = df['Close'].rolling(self.window).std()
            df['z'] = (df['Close'] - df['ma']) / df['std']

    def on_bar_close(self):
        for asset in self.assets:
            df = self.data_by_asset[asset]
            z = df['z'].iloc[-1]
            if pd.isna(z):
                continue
            pos = self.positions[asset].size
            if abs(z) < 0.3 and pos != 0:
                self.close(asset)
            elif z < -1.0 and pos <= 0:
                self.close(asset)
                self.buy(asset, size=10)
            elif z > 1.0 and pos >= 0:
                self.close(asset)
                self.sell(asset, size=10)


def load_data() -> dict[str, pd.DataFrame]:
    """Pull two tickers from yfinance and align on a common index."""
    aapl = yf.download("AAPL", start="2023-01-01", end="2024-01-01",
                       interval="1d", multi_level_index=False)
    nvda = yf.download("NVDA", start="2023-01-01", end="2024-01-01",
                       interval="1d", multi_level_index=False)
    common = aapl.index.intersection(nvda.index)
    return {"AAPL": aapl.loc[common], "NVDA": nvda.loc[common]}


def main() -> None:
    data = load_data()

    bt = Backtest(
        data,
        PortfolioMeanReversion,
        cash=200_000,
        commission=PercentageCommission(percentage=0.0005),
        margin_ratio=0.5,
        trade_on_close=True,
    )
    bt.run(window=20)

    print("\n=== Portfolio stats ===")
    bt.show_stats()

    print("\n=== Per-asset trade breakdown ===")
    for asset, stats in calculate_stats_per_asset(bt.broker).items():
        print(f"--- {asset} ---")
        for k, v in stats.items():
            print(f"  {k:25}: {v}")

    print("\n=== Walk-forward (window=20, train=120, test=30) ===")
    wf = bt.walk_forward_optimize(
        train_window=120,
        test_window=30,
        maximize='Sharpe Ratio',
        window=[10, 20, 30],
    )
    for w in wf['windows']:
        print(
            f"  {w['test_start'].date()} → {w['test_end'].date()}: "
            f"params={w['best_params']}, "
            f"OoS Total Return={w['test_stats']['Total Return [%]']:.2f}%"
        )
    print(f"\n  summary: {wf['summary']}")

    bt.plot()


if __name__ == "__main__":
    main()
