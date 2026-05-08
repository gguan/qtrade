"""End-to-end demo of the v0.6 outputs: heatmap + HTML report + analytics.

Run a small grid search, plot the parameter landscape, then export a
single-file HTML report. Uses synthetic data so it works offline; swap
in `from_yfinance` or `from_akshare_stock_a` for real markets.

    python examples/optimize_and_report.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qtrade.analytics import (
    entries_by_weekday,
    hold_duration_distribution,
    win_loss_feature_comparison,
)
from qtrade.backtest import Backtest, Strategy


class SMAStrategy(Strategy):
    n1 = 5
    n2 = 20

    def prepare(self):
        for df in self._data.values():
            df[f"SMA_{self.n1}"] = df["Close"].rolling(self.n1).mean()
            df[f"SMA_{self.n2}"] = df["Close"].rolling(self.n2).mean()

    def on_bar_close(self):
        s1_prev = self.data[f"SMA_{self.n1}"].iloc[-2]
        s2_prev = self.data[f"SMA_{self.n2}"].iloc[-2]
        s1_last = self.data[f"SMA_{self.n1}"].iloc[-1]
        s2_last = self.data[f"SMA_{self.n2}"].iloc[-1]

        if s1_prev < s2_prev and s1_last > s2_last and self.position.size == 0:
            self.buy(size=10)
        elif s1_prev > s2_prev and s1_last < s2_last and self.position.size > 0:
            self.close()


def _synthetic_data(n_days: int = 250, seed: int = 7) -> pd.DataFrame:
    """Random walk with mild trend — gives the SMA strategy something to do."""
    rng = np.random.default_rng(seed)
    drift = 0.05
    noise = rng.normal(0, 1, size=n_days)
    close = 100 + np.cumsum(drift + noise)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Open": close - 0.4,
            "High": close + 1.2,
            "Low": close - 1.2,
            "Close": close,
            "Volume": rng.integers(800, 1200, size=n_days),
        },
        index=dates,
    )


if __name__ == "__main__":
    data = _synthetic_data()
    bt = Backtest(data, SMAStrategy, cash=10_000)

    # ----- 1. Grid search -----
    print("Running grid search...")
    best_params, best_stats, results = bt.optimize(
        maximize="Sharpe Ratio",
        n1=range(3, 18, 3),
        n2=range(15, 60, 5),
    )
    print(f"Best params: {best_params}")
    print(f"Best Sharpe: {best_stats['Sharpe Ratio']:.3f}")

    # ----- 2. Heatmap -----
    bt.plot_heatmap(
        results,
        x="n1",
        y="n2",
        metric="Sharpe Ratio",
        filename="sharpe_heatmap.html",
        show_plot=False,
    )
    print("Heatmap saved → sharpe_heatmap.html")

    # ----- 3. Re-run at the best params and export the report -----
    bt = Backtest(data, SMAStrategy, cash=10_000)
    bt.run(**best_params)

    print("\n--- Hold duration distribution (hours) ---")
    print(hold_duration_distribution(bt.broker))
    print("\n--- Entries by weekday ---")
    print(entries_by_weekday(bt.broker, metric="profit_sum"))
    print("\n--- Winners vs losers ---")
    print(win_loss_feature_comparison(bt.broker))

    bt.export_report("backtest_report.html", strategy_name="SMA Crossover")
    print("\nReport saved → backtest_report.html")
