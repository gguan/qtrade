"""Mixed portfolio: stocks + multi-category futures, end-to-end with the
high-level Contract API and yfinance data loader.

Demonstrates:

- :func:`qtrade.data.from_yfinance` — multi-ticker download + aligned dict
  in one call.
- :mod:`qtrade.contracts` registry — built-in specs (``ES_CME``, ``GC_COMEX``,
  ``CL_NYMEX``) plus ``STOCK_CASH`` defaults for assets you don't list.
- ``Backtest(..., contracts={...})`` for clean per-asset configuration.

Run with:
    pip install "qtrade-lib[data]"
    python examples/mixed_portfolio.py
"""

from __future__ import annotations

import pandas as pd

from qtrade.backtest import Backtest, Strategy
from qtrade.contracts import CL_NYMEX, ES_CME, GC_COMEX, STOCK_CASH
from qtrade.core import FixedCommission
from qtrade.data import from_yfinance


class EqualWeightMomentum(Strategy):
    """Hold 1 share / contract of each asset whose 20-bar momentum is positive."""

    def prepare(self):
        for df in self._data.values():
            df['mom'] = df['Close'].pct_change(self.lookback)

    def on_bar_close(self):
        for asset in self.assets:
            df = self.data_by_asset[asset]
            if pd.isna(df['mom'].iloc[-1]):
                continue
            should_hold = df['mom'].iloc[-1] > 0
            current = self.positions[asset].size
            if should_hold and current == 0:
                self.buy(asset, size=1)
            elif not should_hold and current > 0:
                self.close(asset)


def main() -> None:
    # 1. Download — one call, aligned indexes, ready to feed Backtest.
    data = from_yfinance(
        ["AAPL", "ES=F", "GC=F", "CL=F"],
        start="2023-01-01",
        end="2024-01-01",
    )

    # 2. Map symbols to contract specs. AAPL is omitted on purpose — assets
    #    not in this dict default to STOCK_CASH (no leverage, multiplier=1).
    contracts = {
        "ES=F": ES_CME,
        "GC=F": GC_COMEX,
        "CL=F": CL_NYMEX,
        # "AAPL" → STOCK_CASH automatically
    }
    # Equivalent explicit form:
    #     contracts["AAPL"] = STOCK_CASH

    bt = Backtest(
        data,
        EqualWeightMomentum,
        cash=200_000,
        commission=FixedCommission(2.50),
        contracts=contracts,
        trade_on_close=True,
    )
    bt.run(lookback=20)

    print("\n=== Portfolio stats ===")
    bt.show_stats()

    print("\n=== Per-asset trade summary ===")
    df = bt.get_trade_history()
    if not df.empty:
        print(df.groupby('Asset').agg(
            n_trades=('Profit', 'size'),
            total_profit=('Profit', 'sum'),
            avg_profit=('Profit', 'mean'),
        ))


if __name__ == "__main__":
    # Re-export STOCK_CASH so the import is tested when the module is loaded.
    _ = STOCK_CASH
    main()
