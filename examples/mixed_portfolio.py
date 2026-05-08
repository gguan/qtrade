"""Mixed portfolio: stocks + multi-category futures with different multipliers.

Demonstrates how to model a realistic portfolio that holds:
- AAPL (stock, multiplier=1, margin_ratio=1.0 — no leverage)
- ES   (E-mini S&P 500 futures, multiplier=50, ~5% initial margin)
- GC   (COMEX gold futures, multiplier=100, ~5% initial margin)
- CL   (NYMEX crude oil futures, multiplier=1000, ~10% initial margin)

Run with:
    pip install yfinance
    python examples/mixed_portfolio.py
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from qtrade.backtest import Backtest, Strategy
from qtrade.core import FixedCommission


CONTRACT_SPECS = {
    # symbol: (yfinance_ticker, multiplier, margin_ratio_approx)
    "AAPL": ("AAPL", 1,    1.0),     # stock, no leverage
    "ES":   ("ES=F", 50,   0.05),    # E-mini S&P
    "GC":   ("GC=F", 100,  0.05),    # COMEX gold
    "CL":   ("CL=F", 1000, 0.10),    # NYMEX crude
}


def load_data() -> dict[str, pd.DataFrame]:
    """Pull each ticker from yfinance and align on the common index."""
    raw = {
        sym: yf.download(spec[0], start="2023-01-01", end="2024-01-01",
                         interval="1d", multi_level_index=False)
        for sym, spec in CONTRACT_SPECS.items()
    }
    common = raw["AAPL"].index
    for df in raw.values():
        common = common.intersection(df.index)
    return {sym: df.loc[common] for sym, df in raw.items()}


class EqualWeightMomentum(Strategy):
    """Hold 1 contract / share of each asset whose 20-bar momentum is positive."""

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
    data = load_data()

    multiplier = {sym: spec[1] for sym, spec in CONTRACT_SPECS.items()}
    margin_ratio = {sym: spec[2] for sym, spec in CONTRACT_SPECS.items()}

    bt = Backtest(
        data,
        EqualWeightMomentum,
        cash=200_000,
        commission=FixedCommission(2.50),     # round-turn for futures; OK for stocks too
        margin_ratio=margin_ratio,
        contract_multiplier=multiplier,
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
    main()
