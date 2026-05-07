# Multi-asset / portfolio backtests

QTrade supports portfolio backtests where a single strategy trades multiple
assets sharing one cash pool. The single-asset workflow from
[Getting Started](getting_started.md) is unchanged — multi-asset is enabled
by passing a dict of DataFrames instead of a single one.

## Data

Pass a `dict[str, pd.DataFrame]` keyed by asset symbol. Every DataFrame
needs the standard OHLCV columns and **must share an identical
`DatetimeIndex`** (align/reindex up-front if your sources don't):

```python
import yfinance as yf
import pandas as pd

aapl = yf.download("AAPL", start="2023-01-01", end="2024-01-01",
                   interval="1d", multi_level_index=False)
nvda = yf.download("NVDA", start="2023-01-01", end="2024-01-01",
                   interval="1d", multi_level_index=False)

# Inner-join indexes to ensure both assets are present on every bar.
common_index = aapl.index.intersection(nvda.index)
data = {
    "AAPL": aapl.loc[common_index],
    "NVDA": nvda.loc[common_index],
}
```

`Backtest` validates each DataFrame independently and raises a clear error
if the indexes diverge.

## Strategy

In multi-asset mode the strategy must specify which asset each order
targets. Iterate `self.assets` and use `self.positions[symbol]` /
`self.data_by_asset[symbol]`:

```python
from qtrade.backtest import Strategy

class PortfolioMeanReversion(Strategy):

    def prepare(self):
        # Compute indicators per asset.
        for asset, df in self._data.items():
            df['ma'] = df['Close'].rolling(self.window).mean()
            df['z']  = (df['Close'] - df['ma']) / df['Close'].rolling(self.window).std()

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
```

API overview when running multi-asset:

| Single-asset shorthand | Multi-asset equivalent          |
|------------------------|---------------------------------|
| `self.buy(size=10)`    | `self.buy("AAPL", size=10)`     |
| `self.sell(size=5)`    | `self.sell("AAPL", size=5)`    |
| `self.close()`         | `self.close()` — closes every asset, or `self.close("AAPL")` for one |
| `self.position`        | `self.positions["AAPL"]`        |
| `self.data`            | `self.data_by_asset["AAPL"]`    |

`self.position` and `self.data` raise `AttributeError` in multi-asset
mode with a message pointing at the dict accessor.

## Running it

```python
from qtrade.backtest import Backtest
from qtrade.core import PercentageCommission

bt = Backtest(
    data,
    PortfolioMeanReversion,
    cash=200_000,
    commission=PercentageCommission(percentage=0.0005),
    margin_ratio=0.5,
    trade_on_close=True,
)
bt.run(window=10)

bt.show_stats()  # portfolio-level metrics
```

The `Buy & Hold Return [%]` shown in `show_stats()` is the
**equal-weighted average** across all assets — a sensible default
benchmark for portfolio strategies.

## Per-asset breakdown

For trade-level statistics on each asset:

```python
from qtrade.utils.stats import calculate_stats_per_asset

per = calculate_stats_per_asset(bt.broker)
for asset, stats in per.items():
    print(f"--- {asset} ---")
    print(f"  Trades: {stats['Total Trades']}")
    print(f"  Win Rate: {stats['Win Rate [%]']:.1f}%")
    print(f"  Buy & Hold: {stats['Buy & Hold Return [%]']:.2f}%")
```

Aggregate metrics (Sharpe, Sortino, drawdown, equity curve) are inherently
portfolio-wide and only available via `calculate_stats(broker)` —
attribution to individual assets requires extra accounting that QTrade
intentionally doesn't try to do.

## Plot

`bt.plot()` renders a multi-panel layout:
- **Top:** portfolio equity vs equal-weighted Buy & Hold (with drawdown overlay).
- **Middle:** trade-returns scatter across all assets.
- **Bottom:** one OHLC panel per asset, stacked vertically with linked x-axes.

Each per-asset panel shows the win/lose trade boxes and buy/sell markers
filtered to that asset's trades only.

## Trade history

`bt.get_trade_history()` returns a DataFrame with an `Asset` column so
you can filter or group by symbol downstream:

```python
df = bt.get_trade_history()
print(df.groupby('Asset')['Profit'].sum())
```
