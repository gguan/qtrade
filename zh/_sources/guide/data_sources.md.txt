# Loading data

A `Backtest` accepts any `pd.DataFrame` (single-asset) or
`dict[str, pd.DataFrame]` (multi-asset) as long as the index is a
`DatetimeIndex` and the columns include `Open, High, Low, Close` (Volume is
nice-to-have). You can build that yourself from a CSV / database / API,
but for the common cases QTrade ships two loaders.

## US markets — `from_yfinance`

Install the `[data]` extra:

```bash
pip install "qtrade-lib[data]"
```

```python
from qtrade.data import from_yfinance

data = from_yfinance(
    ["AAPL", "MSFT", "GC=F"],          # one ticker or a list
    start="2023-01-01",
    end="2024-12-31",
    interval="1d",                      # "1h", "5m", etc. all supported
    auto_adjust=True,                   # split / dividend adjusted closes
)
# data is a dict[str, DataFrame] aligned on the intersection of indexes
# — pass straight to Backtest.
```

What the helper does for you:

- Loops over symbols, raises a clean `ValueError` if any returned empty.
- Drops `NaN` rows from each frame (yfinance occasionally returns them
  on early-listing dates).
- Aligns every frame on the **intersection** of their `DatetimeIndex` so
  `Backtest` accepts them as a multi-asset input.

## Chinese markets — `from_akshare_*`

For A-shares and Chinese futures, install the `[cn]` extra:

```bash
pip install "qtrade-lib[cn]"
```

### A-shares

```python
from qtrade.data import from_akshare_stock_a

data = from_akshare_stock_a(
    ["600519", "000001"],                 # 6-digit codes, no exchange prefix
    start="2024-01-01",                   # accepts "YYYY-MM-DD" or "YYYYMMDD"
    end="2024-12-31",
    period="daily",                       # or "weekly" / "monthly"
    adjust="qfq",                         # 前复权 (default; recommended for backtests)
)
```

### Futures (main-contract continuous)

```python
from qtrade.contracts import Contract
from qtrade.data import from_akshare_futures

data = from_akshare_futures(
    ["AU0", "RB0"],                       # main-contract codes (trailing 0)
    start="2024-01-01",
    end="2024-12-31",
)

# Pair with Contract specs for the right multipliers / margins:
SHFE_AU = Contract(multiplier=1000, margin_ratio=0.08, name="SHFE Gold")
SHFE_RB = Contract(multiplier=10,   margin_ratio=0.10, name="SHFE Rebar")

bt = Backtest(data, MyStrategy, contracts={"AU0": SHFE_AU, "RB0": SHFE_RB})
```

Same shape as the yfinance loader — Chinese column names are translated to
standard `Open, High, Low, Close, Volume`, the index becomes a tz-naive
`DatetimeIndex`, and frames are aligned on intersection.

## Aligning external data

If you've assembled `dict[str, DataFrame]` yourself (e.g. from a CSV folder)
and the indexes don't perfectly match, run:

```python
from qtrade.data import align_indexes
data = align_indexes(data)
```

`align_indexes` reindexes every frame on the intersection of their indexes
— a no-op when they already match, a one-liner when they don't.

## Other data sources

The two helpers here are convenience wrappers, not framework constraints.
For Tushare, JoinQuant, IB, or your own pipeline, just produce a
`dict[str, DataFrame]` (each indexed by datetime, OHLC capitalized) and
hand it to `Backtest`. If the indexes aren't aligned, `align_indexes`
takes care of it.
