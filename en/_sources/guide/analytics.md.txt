# Trade analytics

The `qtrade.analytics` module slices closed trades by time, duration, and
outcome — turning the raw trade history into questions you actually want
answered after a backtest:

- *How long do my winners hold vs my losers?*
- *Are my entries clustered on Mondays? In specific months?*
- *What's different about the trades I lose money on — bigger size, longer
  hold, lower entry price?*

Every function takes either a `Broker` (so you can pass `bt.broker`
directly) or the trade-history `DataFrame` from
`Broker.get_trade_history()`. They return plain pandas objects so you
can `.plot()` them inline, hand them to seaborn, or render them yourself.

## Hold duration distribution

```python
from qtrade.analytics import hold_duration_distribution

print(hold_duration_distribution(bt.broker))
#           count   mean  median    std   min   25%   50%   75%   max
# all        42.0  18.50    12.0  14.20   1.0   8.0  12.0  24.0  72.0
# winners    25.0  22.10    18.0  13.80   2.0  12.0  18.0  30.0  72.0
# losers     17.0  13.20     8.0  12.50   1.0   4.0   8.0  16.0  48.0
```

All durations are reported in **hours** so intraday and daily strategies
share one scale. Pass `by_outcome=False` to get just the `"all"` row.

A common pattern: do my winners hold *longer* than my losers? If so, your
strategy is probably riding trends well — and you might be cutting some
losers too late. If winners are *shorter*, the opposite — you're scalping
small wins and letting losers run.

## Calendar bias

```python
from qtrade.analytics import entries_by_weekday, entries_by_month

# How many entries fire each weekday?
entries_by_weekday(bt.broker, metric="count")
# Monday       12
# Tuesday       8
# ...

# Are some weekdays consistently negative?
entries_by_weekday(bt.broker, metric="profit_sum")

# Win rate by weekday — useful for a "skip Friday" filter check
entries_by_weekday(bt.broker, metric="win_rate")

# Same shape, monthly
entries_by_month(bt.broker, metric="profit_mean")
```

Available `metric` values:

- `"count"` — number of entries per bucket.
- `"profit_sum"` — total profit per bucket.
- `"profit_mean"` — average profit per bucket.
- `"win_rate"` — fraction of trades closing positive (0.0–1.0).

Empty buckets are filled with 0 (count / profit_sum) or NaN
(profit_mean / win_rate). The result is always indexed in calendar order
(Monday → Sunday, January → December).

## Winning vs losing trade comparison

```python
from qtrade.analytics import win_loss_feature_comparison

win_loss_feature_comparison(bt.broker)
#                  winners_mean  losers_mean   diff  winners_median  losers_median
# Size                    85.20        92.10  -6.90           80.00          90.00
# Entry Price            172.40       175.80  -3.40          170.50         174.00
# Exit Price             184.30       170.20  14.10          181.00         169.50
# Duration (h)            22.10        13.20   8.90           18.00           8.00
```

`Size` is compared as `abs(size)` so longs and shorts pool by magnitude.
`diff` is `winners_mean − losers_mean` — quick visual scan to see which
features separate good from bad trades.

To compare extra columns from your trade history (e.g. a custom tag):

```python
trades = bt.get_trade_history()
trades["RSI at Entry"] = ...   # populate however you like
win_loss_feature_comparison(trades, extra_features=["RSI at Entry"])
```

## What's not modeled

- **Time-of-day distribution.** For intraday strategies you'd want
  entries-by-hour. Easy to roll yourself: `pd.to_datetime(trades['Entry Time']).dt.hour.value_counts()`.
- **Equity-curve drawdown analysis.** That's portfolio-level, not
  trade-level — see `calculate_stats` for the equity-curve metrics.
- **Significance testing.** The win/loss comparison shows magnitudes,
  not p-values. Consider sample size before drawing conclusions —
  with 10 winners and 10 losers you're in noise territory.
