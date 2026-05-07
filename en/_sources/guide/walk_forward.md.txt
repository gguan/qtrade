# Walk-forward optimization

`Backtest.optimize()` picks the best parameter combination over the
*entire* dataset — and reports its in-sample performance. This is the
classic recipe for fitting noise. The "best Sharpe 2.4" you see almost
never holds up live.

`Backtest.walk_forward_optimize()` runs the same grid search inside
rolling windows and reports performance on data the strategy was
never tuned on:

```
                        train          test
window 0:   [─────────────────────][──────]
window 1:           [─────────────────────][──────]
window 2:                   [─────────────────────][──────]
                                                step
```

For every window pair, QTrade picks the best params on the train slice
and evaluates them on the immediately following test slice. The
aggregate of the test slices is your honest, out-of-sample (OoS)
estimate of how the strategy actually performs.

## API

```python
result = bt.walk_forward_optimize(
    train_window=200,
    test_window=50,
    maximize='Sharpe Ratio',
    step=50,
    constraint=lambda p: p['n1'] < p['n2'],
    n1=range(5, 30, 5),
    n2=range(20, 60, 10),
)
```

Required:

- **`train_window`**: bars per training slice.
- **`test_window`**: bars per test slice (immediately follows the train slice).
- **`maximize`**: name of the metric from `calculate_stats()` to
  maximize within each train slice. Common choices: `Sharpe Ratio`,
  `Total Return [%]`, `Calmar Ratio`. See the
  [stats glossary](stats_glossary.md) for what each means.
- **`**params_grid`**: same syntax as `optimize()` — keyword arguments
  whose values are iterables of candidates.

Optional:

- **`step`**: how many bars to advance between consecutive train starts.
  Defaults to `test_window` (non-overlapping test windows). Smaller
  values give overlapping (correlated) test windows; larger values
  leave gaps.
- **`constraint`**: filter `lambda p: bool` applied to each parameter
  dict before evaluating. Use this to skip nonsensical combinations
  (e.g. fast SMA window > slow SMA window).

## Reading the result

`result` is a dict:

```python
result['windows']    # list of per-window dicts
result['summary']    # aggregate OoS metrics
```

Each window dict contains:

| Key | Value |
|---|---|
| `train_start`, `train_end` | Timestamps of the train slice. |
| `test_start`, `test_end` | Timestamps of the test slice. |
| `best_params` | The parameter combo that maximized `maximize` on the train slice. |
| `train_stats` | Full stats dict from the train run with `best_params`. |
| `test_stats` | Full stats dict from the test run. |
| `test_equity` | The test slice's `equity_history` Series. |

The summary:

| Key | Meaning |
|---|---|
| `n_windows` | Total number of train/test pairs evaluated. |
| `mean_oos_return` | Average `Total Return [%]` across test windows. |
| `hit_rate` | Fraction of test windows with positive return. |
| `min_oos_return`, `max_oos_return` | Best and worst test windows. |

## A typical workflow

```python
from qtrade.backtest import Backtest
from qtrade.utils.stats import calculate_stats

# 1. Cheap in-sample search to confirm the strategy is even worth tuning.
best_params, best_stats, _ = bt.optimize(
    maximize='Sharpe Ratio',
    n1=range(5, 30, 5),
    n2=range(20, 60, 10),
    constraint=lambda p: p['n1'] < p['n2'],
)
print("In-sample Sharpe:", best_stats['Sharpe Ratio'])

# 2. Walk-forward to get the realistic number.
result = bt.walk_forward_optimize(
    train_window=200,
    test_window=50,
    maximize='Sharpe Ratio',
    n1=range(5, 30, 5),
    n2=range(20, 60, 10),
    constraint=lambda p: p['n1'] < p['n2'],
)
print("OoS hit rate:", result['summary']['hit_rate'])
print("OoS mean return:", result['summary']['mean_oos_return'], '%')

# 3. Inspect window-by-window for parameter stability.
for w in result['windows']:
    print(
        f"{w['test_start'].date()}–{w['test_end'].date()}: "
        f"params={w['best_params']} "
        f"oos_return={w['test_stats']['Total Return [%]']:.2f}%"
    )
```

If the in-sample Sharpe and OoS mean are wildly different, you've
found your overfit. The OoS number is the one to trust.

## Choosing the windows

The right `train_window` and `test_window` depend on:

- **Strategy "memory"**: how many bars does the strategy need to fit a
  parameter? A 20-bar SMA needs at least ~50 bars to give the system
  some room. A 200-bar lookback model needs a much larger train_window.
- **Market regime length**: a train window that's too small only sees
  one regime; too large straddles multiple regimes (averaging conflicting
  optima). 6–12 months of daily bars is a typical starting point.
- **Number of windows you can afford**: longer windows = fewer windows,
  but more data per fit; shorter windows = more windows but each is
  noisier. Aim for **at least 5–10 windows** so the summary statistics
  are meaningful.

Rule of thumb: `train_window ≈ 4–10 × test_window` is a common ratio.

## Common pitfalls

### "OoS hit rate is 0.5"

Coin flip. Either the strategy genuinely doesn't have edge, or your
parameter grid doesn't cover the meaningful space. Try:

1. Look at *which* params win in each window — if they're wildly
   different across windows, the strategy isn't stable.
2. Widen the grid. If the optimum keeps landing at the edge, you're
   not exploring far enough.
3. Try a longer `train_window`.

### "Mean OoS return is positive but min is catastrophic"

A common signature of strategies that work most of the time but blow
up periodically (e.g. martingale-like betting, undefended short vol).
The mean hides the tail.

Always look at the per-window list, not just the summary.

### "Walk-forward Sharpe is much lower than in-sample Sharpe"

Expected and normal. The gap is your overfit. A 2:1 ratio
(in-sample 2.0 → OoS 1.0) is fine; 5:1 means you're fitting noise.
Reduce parameter freedom (smaller grid) or extend the strategy's
inductive bias (use a constraint, share parameters across assets, etc.).

### "Step smaller than test_window"

This produces *overlapping* test windows. The summary metrics
double-count bars and inflate `n_windows`. Sometimes you want this
(e.g. for adaptive walk-forward that retrains weekly), but understand
that the windows aren't statistically independent.

## See also

- [Stats glossary](stats_glossary.md) — pick the right `maximize` metric.
- [Getting Started](getting_started.md) — basic Backtest workflow.
- [`examples/portfolio_strategy.py`](https://github.com/gguan/qtrade/blob/main/examples/portfolio_strategy.py)
  — multi-asset walk-forward in a runnable file.
