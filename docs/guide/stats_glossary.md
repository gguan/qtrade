# Stats glossary

`Backtest.show_stats()` and `qtrade.utils.stats.calculate_stats()` print
a dict with ~25 metrics. This page explains what each one means, how
it's computed, and a rough guide to interpretation.

For per-asset breakdowns in a portfolio backtest, see
`calculate_stats_per_asset(broker)` — it returns the trade-level subset
of these metrics keyed by asset symbol.

## Time / value

| Metric | Meaning |
|---|---|
| **Start** | First bar timestamp (from `equity_history.index[0]`). |
| **End** | Last bar processed (`broker.current_time`). |
| **Duration** | `End - Start` as a `pd.Timedelta`. |
| **Start Value** | Initial equity (typically equal to starting cash). |
| **End Value** | Equity at the end of the backtest. |

## Returns

| Metric | Formula | Notes |
|---|---|---|
| **Total Return [%]** | `(End - Start) / Start × 100` | Raw, not annualized. |
| **Total Commission Cost[%]** | Sum of all commissions paid over the run | Helpful for spotting strategies that bleed on fees. |
| **Buy & Hold Return [%]** | `(close[-1] - close[0]) / close[0] × 100` for the asset (single) or **equal-weighted average** across assets (multi) | Benchmark to beat. Strategies losing to B&H typically aren't worth running over passive holding. |
| **Return (Ann.) [%]** | Geometric mean of daily returns, annualized to 252 (stocks) or 365 (crypto/FX) days | Compounded annual growth rate. |
| **Volatility (Ann.) [%]** | `std(daily_returns) × √annual_days × 100` | Standard deviation of returns, scaled to a year. |

The 252 vs 365 annualization factor is auto-detected: if the equity
curve includes weekends as trading days (`>≈ 60%` of weekend bars are
filled), it's treated as 365-day; otherwise 252.

## Risk

| Metric | Formula | Notes |
|---|---|---|
| **Max Drawdown [%]** | `min((equity - cummax(equity)) / cummax(equity)) × 100` | The biggest peak-to-trough decline. Always ≤ 0. |
| **Max Drawdown Duration** | Length of the longest contiguous "below peak" period | Returned as a `pd.Timedelta`. The drawdown that took longest to recover from, not the deepest one. |

## Trade statistics

| Metric | Notes |
|---|---|
| **Total Trades** | Count of closed trades across all assets. |
| **Win Rate [%]** | Percentage of closed trades with positive profit. |
| **Best Trade [%]** | Largest single trade profit (in account currency, not %). The label says "[%]" for legacy reasons — it's a P&L value. |
| **Worst Trade [%]** | Same, but the most negative profit. |
| **Avg Winning Trade [%]** | Mean profit of winning trades. |
| **Avg Losing Trade [%]** | Mean profit of losing trades (negative). |
| **Avg Winning Trade Duration** | Mean time-in-market for winners. |
| **Avg Losing Trade Duration** | Mean time-in-market for losers. |

If "winners hold longer than losers" you have a "let winners run, cut
losers short" pattern. Reverse means you're cutting winners early.

## Performance ratios

These compress return-and-risk into a single number. Different ratios
penalize different things.

### Profit Factor

```
sum(wins) / sum(|losses|)
```

Higher is better. > 1 means you're net positive. > 2 is rare and good.
NaN if there are no losing trades.

### Expectancy

```
(sum(wins) - sum(|losses|)) / total_trades
```

Average dollar profit per trade. Positive expectancy means each trade
is worth taking on average.

### Sharpe Ratio

```
mean(daily_returns) / std(daily_returns) × √annual_days
```

Risk-free rate is assumed to be 0 (override by editing
`__calculate_performance_ratios` in `stats.py` if you need otherwise).
Rough guide:

- **< 0**: losing strategy.
- **0–1**: marginal; might be in-sample noise.
- **1–2**: solid.
- **> 2**: excellent (and usually too good — check for overfitting or
  look-ahead bias).

### Sortino Ratio

```
mean(daily_returns) / std(daily_returns[< 0]) × √annual_days
```

Like Sharpe, but only penalizes downside volatility. Strategies with
asymmetric return distributions (most positive trend-followers) score
better here than on Sharpe.

### Calmar Ratio

```
abs(annualized_return) / abs(max_drawdown)
```

How much annual return per unit of worst-case drawdown. > 1 is decent
(your annual gains exceed your worst dip).

### Omega Ratio

```
sum(returns > threshold) / |sum(returns < threshold)|
```

Threshold is 0 (so: gains divided by losses, summed across all daily
return observations). Captures the full distribution of returns rather
than mean/std summary. > 1 is positive.

## Sanity-checking your stats

A few things to verify before getting excited about a backtest:

- **Total Trades is reasonable** — single-digit trade counts can produce
  Sharpe = 5 by accident. You want enough trades for the metrics to be
  statistically meaningful.
- **Total Return > Buy & Hold** — beating B&H is the bar. Losing to it
  on a long-only strategy means your timing is hurting more than helping.
- **Win Rate × Avg Win > (1 - Win Rate) × |Avg Loss|** — the basic
  expectancy condition. If this fails, your strategy is structurally
  losing.
- **Use `walk_forward_optimize` before trusting `optimize` results** —
  in-sample Sharpe is almost always too high. See
  [Walk-forward optimization](walk_forward.md).
