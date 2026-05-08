# HTML reports & optimization heatmaps

Two visual outputs, both designed to be shareable as a single file you can
email, attach to a Slack message, or commit to a results folder.

## Single-file backtest report

```python
bt = Backtest(data, MyStrategy, cash=100_000)
bt.run()
bt.export_report("results/q1_run.html")
```

The HTML file bundles:

1. **Header** — strategy class, asset list, date range, starting cash,
   generation timestamp.
2. **Performance table** — every metric `calculate_stats` returns
   (Sharpe, Sortino, Max Drawdown, Profit Factor, etc.).
3. **Charts** — the same Bokeh layout `bt.plot()` produces (equity vs
   Buy & Hold on top, trade-returns scatter, OHLC with entries/exits
   underneath; one OHLC panel per asset for multi-asset).
4. **Per-asset stats** — multi-asset only.
5. **Trade analytics** — hold-duration distribution and winners-vs-losers
   feature comparison from `qtrade.analytics`.
6. **Trade history** — first 200 rows, with `Profit` colored green/red.

Bokeh JS is loaded from CDN so the HTML stays small (~50 KB regardless of
how many trades). No internet needed to *generate* the file — only to
*view* it.

Custom title / strategy label:

```python
bt.export_report(
    "results/q1_run.html",
    title="Q1 2024 Mean-Reversion Backtest",
    strategy_name="MeanRev v3.2",
)
```

## Optimization heatmap

After a 2D grid search, see *where* the parameter landscape is flat vs
sharp — and whether the "best" point sits on a plateau (robust) or a
spike (likely overfit):

```python
best_params, best_stats, results = bt.optimize(
    maximize="Sharpe Ratio",
    n1=range(5, 30, 5),
    n2=range(20, 80, 10),
)

bt.plot_heatmap(
    results,
    x="n1",
    y="n2",
    metric="Sharpe Ratio",
    filename="results/sharpe_heatmap.html",
)
```

What you get: a Bokeh figure with categorical axes (parameter values as
strings, so 5 / 10 / 15 line up evenly even when the grid isn't uniform),
a red→yellow→green color mapping (high = green, suitable for most
metrics — flip the palette for "lower is better" metrics like Max
Drawdown), and a hover tooltip that shows the exact value per cell.

### Reading the heatmap

- **Solid green region** around the best point = robust. Small parameter
  drift won't crater performance — your edge isn't living on a knife.
- **Lone bright cell** surrounded by red = almost certainly overfit. The
  exact (n1, n2) combination that won probably won't generalize.
- **Diagonal stripes** often indicate the metric depends on `n2 − n1` (or
  some ratio) rather than each individually — consider re-parameterizing.

### Different metrics

Color by anything in the `stats` dict:

```python
bt.plot_heatmap(results, x="n1", y="n2", metric="Total Return [%]")
bt.plot_heatmap(results, x="n1", y="n2", metric="Max Drawdown [%]",
                palette=tuple(reversed(__import__('bokeh.palettes', fromlist=['RdYlGn11']).RdYlGn11)))
```

For drawdown / loss-style metrics where lower is better, reverse the
palette so red marks the bad cells. Any Bokeh palette works.

### 3+ parameter grids

The heatmap is 2D. If you optimize over three parameters, choose two for
the axes and let `aggfunc` (default `"mean"`) marginalize over the third:

```python
results = bt.optimize(
    maximize="Sharpe Ratio",
    n1=range(5, 30, 5),
    n2=range(20, 80, 10),
    threshold=[0.01, 0.02, 0.03],
)
bt.plot_heatmap(
    results, x="n1", y="n2", metric="Sharpe Ratio", aggfunc="mean",
)
```

Or use `aggfunc="max"` to see "the best Sharpe achievable at each (n1, n2)
across any threshold" — a softer overfitting check.

### Lower-level access

Both helpers live in `qtrade.utils` if you want to compose them yourself
(e.g. embed the heatmap in a custom report):

```python
from qtrade.utils.heatmap import plot_optimization_heatmap, results_to_dataframe
from qtrade.utils.report import build_html_report
```
