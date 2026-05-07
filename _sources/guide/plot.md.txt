# Reading the Bokeh report

`Backtest.plot()` (and `TradingEnv.plot()`) renders an interactive
Bokeh report combining equity, trade returns, and per-asset OHLC
panels. This page explains what each panel shows and how to interpret
it.

## Saving to file

By default `plot()` calls `bokeh.io.show()` which opens the report in
a browser. To save a self-contained HTML for sharing or archiving:

```python
from qtrade.utils.plot_bokeh import plot_with_bokeh
plot_with_bokeh(bt.broker, filename="report.html")
```

The HTML has all data inlined — no Bokeh server, no external assets.
Email it, commit it, drop it in a wiki.

## Single-asset layout

Three stacked panels with linked x-axes:

```
┌─────────────────────────────────────────────────┐
│ ▒▓▒  Equity vs. Buy & Hold (cumulative returns) │
│                                                 │
├─────────────────────────────────────────────────┤
│ ▲ ▼ ▲       Trade returns (long / short)        │
├─────────────────────────────────────────────────┤
│       OHLC + volume + win/lose trade boxes      │
│       + buy/sell markers                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Top: Equity panel

- **Solid blue line**: portfolio equity (cumulative returns, normalized to
  1.0 at the start).
- **Gray dotted line**: Buy & Hold returns on the same scale.
- **Purple dot**: peak equity point.
- **Blue dot**: final equity point (legend shows the final pct).
- **Gray dot**: final B&H point.
- **Red dot**: max drawdown trough.
- **Red shaded area**: the drawdown region (gap between equity and its
  running max).
- **Red horizontal segment**: the longest drawdown duration.

What to look for:

- Is the blue line above the gray? You're beating Buy & Hold.
- How "smooth" is the blue line vs gray? Smoother = lower volatility =
  better risk-adjusted return (the Sharpe story).
- Is the red shaded area concentrated in one period or spread out?
  Concentrated = strategy has one weak regime; spread out = systemic
  drag.

### Middle: Trade returns

A scatter of every closed trade plotted at its exit time:

- **Green up-triangles**: long trades. y-axis is `(exit/entry − 1)`.
- **Red down-triangles**: short trades. y-axis is the equivalent for shorts.
- **Marker size**: scaled to position size (bigger = larger trade).
- **Dashed horizontal at 0**: break-even line.

What to look for:

- Are most markers above the 0-line? That's your win rate eyeballed.
- Is there a cluster of large-magnitude losses (way below 0)? That's
  where stop-losses (or lack thereof) hurt.
- Are big winners and big losers symmetric around 0? Asymmetric is
  good — small losses + large gains is the trend-following sweet spot.

### Bottom: OHLC + volume + trade boxes

The price action with overlays:

- **Black line**: closing prices.
- **Bottom bars (faint green/red)**: volume (right y-axis), color-coded
  by close ≥/< open.
- **Translucent green rectangles**: winning trades — left/right edges
  are entry/exit times, top/bottom are entry/exit prices.
- **Translucent red rectangles**: losing trades, same convention.
- **Triangle markers**: order fills. Up = buy; down = sell. Size scaled
  to order magnitude.

What to look for:

- Trade boxes that span long horizontal stretches = you held through
  noise. Short stretches = quick in-out trades.
- Buy markers at local highs / sell markers at local lows = bad
  timing, possibly a sign of late signal generation.
- Boxes piling up densely = high turnover. Watch the
  `Total Commission Cost` metric in `show_stats()`.

## Multi-asset layout

The report adds one OHLC panel per asset, all stacked, all sharing the
same x-axis:

```
┌─────────────────────────────────────────────────┐
│  Portfolio equity vs equal-weighted Buy & Hold  │
├─────────────────────────────────────────────────┤
│  Trade returns scatter (all assets combined)    │
├─────────────────────────────────────────────────┤
│  OHLC — AAPL                                    │
├─────────────────────────────────────────────────┤
│  OHLC — MSFT                                    │
├─────────────────────────────────────────────────┤
│  OHLC — NVDA                                    │
└─────────────────────────────────────────────────┘
```

Each per-asset OHLC panel shows trades and orders **filtered to that
asset only**, so you can see exactly when the strategy moved in each
name. The equity panel up top is portfolio-level (one curve, summing
across all assets).

The Buy & Hold benchmark is the **equal-weighted average** of each
asset's first-to-last return — a fair "naive portfolio" comparison.

## Interactive features

- **Drag the x-axis** on any panel to pan; **scroll** to zoom.
- **Click a legend entry** to toggle that series on/off.
- **Hover** for tooltips: dates, prices, returns, profit per trade.
- **Auto-rescaling**: zooming the x-axis triggers a JS callback that
  rescales the y-axes for the visible range only (single-asset layout).

## Common surprises

- **Trade-return scatter looks empty**: the strategy hasn't closed any
  trades yet. SL/TP not triggered + no signal exits = no closed trades.
- **OHLC shows a gap at the end with no buy/sell**: the
  `close_all_positions()` end-of-run sweep records exits at the last
  bar's close, which can produce a flurry of markers right at the
  right edge.
- **Volume is missing**: your input data didn't have a `Volume` column.
  Add one (or accept volume bars don't show).
- **Equity line drops sharply with no visible trade**: you took a
  large mark-to-market loss without exiting. Check `unrealized_pnl`
  in your strategy logic.
