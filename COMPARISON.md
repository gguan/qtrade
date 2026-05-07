# How QTrade compares

There are good Python backtesting libraries already. This page is an
honest take on where QTrade fits — including where it doesn't.

## TL;DR

QTrade sits between **Backtesting.py** (very simple, single-asset only)
and **backtrader** (very featureful, decade-old, complex).

Pick QTrade when you want:
- multi-asset / portfolio backtests with one shared cash pool
- a Gymnasium `TradingEnv` for RL on the same data, no glue code
- walk-forward analysis built in, not bolted on
- a small, type-checked codebase you can read top to bottom in an hour

Pick something else when you need:
- **maximum throughput** for huge parameter sweeps → vectorbt
- **mature broker / live-trading integrations** → backtrader
- **institutional-style infrastructure** (data pipeline, risk model,
  pipeline factor research) → zipline-reloaded

## Feature matrix

| | QTrade 0.4 | [Backtesting.py](https://github.com/kernc/backtesting.py) | [backtrader](https://github.com/mementum/backtrader) | [vectorbt](https://github.com/polakowo/vectorbt) | [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |
|---|---|---|---|---|---|
| Single-asset backtest | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multi-asset / portfolio | ✅ shared cash | ⚠️ workaround | ✅ | ✅ | ✅ |
| Walk-forward analysis | ✅ built in | ❌ | manual | ✅ via splitter | manual |
| Stop-loss / take-profit | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Limit / stop orders | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Margin / leverage | ✅ | ⚠️ basic | ✅ | ❌ | ⚠️ |
| Built-in RL env | ✅ Gymnasium | ❌ | ❌ | ❌ | ❌ |
| Bokeh interactive plot | ✅ | ✅ | matplotlib | matplotlib/plotly | matplotlib |
| Vectorized speed | ❌ event loop | ❌ event loop | ❌ event loop | ✅ Numba | ❌ event loop |
| Live trading adapters | ❌ | ❌ | ✅ many | ❌ | ❌ |
| Type hints / mypy clean | ✅ | ⚠️ partial | ❌ | partial | ⚠️ |
| Lines of source | ~1.3k | ~1.6k | ~50k | ~50k | ~30k |

Legend: ✅ first-class · ⚠️ possible but awkward · ❌ not supported

> The "lines of source" row is approximate (rounded). It's a proxy for
> "how much you'd have to read to understand it," not a quality
> judgment.

## Honest answers to specific questions

### "Can I do parameter sweeps with 1000s of param combos?"

Use **vectorbt**. It vectorizes through Numba and beats every event-loop
library by 100×–1000× for grid searches.

QTrade's `Backtest.optimize` and `walk_forward_optimize` are sequential
event-loop runs — fine for dozens of params, painful for thousands.

### "I want to live-trade through Interactive Brokers"

Use **backtrader**. It has battle-tested live broker integrations
(IBKR, Oanda, etc.) that QTrade doesn't try to compete with.

### "I'm running a quant fund / formal factor research"

Use **zipline-reloaded**. The pipeline / factor / risk-model
abstractions are designed for this.

### "I want to learn RL on financial data"

QTrade's `TradingEnv` is a normal Gymnasium env — drop it into
`stable-baselines3`, write a custom `ActionScheme` / `RewardScheme` /
`ObserverScheme`, and go. There's no equivalent in the others; people
typically wire one up themselves on top of backtrader.

[tensortrade](https://github.com/tensortrade-org/tensortrade) covers
this niche with a much larger surface area; we cover the basics.

### "I want to backtest a pairs-trading or rotation strategy"

This is one of QTrade's sweet spots. `Backtest({"AAPL": df1, "MSFT": df2}, ...)`
gives you per-asset positions sharing one cash pool, with portfolio-level
equity / drawdown / Sharpe. In Backtesting.py you'd be reaching for
workarounds; in backtrader/vectorbt you'd be configuring a fair amount
of machinery.

### "How fast is QTrade?"

Reference numbers ([tests/benchmarks/README.md](tests/benchmarks/README.md))
on M-class macOS, qtrade 0.4.0:

- Pure broker, 10k bars: ~380 ms (~26k bars/s)
- SMA strategy backtest, 10k bars: ~1.4 s (~7k bars/s)
- 4-asset portfolio backtest, 5k bars: ~3.5 s (~1.4k bars/s/asset)

Plenty fast for typical workflows (5–20 years of daily data is 1k–5k
bars). Slow once you cross 100k+ bars or a several-thousand-cell
parameter grid — that's vectorbt's territory.

### "How small / readable is the codebase?"

Pretty small. The whole library is 8 modules (`Broker`, `Order`,
`Trade`, `Position`, `Commission`, `Strategy`, `Backtest`, `TradingEnv`)
plus stats and Bokeh plotting. You can read the entire core in an hour
and modify it without surprises. mypy + ruff are wired into CI; tests
hit ~88% line coverage on the library code.

That said — the corollary is "small surface area, not all features
included." If something feels missing, we may genuinely not have it
yet, vs. it being hidden in a deep config tree.
