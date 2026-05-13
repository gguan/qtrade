# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/) — though as a
0.x library breaking changes can land in any minor bump.

## [Unreleased]

## [0.5.2]

Mid-trade SL/TP modification and trailing stops — long-requested for any
strategy that wants to ratchet stops as a position moves into profit.

### Added
- **`Trade.sl` / `Trade.tp` are now settable.** You can mutate the level on
  an open trade and it takes effect on the next bar's SL/TP check.
- **`Trade.update_exit_levels(sl=..., tp=...)`** — convenience method that
  uses a sentinel to distinguish "leave alone" from "clear":
  ```python
  trade.update_exit_levels(sl=110)        # bump SL only, TP unchanged
  trade.update_exit_levels(tp=None)       # explicitly clear TP
  ```
- **Trailing-stop orders** via `trail_percent=` (fraction) or
  `trail_amount=` (absolute) on `Strategy.buy()` / `Strategy.sell()`:
  ```python
  self.buy(size=10, trail_percent=0.05)   # 5% trailing stop, ratchets up
  self.sell(size=10, trail_amount=2.0)    # $2 trailing stop on a short
  ```
  The Broker auto-bumps the SL each bar based on the per-asset high-water
  mark (long) or low-water mark (short). Coexists with explicit `sl=` —
  whichever is tighter (better for the trader) wins. Trail metadata
  (`trail_percent`, `trail_amount`, `trail_high`, `trail_low`) is preserved
  on the closed-trade record for post-run analysis.

### Changed
- `Trade.__init__` and `Order.__init__` now have `trail_percent` /
  `trail_amount` as **keyword-only** parameters (after a `*`). Existing
  positional calls keep working.
- `Broker.process_bar` calls `__update_trailing_stops` after `__check_sl_tp`
  so the tightened stop takes effect on the *next* bar — conservative
  bar-level semantics consistent with the rest of the engine.

## [0.5.1]

Additive release — new data adapters for the Chinese market, post-run
analytics, an optimization heatmap, and a self-contained HTML backtest
report. No behavior changes for existing code.

### Added
- **AKShare adapter** for Chinese A-shares and futures (optional `[cn]`
  extra). New helpers in `qtrade.data`:
  - `from_akshare_stock_a(symbols, start, end, period="daily", adjust="qfq")`
    — A-shares (Shanghai/Shenzhen) by 6-digit code. Translates
    `日期/开盘/最高/最低/收盘/成交量` to standard `Open/High/Low/Close/Volume`.
  - `from_akshare_futures(symbols, start, end)` — main-contract continuous
    series for SHFE / DCE / CZCE / CFFEX. Pair with custom `Contract`
    specs for the right multiplier and margin.
  - Both accept either `"YYYY-MM-DD"` or `"YYYYMMDD"` date strings,
    a single symbol or a list, and align indexes on intersection.
- **`qtrade.analytics` module** — trade-level analytics that go past the
  aggregate stats:
  - `hold_duration_distribution(broker)` — describe-style summary of trade
    hold times in hours, optionally split by winners / losers.
  - `entries_by_weekday(broker, metric=...)` and
    `entries_by_month(broker, metric=...)` — calendar-bias detection.
    Metrics: `count`, `profit_sum`, `profit_mean`, `win_rate`. Always
    indexed in calendar order.
  - `win_loss_feature_comparison(broker)` — DataFrame comparing mean /
    median Size, Entry Price, Exit Price, hold duration of winning vs
    losing trades. `extra_features=` for custom columns from the trade
    history.
- **`Backtest.plot_heatmap(results, x=, y=, metric=)`** — render an
  `optimize()` grid as a 2D Bokeh heatmap. Categorical axes, hover
  tooltips, RdYlGn palette by default. Lets you check whether the "best"
  point sits on a robust plateau or a lone spike (likely overfit). 3+
  parameter grids marginalize via `aggfunc=`. Lower-level entry point:
  `qtrade.utils.heatmap.plot_optimization_heatmap`.
- **`Backtest.export_report(path)`** — single self-contained HTML file
  bundling stats + Bokeh charts + per-asset breakdown + trade analytics
  + trade history. CDN-hosted Bokeh JS, so the file stays small (~50 KB)
  regardless of trade count. Lower-level entry point:
  `qtrade.utils.report.build_html_report`.
- New optional `[cn]` extra in `pyproject.toml` (pulls in `akshare>=1.13`).
- Three new docs guide pages: `docs/guide/data_sources.md`,
  `docs/guide/analytics.md`, `docs/guide/reports.md`.

### Changed
- `qtrade.utils.plot_bokeh` refactored: layout-builder functions
  `_plot_single_asset_layout` / `_plot_multi_asset_layout` are now
  separate from the show/save side effects, so the report module can
  embed the chart layout via `bokeh.embed.components`. No public-API
  change.

## [0.5.0]

The big v0.5 theme: **futures support**. Run portfolios that mix
stocks (no leverage) with multi-category futures (each with its own
contract multiplier and margin requirement) in one backtest.

### Added
- **High-level `contracts=` API** on `Backtest`. Pass a single
  `Contract` (applies uniformly) or a dict keyed by asset symbol
  (per-asset; missing keys fall back to `STOCK_CASH` so pure-stock
  backtests need zero configuration). Example:
  ```python
  from qtrade.contracts import GC_COMEX, ES_CME
  bt = Backtest(data, MyStrat, cash=100_000, contracts={
      "GC=F": GC_COMEX,   # 1 contract = $100/$1 move, 5% margin
      "ES=F": ES_CME,
      # AAPL omitted → STOCK_CASH (multiplier 1, no leverage)
  })
  ```
- New `qtrade.contracts` module with `Contract` dataclass and a
  registry of built-in specs: `STOCK_CASH`, `STOCK_REGT`, CME equity
  (`ES_CME`, `NQ_CME`, `MES_CME`, `MNQ_CME`, …), COMEX metals
  (`GC_COMEX`, `MGC_COMEX`, `SI_COMEX`, `HG_COMEX`), NYMEX energy
  (`CL_NYMEX`, `NG_NYMEX`, …), and CBOT agriculture. `Contract` is
  frozen — define your own for non-listed instruments.
- New `qtrade.data` module (optional `[data]` extra):
  - `from_yfinance(symbols, start=, end=, ...)` — multi-ticker download
    that returns an index-aligned `dict[str, DataFrame]` ready for
    `Backtest`. One line replaces ~30 lines of yfinance boilerplate.
  - `align_indexes(data)` — utility for any other dict-of-DataFrames.
- **Contract multiplier support** for futures-style instruments. New
  `contract_multiplier` parameter on `Backtest` / `Broker` accepts a
  scalar (single asset) or a `dict[str, float]` (per-asset, e.g.
  `{"AAPL": 1, "GC": 100, "ES": 50, "CL": 1000}`). PnL and margin scale
  by `size × multiplier × price`, so 1 GC contract moving $1 → $100 P&L
  matches reality. Backwards compatible — default is 1.0.
- **Per-asset `margin_ratio`**. Same parameter now accepts a dict so
  mixed portfolios can put `1.0` on stocks and `0.05` on futures.
- New `Broker.multiplier_by_asset` and `Broker.margin_ratio_by_asset`
  properties for inspecting the resolved per-asset settings.
- `examples/mixed_portfolio.py` — realistic stock + multi-category
  futures portfolio (AAPL + ES + GC + CL) with different multipliers
  and margins per asset.

### Fixed
- `equity_history` is now NaN for un-processed future bars instead of
  pre-filled with the starting cash. Reading mid-backtest no longer
  shows a misleading flat-cash line.
- Pure stop orders (`Order(stop=...)` with no limit) now fill at the
  trigger price (or the bar's open if the market gapped past the stop)
  instead of the bar's close. SL/TP attached to a Trade was already
  correct.
- `contracts=` dict now validates that every key matches a real asset.
  Previously a typo (`"GC"` instead of `"GC=F"`) was silently dropped
  and the asset fell back to `STOCK_CASH`, producing a wrong-but-
  plausible backtest with no warning.

## [0.4.1]

### Added
- Automated PyPI releases via GitHub Actions trusted publishing
  ([.github/workflows/release.yml](.github/workflows/release.yml)). Push
  a `v*` tag and the workflow runs the test suite, builds, asserts the
  tag matches `pyproject.toml`, publishes to PyPI, and creates a GitHub
  Release. See `RELEASING.md` for one-time setup steps.
- `tests/benchmarks/` — opt-in performance benchmarks via `pytest-benchmark`.
  Excluded from the default `pytest` run; invoke with
  `pytest tests/benchmarks/`. Baseline numbers and "where to optimize first
  if it ever matters" notes live in `tests/benchmarks/README.md`.
- Regression test for the recommended `prepare()` indicator pattern
  (`for df in self._data.values(): ...`).

### Fixed
- `examples/simple_strategy.py` was silently broken since v0.3.0:
  `self._data['Close']` no longer works because `_data` is a
  `dict[str, DataFrame]`. Updated to iterate `self._data.values()`.
- Same fix in the `getting_started.md` SMA strategy example.

## [0.4.0]

### Added
- `Backtest.walk_forward_optimize(train_window=, test_window=, step=, maximize=, ...)` —
  rolling-window training + immediate out-of-sample test for each window.
  Returns per-window train/test stats plus an aggregate summary
  (mean OoS return, hit rate, min/max). Mitigates the in-sample overfitting
  bias of `Backtest.optimize`.
- `Broker.get_trade_history()` — single source of truth for the
  trade-by-trade DataFrame.
- `examples/portfolio_strategy.py` — multi-asset mean-reversion example.
- `CHANGELOG.md` (this file).

### Changed
- `Backtest.get_trade_history()` and `TradingEnv.get_trade_history()` now
  delegate to `Broker.get_trade_history()` (no more copy-paste).

## [0.3.0]

### Added
- **Multi-asset / portfolio support.** `Backtest` accepts
  `dict[str, pd.DataFrame]` keyed by asset symbol; one Position per asset
  with a shared cash pool.
- New on `Strategy`: `assets`, `positions` (dict), `data_by_asset` (dict).
- `Strategy.buy(asset, ...)`, `sell(asset, ...)`, `close(asset)` — first
  positional is the asset symbol; required when running on >1 asset,
  optional otherwise.
- New `Trade.asset` and `Order.asset` fields.
- `calculate_stats_per_asset(broker)` — per-asset trade-level breakdown.
- `plot_with_bokeh(broker)` renders one OHLC panel per asset stacked
  vertically when given a multi-asset broker; portfolio equity vs
  equal-weighted Buy & Hold up top.
- New docs page: User Guide → Multi-asset / portfolio backtests.

### Fixed
- `calculate_stats` no longer touches `broker.data` directly (which only
  exists in single-asset mode); uses `broker.equity_history` for index
  metrics.
- `Buy & Hold Return` for multi-asset is now the equal-weighted average
  across assets.

## [0.2.0]

### Added
- Type checking with mypy in CI.
- Pre-commit hooks (ruff + mypy).
- `examples/rl_example.py`.
- Optional `[rl]` extras with `stable-baselines3`.

### Changed
- Build system migrated from `setuptools` (setup.py / setup.cfg /
  requirements.txt) to PEP 621 `pyproject.toml` + `hatchling`.
- Dependency floors raised: `numpy>=2.0`, `pandas>=2.2`, `scipy>=1.13`,
  `matplotlib>=3.9`, `bokeh>=3.9`, `tqdm>=4.66`, `gymnasium>=1.0`.
- Python floor raised: `>=3.10`.
- CI lints with `ruff` (replaces flake8); ruff target updated to `py310`.
- Docs build now fails on warnings (`-W --keep-going`).

### Fixed
- `Strategy.{active,closed}_trades` raised `TypeError` because it called
  `position.{active,closed}_trades` (which is a `@property`) with `()`.
- SL/TP synthetic `Order` recorded the closing trade with the wrong sign
  (closing a long was logged as a buy in `filled_orders`, making
  `plot_bokeh` draw it as a green up-arrow).
- Stray debug `print()` in `Strategy.buy()` when default size resolved to 0.
- `plot_bokeh` HoverTool and JS callback used lowercase OHLCV column names
  but the data uses capitalized; tooltips appeared empty and any zoom
  triggered console errors.
- `__remove_closed_orders` now drains `_executing_orders` too — cancelling
  a market order under `trade_on_close=False` no longer crashes the next
  bar.
- `Strategy.buy()/sell()` take `abs()` on explicit size so `sell(size=-5)`
  no longer silently flips into a buy.
- `TradingEnv` raises a clear `ValueError` when `random_start=True` with
  data shorter than `window_size + max_steps`.
- Removed dead `qtrade/utils/plotly.py` (orphaned + referenced
  non-existent broker attributes).
- Ruff and mypy debt cleaned up in `core/`, `backtest/`, `env/` modules.

## [0.1.0] / [0.1.3]

Initial development versions: single-asset Broker / Strategy / Backtest,
basic Gymnasium TradingEnv, Bokeh plotting, stats calculation. See git
history for details.

[Unreleased]: https://github.com/gguan/qtrade/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/gguan/qtrade/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/gguan/qtrade/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/gguan/qtrade/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/gguan/qtrade/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/gguan/qtrade/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/gguan/qtrade/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gguan/qtrade/compare/v0.1.3...v0.2.0
