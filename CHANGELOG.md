# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/) — though as a
0.x library breaking changes can land in any minor bump.

## [Unreleased]

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

[Unreleased]: https://github.com/gguan/qtrade/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/gguan/qtrade/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/gguan/qtrade/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/gguan/qtrade/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gguan/qtrade/compare/v0.1.3...v0.2.0
