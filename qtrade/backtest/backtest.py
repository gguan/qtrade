import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from qtrade.backtest.strategy import Strategy
from qtrade.contracts import STOCK_CASH, Contract
from qtrade.core import Broker, Commission, Order
from qtrade.utils import calculate_stats, plot_with_bokeh


class Backtest:
    """Run a strategy against historical data — single asset or a portfolio.

    Pass a ``DataFrame`` for single-asset; pass ``dict[str, DataFrame]`` to
    run a portfolio strategy across multiple assets.
    """

    def __init__(self,
                 data: pd.DataFrame | dict[str, pd.DataFrame],
                 strategy_class: type[Strategy],
                 cash: float = 10_000,
                 commission: Commission | None = None,
                 margin_ratio: float | dict[str, float] | None = None,
                 trade_on_close: bool = False,
                 verbose: bool = False,
                 contract_multiplier: float | dict[str, float] | None = None,
                 contracts: dict[str, Contract] | Contract | None = None,
                 ):
        """
        Args:
            data: Single OHLCV DataFrame, or a dict mapping asset symbol to
                its OHLCV DataFrame (multi-asset). All DataFrames must have a
                DatetimeIndex and the same index for proper portfolio accounting.
            strategy_class: Strategy class (subclass of Strategy) to instantiate.
            cash: Starting cash.
            commission: Commission calculator (None ⇒ no commission).
            contracts: **Preferred** way to specify per-asset multiplier and
                margin. Pass a single :class:`~qtrade.contracts.Contract` to
                apply to all assets, or a dict keyed by asset symbol — assets
                you don't list resolve to :data:`~qtrade.contracts.STOCK_CASH`
                (no leverage, multiplier 1.0). Built-in specs live in
                :mod:`qtrade.contracts` (``STOCK_CASH``, ``GC_COMEX``,
                ``ES_CME``, etc.); custom ``Contract(multiplier=…, margin_ratio=…)``
                instances work the same way.
            margin_ratio: Lower-level escape hatch (scalar or dict).
                Mutually exclusive with ``contracts``.
            contract_multiplier: Lower-level escape hatch (scalar or dict).
                Mutually exclusive with ``contracts``.
            trade_on_close: If True, market orders fill at the current bar's
                close price; otherwise at the next bar's open.
            verbose: Verbose logging.
        """
        self._is_multi_asset = not isinstance(data, pd.DataFrame)

        if self._is_multi_asset:
            assert len(data) > 0, "Multi-asset data dict cannot be empty."
            data_by_asset = {asset: self._validate_and_sort(df, asset) for asset, df in data.items()}
            # Cross-asset alignment: enforce identical indexes for portfolio accounting.
            first_idx = next(iter(data_by_asset.values())).index
            for asset, df in data_by_asset.items():
                if not df.index.equals(first_idx):
                    raise ValueError(
                        f"Asset '{asset}' has a different index than the first asset; "
                        "all assets must share the same DatetimeIndex (align/reindex before passing in)."
                    )
            self.data: pd.DataFrame | dict[str, pd.DataFrame] = data_by_asset
        else:
            self.data = self._validate_and_sort(data)

        # Resolve the contracts API into the lower-level (margin_ratio,
        # contract_multiplier) pair the Broker consumes. The two APIs are
        # mutually exclusive — pass `contracts` OR the explicit dicts/scalars,
        # not both.
        margin_ratio, contract_multiplier = self._resolve_contracts(
            self.data, contracts, margin_ratio, contract_multiplier,
        )

        self.broker = Broker(
            self.data, cash, commission, margin_ratio, trade_on_close,
            contract_multiplier=contract_multiplier,
        )
        self.strategy_class = strategy_class
        self.current_bar = 0
        self.cash = cash
        self.commission = commission
        self.margin_ratio = margin_ratio
        self.trade_on_close = trade_on_close
        self.contract_multiplier = contract_multiplier
        self.contracts = contracts

        self.order_history: list[Order] = []
        self.stats = None

        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _resolve_contracts(data, contracts, margin_ratio, contract_multiplier):
        """Map the user-facing ``contracts`` API onto Broker's (margin_ratio,
        contract_multiplier) pair. Returns ``(margin_ratio, contract_multiplier)``.

        Rules:
        - If ``contracts`` is None: use the explicit args verbatim (or 1.0 default).
        - If ``contracts`` is a Contract: applies uniformly to every asset.
        - If ``contracts`` is a dict: per-asset; missing keys fall back to STOCK_CASH.
        - Mixing ``contracts`` with ``margin_ratio`` / ``contract_multiplier`` raises.
        """
        if contracts is None:
            # Legacy / lower-level path. Default scalar margin to 1.0 for stocks.
            return (margin_ratio if margin_ratio is not None else 1.0,
                    contract_multiplier)

        if margin_ratio is not None or contract_multiplier is not None:
            raise ValueError(
                "Pass either `contracts=` OR (`margin_ratio=` / `contract_multiplier=`), "
                "not both."
            )

        # Determine the asset set from data (for both single- and multi-asset).
        if isinstance(data, dict):
            assets = list(data.keys())
        else:
            assets = ["default"]

        if isinstance(contracts, Contract):
            resolved = {a: contracts for a in assets}
        else:
            # Per-asset dict: missing keys fall back to STOCK_CASH so that pure
            # stock backtests need zero per-asset configuration.
            resolved = {a: contracts.get(a, STOCK_CASH) for a in assets}

        return (
            {a: c.margin_ratio for a, c in resolved.items()},
            {a: c.multiplier for a, c in resolved.items()},
        )

    @staticmethod
    def _validate_and_sort(df: pd.DataFrame, asset: str | None = None) -> pd.DataFrame:
        prefix = f"Asset '{asset}': " if asset else ""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{prefix}Data index must be a DatetimeIndex")
        if {'open', 'high', 'low', 'close'} - {col.lower() for col in df.columns}:
            raise ValueError(f"{prefix}Data must contain columns: 'open', 'high', 'low', 'close'")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df.copy(deep=False)

    def _index(self) -> pd.DatetimeIndex:
        """Common datetime index across assets (validated equal in __init__)."""
        if self._is_multi_asset:
            return next(iter(self.data.values())).index  # type: ignore[union-attr]
        return self.data.index  # type: ignore[union-attr]

    def _start_idx(self) -> int:
        """First bar index where every asset has all OHLC columns non-NaN.

        Skips warm-up periods of indicators that the strategy populated in
        prepare(). Multi-asset: take the latest such bar across all assets.
        """
        starts: list[int] = []
        for df in self.strategy._data.values():
            starts.append(1 + int(np.argmax(df.notna().all(axis=1).values)))
        return max(starts)

    def run(self, **strategy_params):
        """Run the backtest end-to-end."""
        self.strategy = self.strategy_class(self.broker, self.data, strategy_params)
        self.strategy.prepare()

        index = self._index()
        start = self._start_idx()
        for i in tqdm(range(start, len(index)), desc="Running Backtest"):
            self.current_bar = i
            current_time = index[i]
            self.broker.process_bar(current_time)
            self.strategy.on_bar_close()

        self.broker.close_all_positions()

    def optimize(self,
                 maximize: str,
                 constraint: Callable[[Any], bool] | None = None,
                 **params_grid):
        """Grid-search strategy parameters and return the best.

        Args:
            maximize: Metric name from `calculate_stats` to maximize.
            constraint: Optional filter on parameter dicts (return False to skip).
            **params_grid: Parameter ranges, e.g. ``n1=range(5, 30, 5)``.

        Returns:
            ``(best_params, best_stats, all_results)``.
        """
        from itertools import product

        best_params = None
        best_stats = None
        all_results = []

        keys = list(params_grid.keys())
        for combination in product(*params_grid.values()):
            param_dict = dict(zip(keys, combination, strict=True))

            if constraint and not constraint(param_dict):
                continue

            # Fresh broker for each parameter combination.
            self.broker = Broker(
                self.data, self.cash, self.commission, self.margin_ratio, self.trade_on_close,
                contract_multiplier=self.contract_multiplier,
            )
            self.run(**param_dict)
            stats = calculate_stats(self.broker)

            all_results.append({'params': param_dict, 'stats': stats})

            if best_stats is None or stats[maximize] > best_stats[maximize]:
                best_stats = stats
                best_params = param_dict

        return best_params, best_stats, all_results

    def walk_forward_optimize(self,
                              *,
                              train_window: int,
                              test_window: int,
                              maximize: str,
                              step: int | None = None,
                              constraint: Callable[[Any], bool] | None = None,
                              **params_grid) -> dict:
        """Walk-forward optimization: rolling-window training + immediate
        out-of-sample test.

        For each (train, test) window pair:

        1. Run :meth:`optimize` on the train slice to pick the best params.
        2. Run a fresh backtest on the test slice using those params.
        3. Record the out-of-sample (OoS) statistics.

        This is the standard antidote to the in-sample overfitting that bare
        :meth:`optimize` produces — every reported metric is on data the
        strategy was never tuned on.

        Args:
            train_window: Number of bars in each training window.
            test_window: Number of bars in the test window immediately after.
            maximize: Metric name to maximize within each train window
                (same as :meth:`optimize`).
            step: How many bars to advance between successive train starts.
                Defaults to ``test_window`` (non-overlapping test windows).
            constraint: Optional filter on parameter dicts.
            **params_grid: Parameter ranges, same as :meth:`optimize`.

        Returns:
            ``{'windows': [...], 'summary': {...}}``.

            Each entry in ``windows`` is a dict with ``train_start``,
            ``train_end``, ``test_start``, ``test_end``, ``best_params``,
            ``train_stats``, ``test_stats``, and ``test_equity``
            (the test-window equity Series).

            ``summary`` contains aggregate OoS metrics: ``n_windows``,
            ``mean_oos_return``, ``hit_rate``, ``min_oos_return``,
            ``max_oos_return``.
        """
        if train_window <= 0 or test_window <= 0:
            raise ValueError("train_window and test_window must be positive")

        index = self._index()
        n = len(index)
        if train_window + test_window > n:
            raise ValueError(
                f"train_window ({train_window}) + test_window ({test_window}) "
                f"exceeds data length ({n})"
            )

        if step is None:
            step = test_window

        windows: list[dict] = []
        i = 0
        while i + train_window + test_window <= n:
            train_slc = slice(i, i + train_window)
            test_slc = slice(i + train_window, i + train_window + test_window)

            # Train: optimize on the train slice
            train_bt = Backtest(
                self._slice_data(train_slc),
                self.strategy_class,
                cash=self.cash,
                commission=self.commission,
                margin_ratio=self.margin_ratio,
                trade_on_close=self.trade_on_close,
                contract_multiplier=self.contract_multiplier,
            )  # contracts already resolved to margin_ratio/multiplier above
            best_params, train_stats, _ = train_bt.optimize(
                maximize=maximize, constraint=constraint, **params_grid,
            )

            # Test: fresh backtest with the best params, on data the strategy
            # has never seen.
            test_bt = Backtest(
                self._slice_data(test_slc),
                self.strategy_class,
                cash=self.cash,
                commission=self.commission,
                margin_ratio=self.margin_ratio,
                trade_on_close=self.trade_on_close,
                contract_multiplier=self.contract_multiplier,
            )  # contracts already resolved to margin_ratio/multiplier above
            test_bt.run(**(best_params or {}))
            test_stats = calculate_stats(test_bt.broker)

            windows.append({
                'train_start': index[i],
                'train_end': index[i + train_window - 1],
                'test_start': index[i + train_window],
                'test_end': index[i + train_window + test_window - 1],
                'best_params': best_params,
                'train_stats': train_stats,
                'test_stats': test_stats,
                'test_equity': test_bt.broker.equity_history.copy(),
            })
            i += step

        oos_returns = [w['test_stats']['Total Return [%]'] for w in windows]
        summary = {
            'n_windows': len(windows),
            'mean_oos_return': float(np.mean(oos_returns)) if oos_returns else 0.0,
            'hit_rate': float(np.mean([r > 0 for r in oos_returns])) if oos_returns else 0.0,
            'min_oos_return': float(np.min(oos_returns)) if oos_returns else 0.0,
            'max_oos_return': float(np.max(oos_returns)) if oos_returns else 0.0,
        }

        return {'windows': windows, 'summary': summary}

    def _slice_data(self, slc: slice):
        """Slice the input data by row index. Handles single and multi-asset."""
        if self._is_multi_asset:
            return {asset: df.iloc[slc] for asset, df in self.data.items()}  # type: ignore[union-attr]
        return self.data.iloc[slc]  # type: ignore[union-attr]

    def show_stats(self):
        if not self.stats:
            self.stats = calculate_stats(self.broker)
        for key, value in self.stats.items():
            print(f"{key:30}: {value}")

    def get_trade_history(self) -> pd.DataFrame:
        """Trade-by-trade DataFrame across all assets."""
        return self.broker.get_trade_history()

    def plot(self):
        plot_with_bokeh(self.broker)
