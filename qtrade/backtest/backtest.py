import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from qtrade.backtest.strategy import Strategy
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
                 margin_ratio: float | dict[str, float] = 1.0,
                 trade_on_close: bool = False,
                 verbose: bool = False,
                 contract_multiplier: float | dict[str, float] | None = None,
                 ):
        """
        Args:
            data: Single OHLCV DataFrame, or a dict mapping asset symbol to
                its OHLCV DataFrame (multi-asset). All DataFrames must have a
                DatetimeIndex and the same index for proper portfolio accounting.
            strategy_class: Strategy class (subclass of Strategy) to instantiate.
            cash: Starting cash.
            commission: Commission calculator (None ⇒ no commission).
            margin_ratio: Margin requirement. Pass a scalar in (0, 1] to apply
                uniformly, or a dict keyed by asset for per-asset values
                (e.g. ``{"AAPL": 1.0, "GC": 0.05}`` for a stock + futures portfolio).
            trade_on_close: If True, market orders fill at the current bar's
                close price; otherwise at the next bar's open.
            verbose: Verbose logging.
            contract_multiplier: Contract size for futures-style instruments —
                the dollar P&L per 1 unit of price movement per contract. ``100``
                for COMEX gold (GC), ``50`` for E-mini S&P (ES), ``1000`` for
                crude (CL), ``1`` for stocks (the default). Pass a dict keyed by
                asset for portfolios mixing stocks and futures across categories.
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

        self.order_history: list[Order] = []
        self.stats = None

        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

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
            )
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
            )
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
