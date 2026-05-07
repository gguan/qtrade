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
                 margin_ratio: float = 1.0,
                 trade_on_close: bool = False,
                 verbose: bool = False,
                 ):
        """
        Args:
            data: Single OHLCV DataFrame, or a dict mapping asset symbol to
                its OHLCV DataFrame (multi-asset). All DataFrames must have a
                DatetimeIndex and the same index for proper portfolio accounting.
            strategy_class: Strategy class (subclass of Strategy) to instantiate.
            cash: Starting cash.
            commission: Commission calculator (None ⇒ no commission).
            margin_ratio: Margin requirement (0 < ratio ≤ 1).
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

        self.broker = Broker(self.data, cash, commission, margin_ratio, trade_on_close)
        self.strategy_class = strategy_class
        self.current_bar = 0
        self.cash = cash
        self.commission = commission
        self.margin_ratio = margin_ratio
        self.trade_on_close = trade_on_close

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
            self.broker = Broker(self.data, self.cash, self.commission, self.margin_ratio, self.trade_on_close)
            self.run(**param_dict)
            stats = calculate_stats(self.broker)

            all_results.append({'params': param_dict, 'stats': stats})

            if best_stats is None or stats[maximize] > best_stats[maximize]:
                best_stats = stats
                best_params = param_dict

        return best_params, best_stats, all_results

    def show_stats(self):
        if not self.stats:
            self.stats = calculate_stats(self.broker)
        for key, value in self.stats.items():
            print(f"{key:30}: {value}")

    def get_trade_history(self) -> pd.DataFrame:
        """Trade-by-trade DataFrame across all assets."""
        trade_history = self.broker.closed_trades
        return pd.DataFrame({
            'Asset': [trade.asset for trade in trade_history],
            'Type': ['Long' if trade.is_long else 'Short' for trade in trade_history],
            'Size': [trade.size for trade in trade_history],
            'Entry Price': [trade.entry_price for trade in trade_history],
            'Exit Price': [trade.exit_price for trade in trade_history],
            'Entry Time': [trade.entry_date for trade in trade_history],
            'Exit Date': [trade.exit_date for trade in trade_history],
            'Profit': [trade.profit for trade in trade_history],
            'Tag': [trade.tag for trade in trade_history],
            'Exit Reason': [trade.exit_reason for trade in trade_history],
            'Duration': [trade.exit_date - trade.entry_date for trade in trade_history],
        })

    def plot(self):
        plot_with_bokeh(self.broker)
