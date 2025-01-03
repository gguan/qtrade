import pandas as pd
import numpy as np
from typing import Any, Callable, Optional

import logging
from qtrade.backtest.strategy import Strategy
from tqdm import tqdm

from qtrade.core import Order, Broker, Commission
from qtrade.utils import calculate_stats, plot_with_bokeh


# Backtest class
class Backtest:
    def __init__(self,
                 data: pd.DataFrame,
                 strategy_class: type[Strategy],
                 cash: float = 10_000,
                 commission: Optional[Commission] = None,
                 margin_ratio: float = 1.0,
                 trade_on_close: bool = False,  # Determines fill_price_mode
                 verbose: bool = False,
                 ):
        """
        Initialize the backtest.

        :param data: DataFrame containing market data
        :param strategy_class: Strategy class to use
        :param cash: Starting cash
        :param commission: Commission per trade
        :param margin_ratio: Margin requirements
        :param trade_on_close: If True, trades are filled on close price
        :param verbose: If True, enables verbose logging
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")
        
        if {'open', 'high', 'low', 'close'} - set(col.lower() for col in data.columns):
            raise ValueError("Data must contain columns: 'open', 'high', 'low', 'close'")

        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        self.data = data.copy(deep=False)
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

    def run(self, **strategy_params):
        """
        Run the backtest.
        """
        self.strategy = self.strategy_class(self.broker, self.data, strategy_params)
        self.strategy.prepare()

        # skip the first n bars where data contains NaN
        start = 1 + np.argmax(self.strategy._data.notna().all(axis=1))

        for i in tqdm(range(start, len(self.data)), desc="Running Backtest"):
            self.current_bar = i
            current_time = self.data.index[i]

            self.broker.process_bar(current_time)

            self.strategy.on_bar_close()


        # Close all positions at the end
        self.broker.close_all_positions()

    def optimize(self,
                 maximize: str,
                 constraint: Optional[Callable[[Any], bool]] = None,
                 **params_grid):
        """
        Optimize strategy parameters.

        :param maximize: The metric name to compare, e.g., 'Equity Final [$]'
        :param constraint: Optional constraint function, e.g., lambda p: p.n1 < p.n2
        :param params_grid: Parameter ranges, e.g., n1=range(5, 30, 5), n2=range(10, 70, 5)
        :return: (best_params, best_stats, all_results)
        """
        from itertools import product

        best_params = None
        best_stats = None
        all_results = []

        # Convert params_grid to iterable parameter combinations [(n1, n2), (n1, n2), ...]
        keys = list(params_grid.keys())
        for combination in product(*params_grid.values()):
            # param_dict is like {'n1': 10, 'n2': 20, ...}
            param_dict = dict(zip(keys, combination))

            # If a constraint function is provided, check if it is satisfied
            if constraint and not constraint(param_dict):
                continue

            # Run backtest
            self.broker = Broker(self.data, self.cash, self.commission, self.margin_ratio, self.trade_on_close)
            self.run(**param_dict)
            stats = calculate_stats(self.broker)

            # Record the result
            result = {
                'params': param_dict,
                'stats': stats
            }
            all_results.append(result)

            # Update best
            if best_stats is None or stats[maximize] > best_stats[maximize]:
                best_stats = stats
                best_params = param_dict

        # Optimization complete, return best parameters, best metrics, and all results
        return best_params, best_stats, all_results

    def show_stats(self):
        if not self.stats:
            self.stats = calculate_stats(self.broker)
        for key, value in self.stats.items():
            print(f"{key:30}: {value}")

  
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get detailed information about all trades.

        :return: DataFrame with trade details
        """
        trade_history = self.broker.closed_trades
        return pd.DataFrame({
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
        # Implement plotting if needed
        plot_with_bokeh(self.broker)


