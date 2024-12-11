import pandas as pd
import numpy as np
from typing import Optional, List, Type

import logging
from scipy.stats import gmean
from qtrade.backtest.strategy import Strategy
from tqdm import tqdm

from qtrade.core import Order, Broker, Commission
from qtrade.utils import calculate_stats, plot_with_bokeh


# Backtest class
class Backtest:
    def __init__(self,
                 data: pd.DataFrame,
                 strategy_class: Type[Strategy],
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
        :param margin: Margin requirements
        :param trade_on_close: If True, trades are filled on close price
        :param hedging: If True, allows hedging
        :param exclusive_orders: If True, only one order can be active at a time
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
        self.strategy = strategy_class(self.broker, self.data)
        self.current_bar = 0
        self.cash = cash
        self.commission = commission

        self.order_history: List[Order] = []
        self.stats = None

        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def run(self):
        """
        Run the backtest.
        """
        self.strategy.init()

        # skip the first n bars where data contains NaN
        start = 1 + np.argmax(self.data.notna().all(axis=1))
        
        for i in tqdm(range(start, len(self.data)), desc="Running Backtest"):
            self.current_bar = i
            current_time = self.data.index[i]

            self.broker.process_bar(current_time)

            self.strategy.next()

            self.broker.process_new_orders()

            self.broker.update_account_value_history()

        # Close all positions at the end
        self.broker.close_all_positions()
        self.broker.update_account_value_history()


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
        trade_history = self.broker.trade_history
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
        

