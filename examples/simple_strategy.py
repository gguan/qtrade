import yfinance as yf
from qtrade.backtest import Strategy, Backtest

class SMAStrategy(Strategy):
    n1 = 3
    n2 = 10
    def prepare(self):
        self._data[f'SMA_{self.n1}'] = self._data['Close'].rolling(self.n1).mean()
        self._data[f'SMA_{self.n2}'] = self._data['Close'].rolling(self.n2).mean()

    def on_bar_close(self):
        sma_n1_prev = self.data[f'SMA_{self.n1}'].iloc[-2]
        sma_n2_prev = self.data[f'SMA_{self.n2}'].iloc[-2]
        sma_n1_last = self.data[f'SMA_{self.n1}'].iloc[-1]
        sma_n2_last = self.data[f'SMA_{self.n2}'].iloc[-1]

        # Golden Cross
        if sma_n1_prev < sma_n2_prev and sma_n1_last > sma_n2_last:
            self.buy()
        
        # Dead Cross
        elif sma_n1_prev > sma_n2_prev and sma_n1_last < sma_n2_last:
            self.close()

if __name__ == "__main__":

    """Download data from Yahoo Finance"""
    data = yf.download(
            "GC=F", 
            start="2023-01-01", 
            end="2024-01-01", 
            interval="1d", 
            multi_level_index=False
    )

    # data.reset_index(inplace=True)
    print(data.head())

    """Run backtest with SMAStrategy"""
    bt = Backtest(
        data=data,
        strategy_class=SMAStrategy,
        cash=5000,
        commission=None,
        margin_ratio=0.5,
        trade_on_close=True,
    )

    bt.run()
    bt.show_stats()
    trade_details = bt.get_trade_history()
    print(trade_details)
    bt.plot()

    # best_params, best_stats, all_results = bt.optimize(
    #     n1=[3, 5, 10, 15],
    #     n2=[5, 10, 20, 30, 40],
    #     maximize='Total Return [%]',
    #     constraint=lambda p: p['n1'] < p['n2']
    # )

    # for result in all_results:
    #     print(result['params'], result['stats']['Total Return [%]'])
    
    # print("Optimal Parameters:", best_params)
    # print("Optimal Results:", best_stats)

