import yfinance as yf
from qtrade.backtest import Strategy, Backtest

class SMAStrategy(Strategy):
    def init(self):
        return super().init()
    def next(self):
        if self.data.SMA3[-2] < self.data.SMA10[-2] and \
            self.data.SMA3[-1] > self.data.SMA10[-1]:
            self.buy()
        if self.data.SMA3[-2] > self.data.SMA10[-2] and \
            self.data.SMA3[-1] < self.data.SMA10[-1]:
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
    data['SMA3'] = data['Close'].rolling(3).mean()
    data['SMA10'] = data['Close'].rolling(10).mean()

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

    