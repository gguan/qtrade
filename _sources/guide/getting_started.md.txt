# Getting Started

This guide will help you get started with backtesting trading strategies using QTrade.

## Strategy

Let's create a simple SMA strategy to backtest. Our custome strategy need inherit from qtrade.Strategy and implement init() and next() method

```python
from qtrade.backtest import Strategy

class SMAStrategy(Strategy):
    def init(self):
        return super().init()
    def next(self):
        if self.data.SMA3[-2] < self.data.SMA10[-2] and \
            self.data.SMA3[-1] > self.data.SMA10[-1]:
            self.buy()
        if self.data.SMA10[-2] > self.data.SMA10[-2] and \
            self.data.SMA10[-1] < self.data.SMA10[-1]:
            self.close()
```

## Data

For simplicity, the QTrade library does not provide a data processing component. Users need to prepare OHLC (Open, High, Low, Close) data as a `pandas.DataFrame` with the required columns: 'Open', 'High', 'Low', 'Close', and optionally 'Volume'. The data frame should also include any indicators that will be used.

In this guide, we'll use `yfinance` to obtain the data:

```bash
$ pip install yfinance
```

```python
import yfinance as yf

# Download gold data with daily intervals
data = yf.download(
    "GC=F", 
    start="2023-01-01", 
    end="2024-01-01", 
    interval="1d", 
    multi_level_index=False
)
data['SMA3'] = data['Close'].rolling(3).mean()
data['SMA10'] = data['Close'].rolling(10).mean()
```

# Backtest

Now let's backtest our stratgy on prepared data.

```python

from qtrade.backtest import Backtest

bt = Backtest(
    data=data,
    strategy_class=SMAStrategy,
    cash=10000,
)
bt.run()

# Show backtest results
bt.show_stats()
```

```plaintext
Start                         : 2023-01-03 00:00:00
End                           : 2023-12-29 00:00:00
Duration                      : 360 days 00:00:00
Start Value                   : 5000.0
End Value                     : 5504.3994140625
Total Return [%]              : 10.08798828125
Total Commission Cost[%]      : 0
Buy & Hold Return [%]         : 12.10523221626531
Return (Ann.) [%]             : 18.074280342264217
Volatility (Ann.) [%]         : 7.44
Max Drawdown [%]              : -3.8180767955781354
Max Drawdown Duration         : 186 days 00:00:00
Total Trades                  : 14
Win Rate [%]                  : 42.857142857142854
Best Trade [%]                : 239.0
Worst Trade [%]               : -58.0
Avg Winning Trade [%]         : 126.69986979166667
Avg Losing Trade [%]          : -31.9749755859375
Avg Winning Trade Duration    : 21 days 00:00:00
Avg Losing Trade Duration     : 4 days 15:00:00
Profit Factor                 : 2.9718522251363866
Expectancy                    : 36.028529575892854
Sharpe Ratio                  : 2.271145457445316
Sortino Ratio                 : 2.6654676881799224
Calmar Ratio                  : 4.733870299098423
Omega Ratio                   : 1.7873724100138564
```

plot result
```python
bt.plot()
```


<!-- 嵌入 HTML 文件 -->
<iframe src="../_static/test.html" width="100%" height="600px" style="border:none;"></iframe>
