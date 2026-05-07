# Getting Started

This guide will help you get started with backtesting trading strategies using QTrade.

## Strategy

Let's create a simple SMA-crossover strategy. Custom strategies inherit from `qtrade.Strategy` and implement `prepare()` (one-time setup) and `on_bar_close()` (called on every bar):

```python
from qtrade.backtest import Strategy

class SMAStrategy(Strategy):

    def prepare(self):
        # Indicators are computed up-front so on_bar_close can read them cheaply.
        # self._data is the full DataFrame; mutate it freely here.
        self._data['SMA3'] = self._data['Close'].rolling(3).mean()
        self._data['SMA10'] = self._data['Close'].rolling(10).mean()

    def on_bar_close(self):
        # self.data is the slice up to the current bar (no look-ahead).
        sma3 = self.data['SMA3']
        sma10 = self.data['SMA10']
        # Golden cross → go long
        if sma3.iloc[-2] < sma10.iloc[-2] and sma3.iloc[-1] > sma10.iloc[-1]:
            self.buy()
        # Death cross → flatten
        elif sma3.iloc[-2] > sma10.iloc[-2] and sma3.iloc[-1] < sma10.iloc[-1]:
            self.close()
```

## Data

QTrade doesn't ship data loaders — bring your own OHLCV `pandas.DataFrame`
with `Open`, `High`, `Low`, `Close` (plus optional `Volume`) columns and a
`DatetimeIndex`. Column names are case-insensitive (`open`, `Open`, etc.).

In this guide, we'll use `yfinance`:

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
    multi_level_index=False,
)
```

Indicators can be added in the strategy's `prepare()` (as shown above) or
on the DataFrame before passing it to `Backtest`. Indicators added to the
input DataFrame are visible to the strategy via `self.data`.

## Backtest

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

```text
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

## Plot

Qtrade uses Bokeh to plot result charts. You can generate a plot of your backtest results with the following command:

```python
bt.plot()
```

<!-- 嵌入 HTML 文件 -->
<iframe src="../_static/demo.html" width="100%" height="600px" style="border:none;"></iframe>


## Trades

You can also check all trades by using the following commands:

```python
trade_details = bt.get_trade_history()
print(trade_details)
```

```text
     Asset  Type  Size  Entry Price   Exit Price Entry Time  Exit Date      Profit   Tag Exit Reason Duration
0  default  Long   2.0  1833.500000  1812.699951 2023-03-02 2023-03-08  -41.600098  None      signal   6 days
1  default  Long   2.0  1862.000000  2007.400024 2023-03-10 2023-04-18  290.800049  None      signal  39 days
... (rows omitted)
```

The `Asset` column is `"default"` for single-asset backtests and the
asset symbol when running on a portfolio (see [Multi-asset / portfolio backtests](multi_asset.md)).

## Optimizing parameters (and avoiding overfitting)

`Backtest.optimize` runs a grid search across the parameter space:

```python
best_params, best_stats, all_results = bt.optimize(
    n1=[3, 5, 10],
    n2=[10, 20, 30],
    maximize='Sharpe Ratio',
    constraint=lambda p: p['n1'] < p['n2'],
)
```

The catch: those "best" params are picked **on the same data they're
evaluated on**. They will look great in the backtest and frequently
disappoint live. Use `walk_forward_optimize` instead to get an
out-of-sample estimate:

```python
result = bt.walk_forward_optimize(
    train_window=120,        # bars used for parameter selection per window
    test_window=30,          # bars of out-of-sample evaluation that follow
    maximize='Sharpe Ratio',
    n1=[3, 5, 10],
    n2=[10, 20, 30],
    constraint=lambda p: p['n1'] < p['n2'],
)
print(result['summary'])
# {'n_windows': 7, 'mean_oos_return': 0.84, 'hit_rate': 0.57, ...}
```

Each window picks its own best params on the train slice and is
evaluated on the immediately following test slice; `summary` reports
aggregate out-of-sample stats and `windows` has per-window details.