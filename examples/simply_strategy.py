import pandas as pd
import numpy as np
import pandas_ta as ta
from qtrade.backtest import Strategy, Backtest


class SimpleRSIStrategy(Strategy):
    def __init__(self, broker, data):
        super().__init__(broker, data)
        # 策略参数
        self.rsi_period = 14
        self.rsi_overbought = 80
        self.rsi_oversold = 20
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.risk_reward_ratio = 3.0
        self.current_date = None

    def init(self):
        """初始化策略，计算所需技术指标"""
        # 计算RSI
        self.raw_data['RSI'] = ta.rsi(self.raw_data['close'], length=self.rsi_period)
        # 计算ATR用于止损
        self.raw_data['ATR'] = ta.atr(
            self.raw_data['high'], self.raw_data['low'], self.raw_data['close'], length=self.atr_period
        )
        self.raw_data.dropna(inplace=True)

    def next(self):
        """每个时间步执行的策略逻辑"""
        current_time = self.data.index[-1]
        current_date = current_time.date()

        # 重置每日交易计数（在此策略中已移除每日交易次数限制）
        if self.current_date != current_date:
            self.current_date = current_date

        # 获取最新数据点
        rsi = self.data['RSI'].iloc[-1]
        rsi_prev = self.data['RSI'].iloc[-2]
        atr = self.data['ATR'].iloc[-1]
        close_price = self.data['close'].iloc[-1]
        pos_size = self.position.size

        # 判断是否为每日最后一个15分钟K线（例如15:45-16:00）
        # 根据您的数据时间范围调整
        # 假设交易时间为09:30-16:00，最后一个K线为15:45
        if current_time.hour == 15 and current_time.minute == 45:
            if pos_size != 0:
                self.close()
            return

        # 买入信号：RSI从下方穿过超卖区
        if rsi_prev <= self.rsi_oversold and rsi > self.rsi_oversold:
            if pos_size <= 0:  # 如果当前无仓位或持有空头，先平仓
                if pos_size < 0:
                    self.close()
                # 设置止损和止盈
                sl_price = close_price - 5
                tp_price = close_price + self.risk_reward_ratio * (close_price - sl_price)
                self.buy(size=1, sl=sl_price, tp=tp_price, tag='long_entry')
                return

        # 卖出信号：RSI从上方穿过超买区
        if rsi_prev >= self.rsi_overbought and rsi < self.rsi_overbought:
            if pos_size >= 0:  # 如果当前无仓位或持有多头，先平仓
                if pos_size > 0:
                    self.close()
                # 设置止损和止盈
                sl_price = close_price + 5
                tp_price = close_price - self.risk_reward_ratio * (sl_price - close_price)
                self.sell(size=1, sl=sl_price, tp=tp_price, tag='short_entry')
                return


def run_backtest(strategy_class, data, cash=10000):
    """运行回测"""
    bt = Backtest(
        data=data,
        strategy_class=strategy_class,
        cash=cash,
        commission=None,
        margin_ratio=1.0,
        trade_on_close=True,
        verbose=True,
    )
    bt.run()
    bt.show_stats()
    trade_details = bt.get_trade_history()
    print(trade_details)
    bt.plot()


if __name__ == "__main__":
    """加载并处理数据"""
    df = pd.read_csv('examples/data/XAUUSD_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    # 计算技术指标
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['SIGNAL'] = macd['MACDs_12_26_9']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['MA5'] = ta.sma(df['close'], length=5)
    df['MA20'] = ta.sma(df['close'], length=20)

    df.dropna(inplace=True)


    # 选择策略并运行回测
    print("Running RSI Strategy...")
    run_backtest(SimpleRSIStrategy, df)

    # print("\nRunning High Return Intraday Strategy...")
    # run_backtest(HighReturnIntradayStrategy, test_df)