import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import logging
from typing import Optional
from qtrade.core import Broker, Commission
from qtrade.core.commission import NoCommission
from qtrade.core.order import Order
from qtrade.core.position import Position
from qtrade.core.trade import Trade
from qtrade.env.actions import ActionScheme, DefaultAction
from qtrade.env.rewards import RewardScheme, DefaultReward
from qtrade.env.observers import ObserverScheme, DefaultObserver
from qtrade.utils.plot_bokeh import plot_with_bokeh
from qtrade.utils.stats import calculate_stats

# 更新后的 TradingEnv 类
class TradingEnv(gym.Env):
    """
    自定义的 Gymnasium 交易环境, 使用 Position 类管理持仓
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 12}

    def __init__(self, data: pd.DataFrame, cash: float = 10000, 
                 commission: Optional[Commission] = None,
                 margin_ratio: float = 1.0,
                 trade_on_close: bool = True,
                 window_size: int = 10, 
                 max_steps = 3000, 
                 random_start: bool = False,
                 action_scheme: Optional[ActionScheme] = None,
                 reward_scheme: Optional[RewardScheme] = None,
                 observer_scheme: Optional[ObserverScheme] = None,
                 render_mode: Optional[str] = 'human',
                 verbose: bool = False
                 ):
        
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_scheme = action_scheme if action_scheme else DefaultAction()
        self.reward_scheme = reward_scheme if reward_scheme else DefaultReward()
        self.observer_scheme = observer_scheme if observer_scheme else DefaultObserver(
            window_size, [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        )
        

        self.action_space = self.action_scheme.action_space
        self.observation_space = self.observer_scheme.observation_space

        self._data = data.copy()
        self.cash = cash
        self.margin_ratio = margin_ratio
        self.commission = commission if commission else NoCommission()
        self.trade_on_close = trade_on_close

        self.max_steps = max_steps
        self.random_start = random_start
        self.window_size = window_size

        # 初始化图形
        self.fig, self.axes = None, None
        mc = mpf.make_marketcolors(up='limegreen', down='orangered',
                                   edge='inherit',
                                   wick={'up': 'limegreen', 'down': 'orangered'},
                                   volume='deepskyblue',
                                   ohlc='i')
        self.style = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8-whitegrid', marketcolors=mc)

        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
  
        self.reset()
        

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self.start_idx = self.window_size if not self.random_start else self.np_random.integers(
            self.window_size, len(self._data) - self.max_steps
        )

        self.current_step = self.start_idx
        self._broker = Broker(
            data = self._data.iloc[self.start_idx - self.window_size:].copy(), 
            cash = self.cash, 
            commission = self.commission, 
            margin_ratio = self.margin_ratio, 
            trade_on_close = self.trade_on_close
        )
        self.truncated = False
        self.terminated = False
        self.stats = None

        current_time = self._data.index[self.current_step]
        self._broker.process_bar(current_time)

        logging.info(f'Reset at {current_time}, Price: {self._data["Close"].loc[current_time]}')

        if self.reward_scheme:
            self.reward_scheme.reset()

        return self.observer_scheme.get_observation(self), {}

    def step(self, action):
        self.truncated = False
        self.current_step += 1
        current_time = self._data.index[self.current_step]

        self._broker.process_bar(current_time)
        
        orders = self.action_scheme.get_orders(action, self)

        self._broker.place_orders(orders)

        # 计算奖励
        reward = self.reward_scheme.get_reward(self)

        # 检查是否结束
        self.truncated = self.current_step - self.start_idx >= self.max_steps
        self.terminated = self.current_step >= len(self._data) - 1

        if self.terminated or self.truncated:
            self._broker.close_all_positions()

        # 构建下一个状态
        obs = self.observer_scheme.get_observation(self)

        trades_profit =sum([t.profit for t in self.closed_trades]) if self.closed_trades else 0
        avg_trade_duration = np.mean([t.exit_index - t.entry_index for t in self.closed_trades]) if self.closed_trades else 0
        
        # 附加信息
        info = {
            'equity': self._broker.equity,
            'unrealized_pnl': self._broker.unrealized_pnl,
            'cumulative_return': self._broker.cumulative_returns,
            'position': self._broker.position.size,
            'total_trades': len(self.closed_trades),
            'trades_profit': trades_profit,
            'avg_trade_duration': avg_trade_duration,
            'is_success': trades_profit > 0,
        }

        return obs, reward, self.terminated, self.truncated, info


    @property
    def current_time(self):
        """
        Get the current time.
        """
        return self._broker.current_time

    @property
    def position(self) -> Position:
        """
        Get the current position.
        """
        return self._broker.position

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the market data, can only see data up to the current index.

        """
        return self._data[ :self._broker.current_time]
    
    @property
    def filled_orders(self)  -> tuple[Order, ...]:
        """
        Get the filled orders.
        """
        return self._broker.filled_orders
    
    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """
        Get the closed trades.
        """
        return self._broker.closed_trades

    def render(self, mode=None):
         # 获取要绘制的数据
        data = self._broker.data.loc[:self._broker.current_time].tail(300)
        account_value = self._broker.equity_history.loc[:self._broker.current_time].tail(300)
        
        if self.fig is None:
            # 初次绘制
            addplots = [
                mpf.make_addplot(account_value,  color='orange', panel=1, label='Net Worth'),
            ]
            self.fig, self.axes = mpf.plot(
                data,
                type="candle",
                volume=False,
                addplot=addplots,
                returnfig=True,
                ylabel='Trade',
                ylabel_lower='Net Worth',
                style=self.style,
            )
        else:
            # 清空并重绘
            for ax in self.axes:
                ax.clear()

            addplots = [
                mpf.make_addplot(account_value, color='orange', panel=1, label='Equity',ax=self.axes[2], secondary_y=False),
            ]

            # 提取在当前窗口内的买卖订单
            buy_df = pd.DataFrame(index=data.index, columns=['price'])
            sell_df = pd.DataFrame(index=data.index, columns=['price'])

            for order in self._broker.filled_orders:
                try:
                    if order._is_filled:
                        if order.size > 0:
                            buy_df.index.get_loc(order._fill_date)
                            buy_df.loc[order._fill_date, 'price'] = order._fill_price - 10
                        elif order.size < 0:
                            sell_df.index.get_loc(order._fill_date)
                            sell_df.loc[order._fill_date, 'price'] = order._fill_price + 10
                except KeyError:
                    # 如果 order['entry_date'] 不在 self.data.index 中, 跳过
                    continue
            
            # 定义标记样式
            if 'price' in buy_df.columns and not buy_df['price'].isna().all():
                addplots.append(
                    mpf.make_addplot(buy_df['price'], type='scatter', markersize=50, marker='^', color='g',ax=self.axes[0])
                )
            if 'price' in sell_df.columns and not sell_df['price'].isna().all():
                addplots.append(
                    mpf.make_addplot(sell_df['price'], type='scatter', markersize=50, marker='v', color='r',ax=self.axes[0])
                )

            # 准备蜡烛图数据
            mpf.plot(
                data,
                type="candle",
                ax=self.axes[0],
                volume=False,
                addplot=addplots,
                ylabel='Trade',
                ylabel_lower='Net Worth',
                style=self.style,
            )
            self.axes[2].set_ylabel('Net Equity')

        total_trades = len(self._broker.closed_trades)
        success_trades = len([t for t in self._broker.closed_trades if t.profit > 0])
        failed_trades = total_trades - success_trades
        display_title = f'Trading Gym Env Step:{self.current_step - self.start_idx}' \
                        f'\nEquity:{self._broker.equity:.2f} Position:{self.position.size} ' \
                        f'Unrealized Pnl:{self._broker.unrealized_pnl}' \
                        f'\nTotal trades: {total_trades}, success trades: {success_trades}, ' \
                        f'failed trades: {failed_trades}, active trades: {len(self._broker.position.active_trades)}'
        # 设置标题显示当前净值
        self.fig.suptitle(display_title)

        render_mode = mode if mode else self.render_mode
        if render_mode not in self.metadata['render_modes']:
            raise ValueError(f"Unsupported render mode: {mode}")

        if render_mode == 'human':
            plt.pause(0.01)  # 模拟实时更新的延迟
        elif render_mode == 'rgb_array':
            # 返回RGB array
            # 将绘图渲染到canvas, 然后转换为RGB数组
            self.fig.canvas.draw()
            actual_width = int(self.fig.get_size_inches()[0] * self.fig.dpi)
            actual_height = int(self.fig.get_size_inches()[1] * self.fig.dpi)
            if hasattr(self.fig.canvas, 'tostring_rgb'):
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
                img = img.reshape((actual_height, actual_width, 3))  # RGB has 3 channels
            elif hasattr(self.fig.canvas, 'tostring_argb'):
                img = np.frombuffer(self.fig.canvas.tostring_argb(), dtype='uint8')
                img = img.reshape((actual_height, actual_width, 4))  # ARGB has 4 channels
                img = img[:, :, 1:]  # 去掉Alpha通道, 保留RGB通道
            else:
                raise RuntimeError("Canvas does not support tostring_rgb or tostring_argb")
            return img
        

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def close(self):
        plt.close()

    def show_stats(self):
        if not self.stats:
            self.stats = calculate_stats(self._broker)
        for key, value in self.stats.items():
            print(f"{key:30}: {value}")

  
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get detailed information about all trades.

        :return: DataFrame with trade details
        """
        trade_history = self._broker.closed_trades
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
    
    def plot(self, filename=None):
        plot_with_bokeh(self._broker, filename=filename)