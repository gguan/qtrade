import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import logging
from typing import Optional, List
from qtrade.core import Position, Order, Broker, Commission
from qtrade.env.actions import ActionScheme, DefaultAction
from qtrade.env.rewards import RewardScheme, DefaultReward

# 更新后的 TradingEnv 类
class TradingEnv(gym.Env):
    """
    自定义的 Gymnasium 交易环境，使用 Position 类管理持仓
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 commission: Optional[Commission] = None,
                 margin_ratio: float = 1.0,
                 window_size: int = 20, max_position: int = 2, trade_on_close: bool = True,
                 timeseries_features: List = [], static_features: List = [], max_steps = 3000, random_start: bool = False,
                 action_scheme: ActionScheme = DefaultAction(),
                 reward_scheme: RewardScheme = DefaultReward()):
        super(TradingEnv, self).__init__()
        print(action_scheme)
        print(reward_scheme)
        self.action_scheme = action_scheme 
        self.reward_scheme = reward_scheme

        self.data = data.copy()
        self.initial_balance = initial_balance
        self.margin_ratio = margin_ratio
        self.commission = commission
        self.max_postion = abs(max_position)
        self.max_steps = max_steps
        self.random_start = random_start
        self.window_size = window_size
        self.hedging = False
        self.trade_on_close = trade_on_close

        self.episode_number = 0

        self.timeseries_features = timeseries_features
        self.static_features = static_features

        # 初始化图形
        self.fig, self.axes = None, None
        mc = mpf.make_marketcolors(up='limegreen', down='orangered',
                                   edge='inherit',
                                   wick={'up': 'limegreen', 'down': 'orangered'},
                                   volume='deepskyblue',
                                   ohlc='i')
        self.style = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8-whitegrid', marketcolors=mc)

        self.action_space = action_scheme.action_space

        # 状态空间：过去 window_size 天的收盘价 + 余额 + 持仓大小 + 净值 + 移动平均线
        # 例如，window_size = 10，则状态维度为 10 + 4 = 14
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(window_size + 4,), dtype=np.float32
        # )
        self.observation_space = gym.spaces.Dict({
            "timeseries_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, len(timeseries_features)), dtype=np.float32),
            # "static_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(static_features),), dtype=np.float32),
            # position_size, position_pnl
            "balances": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        # 定义观察空间
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.window_size * len(timeseries_features) + 2,),
        #     dtype=np.float32
        # )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        
        self.cash = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.broker = Broker(self.data, self.initial_balance, self.commission, self.margin_ratio, self.trade_on_close)
        self.start_idx = self.window_size if not self.random_start else np.random.randint(self.window_size, len(self.data) - self.max_steps)
        self.current_step = self.start_idx
        self.truncated = False
        self.terminated = False
        self.step_returns = []
        self.order_history = []
        self.position_history = [0]
        self.closed_trades = []
        self.last_trade = None
        self.episode_number += 1
        logging.debug(f'Reset at {self.data.index[self.current_step]}, price: {self.data["close"].iloc[self.current_step]}')

        if self.reward_scheme:
            self.reward_scheme.reset()

        return self._get_observation(), {}

    def _get_observation(self):

        start = max(0, self.current_step - self.window_size+1)
        timeseries_features = self.data[self.timeseries_features].iloc[start:self.current_step+1].values
        
        static_features = self.data[self.static_features].iloc[self.current_step].values
        
        current_price = self.data['close'].iloc[self.current_step]
        unrealized_pnl = self.broker.unrealized_pnl
        
        balances = np.array([
            self.broker.position.size / self.max_postion,
            np.sign(unrealized_pnl) * np.log(1 + abs(unrealized_pnl)) # log scaled pnl
        ])
        # balances = np.array([
        #     self.position.size,
        #     pnl
        # ])

        # 合并状态
        obs = {
            "timeseries_features": timeseries_features.astype(np.float32),
            # "static_features": static_features.astype(np.float32),
            "balances": balances.astype(np.float32)
        }
        # print(timeseries_features)
        # print(balances)
        # obs = np.concatenate([timeseries_features.flatten(), balances], axis=0).astype(np.float32)
        # print(obs)
        return obs
    

    def step(self, action):
        self.truncated = False
        self.current_step += 1
        current_time = self.data.index[self.current_step]
        self.closed_trades = []
        logging.debug(f"action:", action)
        # 初始化奖励
        reward = 0

        self.broker.process_bar(current_time)
        
        orders = self.action_scheme.get_orders(action, self)
        self.broker.process_new_orders(orders)
        
        self.broker.update_account_value_history()

        # 计算净值
        prev_net_worth = self.net_worth

        # 计算持仓价值
        position_value = self.broker.unrealized_pnl

        self.net_worth = self.broker.account_value
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.position_history.append(self.broker.position.size)

        # 计算每步收益
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth != 0 else 0
        self.step_returns.append(step_return)
        if len(self.step_returns) > 1000:
            self.step_returns.pop(0)

        # 计算夏普比率
        if len(self.step_returns) > 1 and np.std(self.step_returns) != 0:
            sharpe_ratio = np.mean(self.step_returns) / np.std(self.step_returns)
        else:
            sharpe_ratio = 0

        # 计算奖励
        reward = self.reward_scheme.get_reward(self)

        # 检查是否结束
        self.truncated = self.current_step - self.start_idx >= self.max_steps
        self.terminated = self.current_step >= len(self.data) - 1

        # 构建下一个状态
        obs = self._get_observation()

        # 附加信息
        info = {
            'net_worth': self.net_worth,
            'balance': self.cash,
            'position_size': self.position.size,
            'total_profit': self.net_worth - self.initial_balance,
            'sharpe_ratio': sharpe_ratio,
        }

        return obs, reward, self.terminated, self.truncated, info



    def render(self, mode='human'):
         # 获取要绘制的数据
         # 提取从 episode 开始到当前步骤的数据
        start = self.start_idx
        end = self.current_step + 1  # 包含当前步骤
        data = self. data.iloc[start:end].copy()

        if self.fig is None:
            addplots = [
                mpf.make_addplot(self.net_worth_history,  color='orange', panel=1, label='Net Worth'),
                mpf.make_addplot(self.position_history, type='bar', width=0.7, color='deepskyblue', panel=1, label='Position', ylim=(-self.max_postion, self.max_postion)),
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
            for ax in self.axes:
                ax.clear()

            addplots = [
                # mpf.make_addplot(self.net_worth_history, color='orange', panel=1, label='Net Worth',ax=self.axes[2], secondary_y=False),
                mpf.make_addplot(self.position_history, type='bar', width=0.6, color='deepskyblue', panel=1, label='Position', ylim=(-self.max_postion, self.max_postion), ax=self.axes[3]),
            ]

            # 提取在当前窗口内的买卖订单
            buy_df = pd.DataFrame(index=data.index)
            sell_df = pd.DataFrame(index=data.index)
            for order in self.order_history:
                try:
                    trade_step = order['entry_step']
                    if start <= trade_step < end:
                        trade_time = self.data.index[trade_step]
                        if order['size'] > 0:
                            buy_df.loc[trade_time, 'price'] = order['entry_price'] - 2
                        elif order['size'] < 0:
                            sell_df.loc[trade_time, 'price'] = order['entry_price'] + 2
                except KeyError:
                    # 如果 order['entry_date'] 不在 self.data.index 中，跳过
                    continue

            # 定义标记样式
            if 'price' in buy_df.columns and not buy_df['price'].isna().all():
                buy_plot = mpf.make_addplot(buy_df['price'], type='scatter', markersize=100, marker='^', color='g',ax=self.axes[0])
                addplots.append(buy_plot)
            if 'price' in sell_df.columns and not sell_df['price'].isna().all():
                sell_plot = mpf.make_addplot(sell_df['price'], type='scatter', markersize=100, marker='v', color='r',ax=self.axes[0])
                addplots.append(sell_plot)

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
            self.axes[2].plot(self.net_worth_history, color='orange')
            self.axes[2].set_ylabel('Net Worth')

        current_price = data['close'].iloc[-1]
        total_trades = len(self.trade_history)
        success_trades = len([t for t in self.trade_history if t.profit > 0])
        failed_trades = total_trades - success_trades
        display_title = f'Trading Gym Env Episode:{self.episode_number} Step:{self.current_step - self.start_idx}' \
                        f'\nNet Worth:{self.net_worth:.2f} Position:{self.position.size} Unrealized Pnl:{self.position.position_pnl(current_price)}' \
                        f'\nTotal trades: {total_trades}, success trades: {success_trades}, failed trades: {failed_trades}, active trades: {len(self.position.active_trades)}'
        # 设置标题显示当前净值
        self.fig.suptitle(display_title)

        plt.pause(0.01)  # 模拟实时更新的延迟
        
        
    def render_all(self, savefig: Optional[str] = None):
        
        # 确保当前步骤不超过数据范围
        assert self.current_step >= self.start_idx, "当前步骤在起始索引之前，无法渲染图表。"

        # 提取从 episode 开始到当前步骤的数据
        start = self.start_idx
        end = self.current_step + 1  # 包含当前步骤
        data = self.data[['open', 'high', 'low', 'close']].iloc[start:end].copy()
        
        # 提取在当前窗口内的买卖订单
        buy_df = pd.DataFrame(index=data.index)
        sell_df = pd.DataFrame(index=data.index)
       
        for order in self.order_history:
            try:
                trade_step = order['entry_step']
                if start <= trade_step < end:
                    trade_time = self.data.index[trade_step]
                    if order['size'] > 0:
                        buy_df.loc[trade_time, 'price'] = order['entry_price']
                    elif order['size'] < 0:
                        sell_df.loc[trade_time, 'price'] = order['entry_price']
            except KeyError:
                # 如果 order['entry_date'] 不在 self.data.index 中，跳过
                continue

        # 定义标记样式
        addplots = [
                mpf.make_addplot(self.net_worth_history,  color='orange', panel=1, label='Net Worth'),
                mpf.make_addplot(self.position_history, type='step', color='deepskyblue', panel=1, label='Position'),
            ]
        if 'price' in buy_df.columns and not buy_df['price'].isna().all():
            buy_plot = mpf.make_addplot(buy_df['price'], type='scatter', markersize=50, marker='^', color='g')
            addplots.append(buy_plot)
        if 'price' in sell_df.columns and not sell_df['price'].isna().all():
            sell_plot = mpf.make_addplot(sell_df['price'], type='scatter', markersize=50, marker='v', color='r')
            addplots.append(sell_plot)
        
        current_price = data['close'].iloc[-1]
        total_trades = len(self.trade_history)
        success_trades = len([t for t in self.trade_history if t.profit > 0])
        failed_trades = total_trades - success_trades
        display_title = f'Trading Gym Env Episode:{self.episode_number} Step:{self.current_step - self.start_idx}' \
                        f'\nNet Worth:{self.net_worth:.2f} Position:{self.position.size} Unrealized Pnl:{self.position.position_pnl(current_price)}' \
                        f'\nTotal trades: {total_trades}, success trades: {success_trades}, failed trades: {failed_trades}, active trades: {len(self.position.active_trades)}'

        # 获取图形和轴对象
        fig, _ = mpf.plot(
            data,
            type='line',
            addplot=addplots,
            volume=False,
            show_nontrading=False,
            style=self.style,
            returnfig=True
        )
        fig.suptitle(display_title)

        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def close(self):
        plt.close()