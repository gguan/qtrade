import pandas as pd
import numpy as np
from math import copysign
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_with_plotly(broker):
    # 获取账户价值数据
    account_value = broker.account_value_history.copy(deep=True)
    datatime = account_value.index
    # 重置索引，使用整数索引绘图
    account_value.reset_index(drop=True, inplace=True)

    # 累计收益曲线
    cumulative_returns = account_value / account_value.iloc[0]

    # 计算最大回撤信息
    cumulative_max = account_value.cummax()
    drawdowns = (account_value - cumulative_max) / cumulative_max
    dd_max_time = drawdowns.idxmin()
    dd_max_val = cumulative_returns[dd_max_time]
    dd_max_drawdown = drawdowns.min() * 100

    final_time = cumulative_returns.index[-1]
    final_value = cumulative_returns[final_time]

    # 回撤持续时间
    drawdown_flag = drawdowns < 0
    drawdown_periods = drawdown_flag.ne(drawdown_flag.shift()).cumsum()
    drawdown_periods = drawdown_periods[drawdown_flag]
    drawdown_stats = drawdown_periods.groupby(drawdown_periods).agg(
        start=lambda x: x.index[0]-1,
        end=lambda x: x.index[-1],
        duration=lambda x: datatime[x.index[-1]] - datatime[x.index[0]-1]
    )
    max_dd_duration = None
    max_dd_start = None
    max_dd_end = None
    if not drawdown_stats.empty:
        longest_dd = drawdown_stats.loc[drawdown_stats['duration'].idxmax()]
        max_dd_duration = longest_dd['duration']
        max_dd_start = longest_dd['start']
        max_dd_end = longest_dd['end']
        max_dd_value = cumulative_returns[max_dd_start]

    # 获取交易数据
    trades = broker.trade_history
    datatime = broker.account_value_history.index
    index = np.arange(len(broker.account_value_history))
    trade_exit_idx = np.array([datatime.get_loc(t.exit_date) if t.exit_date in datatime else np.nan for t in trades])
    trade_exit_price = np.array([t.exit_price for t in trades])
    trade_size = np.array([t.size for t in trades])
    return_pct = np.array([copysign(1, t.size) * (t.exit_price / t.entry_price - 1) for t in trades])
    abs_size = np.abs(trade_size)
    marker_size = np.interp(abs_size, (abs_size.min(), abs_size.max()), (5, 10))

    returns_long = np.where(trade_size > 0, return_pct, np.nan)
    returns_short = np.where(trade_size < 0, return_pct, np.nan)

    # 创建Plotly子图布局，上图为账户价值曲线，下图为交易盈亏散点
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=("Cumulative Returns", "Trade Returns"))

    # 上图：累计收益曲线
    # 折线图
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                             mode='lines', name='Account Value',
                             line=dict(width=1.5, color='blue')), row=1, col=1)

    # 最终点
    fig.add_trace(go.Scatter(x=[final_time], y=[final_value],
                             mode='markers', name=f"Final ({final_value*100:.0f}%)",
                             marker=dict(color='blue', size=8)), row=1, col=1)

    # 最大回撤点
    fig.add_trace(go.Scatter(x=[dd_max_time], y=[dd_max_val],
                             mode='markers', name=f"Max Drawdown ({dd_max_drawdown:.1f}%)",
                             marker=dict(color='red', size=8)), row=1, col=1)

    # 最大回撤持续时间线（如果有）
    if max_dd_duration is not None:
        fig.add_trace(go.Scatter(x=[max_dd_start, max_dd_end], y=[max_dd_value, max_dd_value],
                                 mode='lines', name=f"Max Dd Dur. ({max_dd_duration})",
                                 line=dict(color='red', width=2)), row=1, col=1)

        # 绘制回撤区域填充（cumulative_returns与cummax的差）:
        # 我们用fill='tonexty'需要两条线，一条为cummax, 一条为cumreturns
        cummax_vals = cumulative_returns.cummax()
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cummax_vals,
                                 mode='lines', line=dict(color='rgba(255,0,0,0)'),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                                 mode='lines', fill='tonexty', 
                                 fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'),
                                 showlegend=False), row=1, col=1)

    # 下图：交易盈亏散点
    fig.add_trace(go.Scatter(x=trade_exit_idx, y=returns_long,
                             mode='markers', name='Long Trades',
                             marker=dict(color='green', size=marker_size)), row=2, col=1)
    fig.add_trace(go.Scatter(x=trade_exit_idx, y=returns_short,
                             mode='markers', name='Short Trades',
                             marker=dict(color='red', size=marker_size)), row=2, col=1)

    # 设置 y 轴格式为百分比
    fig.update_yaxes(tickformat=".2%", row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)

    # 添加一条水平线在第二个图中用于表示0%
    fig.add_shape(type="line", xref='x', yref='y2',
                  x0=index.min(), x1=index.max(), y0=0, y1=0,
                  line=dict(color="#666666", dash='dash'), row=2, col=1)

    # 更新布局
    fig.update_layout(
        height=600,
        title="Backtest Results with Plotly",
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=0, r=10, t=50, b=50)
    )

    return fig
