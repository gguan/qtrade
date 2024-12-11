from math import copysign
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import ColumnDataSource, HoverTool, Span, NumeralTickFormatter, DatetimeTickFormatter, Range1d,LinearAxis
from bokeh.layouts import gridplot, layout
from bokeh.io import curdoc
from bokeh.models import LinearColorMapper
from bokeh.models import CustomJSTickFormatter
from bokeh.transform import factor_cmap
from bokeh.colors.named import lime as BULL_COLOR, tomato as BEAR_COLOR

from qtrade.core.broker import Broker

# 假设在Jupyter中使用，如果在脚本中请使用 output_file 或其它方式
# output_notebook()
# output_file('backtest.html')

def _plot_account_value(broker: Broker):
    account_value = broker.account_value_history.loc[:broker.current_time].copy(deep=True)

    datatime = account_value.index

    account_value.reset_index(drop=True, inplace=True)

    # 累计收益曲线（从初始值归一化）
    cumulative_returns = account_value / account_value[0]

    # source = ColumnDataSource(pd.DataFrame({
    #     'datetime': datatime,
    #     'account_value': account_value,
    #     'cumulative_returns': cumulative_returns,
    # }))
    # source.data['index'] = np.arange(len(cumulative_returns))
    # 创建顶部的权益曲线图（包含最大回撤与持续时间）
    fig1 = figure(height=100, tools="xpan,xwheel_zoom,reset,save",
                        active_drag='xpan', active_scroll='xwheel_zoom')
    fig1.line(cumulative_returns.index, cumulative_returns, line_width=1.5, line_alpha=1)
    
    # fig_equity.add_tools(hover)

    peak_time = cumulative_returns.idxmax()
    peak_value = cumulative_returns[peak_time]
    fig1.scatter(peak_time, peak_value, color='cyan', size=8, legend_label=f"Peak ({peak_value*100:.0f}%)")

    final_time = cumulative_returns.index[-1]
    final_value = cumulative_returns[final_time]
    fig1.scatter(final_time, final_value, color='blue', size=8, legend_label=f"Final ({final_value*100:.0f}%)")

    # 最大回撤点
    cumulative_max = account_value.cummax()
    
    drawdowns = (account_value - cumulative_max) / cumulative_max
    
    # 识别峰值点（累计最大值更新的点）
    dd_max_time = drawdowns.idxmin()
    dd_max_val = cumulative_returns[dd_max_time]
    dd_max_drawdown = drawdowns.min() * 100
    fig1.scatter(dd_max_time, dd_max_val, color=BEAR_COLOR, size=8,
                      legend_label=f"Max Drawdown ({dd_max_drawdown:.1f}%)")

    # 最大回撤持续时间（在图上用红色水平线段标记回撤开始结束）
    # 假设回撤持续时间对应的起始与结束点与dd_max有关，这里简化为在peak与最大dd点之间画线
    # 实际中可根据前面drawdown_periods细化处理
    # 标记回撤阶段
    drawdown_flag = drawdowns < 0
    drawdown_periods = drawdown_flag.ne(drawdown_flag.shift()).cumsum()
    drawdown_periods = drawdown_periods[drawdown_flag]
    # 计算每个回撤阶段的持续时间
    drawdown_stats = drawdown_periods.groupby(drawdown_periods).agg(
        start=lambda x: x.index[0]-1,
        end=lambda x: x.index[-1],
        duration=lambda x: datatime[x.index[-1]] - datatime[x.index[0]-1]
    )
   
    # 找出最长的回撤持续时间
    if not drawdown_stats.empty:
        longest_dd = drawdown_stats.loc[drawdown_stats['duration'].idxmax()]
        max_dd_duration = longest_dd['duration']
        max_dd_start = longest_dd['start']
        max_dd_end = longest_dd['end']
        max_dd_value= cumulative_returns[max_dd_start]
        fig1.line([max_dd_start, max_dd_end], max_dd_value,
                    color=BEAR_COLOR, line_width=2, legend_label=f"Max Dd Dur. ({max_dd_duration})")
        fig1.varea(x=cumulative_returns.index, y1=cumulative_returns, y2=cumulative_returns.cummax(), color='red', alpha=0.2)

    # if relative_account_value:
    fig1.yaxis.formatter = NumeralTickFormatter(format='0,0.[00]%')
    fig1.legend.title = 'Account Value'
    fig1.xaxis.visible = False

    # 再添加一个工具，查看drawdown
    hover_dd = HoverTool(tooltips=[("Drawdown", "@drawdown{0.0}%")], formatters={"@datetime": "datetime"}, mode='vline')
    fig1.add_tools(hover_dd)
    return fig1

def _plot_trades(broker: Broker, x_range):
    # 交易记录处理：在价格图中标记交易点（进入与退出），并用不同颜色表示盈亏
    trades = broker.trade_history
    datatime = broker.account_value_history.loc[:broker.current_time].index
    index = np.arange(len(broker.account_value_history.loc[:broker.current_time]))
    trade_source = ColumnDataSource(dict(
        index=np.array([datatime.get_loc(trade.exit_date) if trade.exit_date in datatime else np.nan for trade in trades ]),
        # index=index,
        datetime=np.array([trade.exit_date for trade in trades]),
        exit_price=np.array([trade.exit_price for trade in trades]),
        size=np.array([trade.size for trade in trades]),
        return_pct=np.array([copysign(1, trade.size) * (trade.exit_price / trade.entry_price - 1) for trade in trades]),
    ))
    size = np.abs(trade_source.data['size'])
    size = np.interp(size, (size.min(), size.max()), (5, 10))
    trade_source.add(size, 'marker_size')
   
    returns_long = np.where(trade_source.data['size'] > 0, trade_source.data['return_pct'], np.nan)
    returns_short = np.where(trade_source.data['size'] < 0, trade_source.data['return_pct'], np.nan)
    trade_source.add(returns_long, 'returns_long')
    trade_source.add(returns_short, 'returns_short')
    
    fig2 = figure(height=100, tools="xpan,xwheel_zoom,reset,save",
                        active_drag='xpan', active_scroll='xwheel_zoom',
                        x_range=x_range)
    fig2.add_layout(Span(location=0, dimension='width', line_color='#666666',
                            line_dash='dashed', line_width=1))

    r1 = fig2.scatter('index', 'returns_long', source=trade_source, color=BULL_COLOR, size='marker_size', line_color='black', line_width=0.5, legend_label='Profit')
    r2 = fig2.scatter('index', 'returns_short', source=trade_source, color=BEAR_COLOR, size='marker_size', line_color='black', line_width=0.5, legend_label='Loss')
    tooltips = [("Size", "@size{0,0}")]
    # fig2.add_tools(HoverTool(tooltips=tooltips, formatters={"@datetime": "datetime"}, vline=False, renderers=[r1]))
    # fig2.add_tools(HoverTool(tooltips=tooltips, formatters={"@datetime": "datetime"}, vline=False, renderers=[r2]))
    
    fig2.legend.title = 'Trades - Net Profit/Loss'
    fig2.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    fig2.xaxis.visible = False

    return fig2

def _plot_ohlc(broker: Broker, x_range):
    source = ColumnDataSource(broker.data.loc[:broker.current_time])
    source.add((broker.data.loc[:broker.current_time].close >= broker.data.loc[:broker.current_time].open).values.astype(np.uint8).astype(str), 'inc')
    source.data['index'] = np.arange(len(broker.data.loc[:broker.current_time]))
    source.data['datetime'] = broker.data.loc[:broker.current_time].index

    fig3 = figure(height=300, tools="xpan,xwheel_zoom,reset",
                       active_drag='xpan', active_scroll='xwheel_zoom',
                       x_range=x_range
                       )
    fig3.line('index', 'close', source=source, line_width=1.5, color='black')

    y_min = broker.data['close'].loc[:broker.current_time].min()
    y_max = broker.data['close'].loc[:broker.current_time].max()
    fig3.y_range = Range1d(start=y_min * 0.99, end=y_max * 1.01)
    # 设置额外的 Y 轴范围用于成交量
    fig3.extra_y_ranges = {"volume": Range1d(start=0, end=broker.data.volume.loc[:broker.current_time].mean() * 8)}
    
    # 添加第二个 Y 轴（右侧）用于成交量
    fig3.add_layout(
        LinearAxis(y_range_name="volume", axis_label="Volume"),
        'right'
    )

    # 使用 factor_cmap 映射颜色
    color_mapper = factor_cmap('inc', palette=[BEAR_COLOR, BULL_COLOR], factors=['0', '1'])

    # 绘制成交量柱状图，绑定到第二个 Y 轴
    volume_bars = fig3.vbar(
        x='index',
        top='volume',
        width=0.8,
        source=source,
        color=color_mapper,
        alpha=0.2,
        y_range_name="volume"
    )

    # 添加成交量的 Hover 工具
    hover_volume = HoverTool(
        tooltips=[
            ("Volume", "@volume{0,0}"),
            ("Index", "@index")
        ],
        renderers=[volume_bars],
        mode='vline'
    )
    fig3.add_tools(hover_volume)

    # 格式化 y 轴
    fig3.yaxis[0].formatter = NumeralTickFormatter(format="0.[00]")
    fig3.yaxis[1].formatter = NumeralTickFormatter(format="0,0")

    fig3.xaxis.formatter = CustomJSTickFormatter(
            args=dict(axis=fig3.xaxis[0],
                      formatter=DatetimeTickFormatter(days='%a, %d %b',
                                                      months='%m/%Y'),
                      source=source),
            code='''
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
        ''')
    
    trades = broker.trade_history
    datatime = broker.account_value_history.loc[:broker.current_time].index
    trade_source = ColumnDataSource(dict(
        top=np.array([trade.exit_price for trade in trades]),
        bottom=np.array([trade.entry_price for trade in trades]),
        left=np.array([datatime.get_loc(trade.entry_date) if trade.entry_date in datatime else np.nan for trade in trades ]),
        right=np.array([datatime.get_loc(trade.exit_date) if trade.exit_date in datatime else np.nan for trade in trades ]),
        color=np.where(np.array([trade.profit for trade in trades]) > 0, BULL_COLOR, BEAR_COLOR),
    ))
    # 绘制交易信号
    fig3.quad(left='left', right='right', top='top', bottom='bottom', source=trade_source, color='color', alpha=0.2, legend_label=f'Trades({len(trades)})')
    
    # 绘制买卖点
    orders = [order for order in broker.order_history if order._is_filled]
    order_source = ColumnDataSource(dict(
        index=np.array([datatime.get_loc(order._fill_date) if order._fill_date in datatime else np.nan for order in orders ]),
        size=np.array([abs(order.size) for order in orders]),
        fill_price=np.array([order._fill_price - 10 if order.size > 0 else order._fill_price + 10 for order in orders]),
        color_mapper=np.where(np.array([order.size for order in orders]) > 0, BULL_COLOR, BEAR_COLOR),
        marker_shape=np.where(np.array([order.size for order in orders]) > 0, 'triangle', 'inverted_triangle'),
    ))
    size = np.abs(order_source.data['size'])
    size = np.interp(size, (size.min(), size.max()), (8, 16))
    order_source.add(size, 'marker_size')
    fig3.scatter('index', 'fill_price', source=order_source, 
                 color='color_mapper', size='marker_size', marker='marker_shape',
                 line_color='black', line_width=0.5)
    # fig3.scatter(x=[datatime.get_loc(order.datetime) for order in orders if order.type == 'BUY'],
    #              y=[order.price for order in orders if order.type == 'BUY'],
    #              color='green', size=5, legend_label='Buy')
    # 绘制买点和卖点
    return fig3


def plot_with_bokeh(broker: Broker):
    """
    根据broker数据绘制回测结果图，包括：
    1. 权益曲线（含最大回撤、最大回撤持续时间）
    2. 盈亏标记图
    3. 价格主图（含交易进出场标记）
    4. 成交量图
    5. RSI指标图（示例）

    参数：
    broker: 已包含回测结果数据的对象
    """
    #=========================================================
    # 数据准备
    #=========================================================
    # 假设 broker.account_value_history 是一个 pd.Series，索引为 DatetimeIndex
    
    # 假设 trade_history 包含交易记录：EntryTime, ExitTime, EntryPrice, ExitPrice, Size, ReturnPct


    # 布局
    # 上部：Equity, PL
    # 中部：Price
    # 下部：Volume, RSI
    fig1 = _plot_account_value(broker)
    fig2 = _plot_trades(broker, fig1.x_range)
    fig3 = _plot_ohlc(broker, fig1.x_range)

    for f in [fig1, fig2, fig3]:
        if f.legend:
            f.legend.location = 'top_left'
            f.legend.border_line_width = 1
            f.legend.border_line_color = '#333333'
            f.legend.padding = 5
            f.legend.spacing = 0
            f.legend.margin = 5
            f.legend.label_text_font_size = '7pt'
            f.legend.click_policy = "hide"
            f.legend.title_text_font_size = "12px"
            f.legend.background_fill_alpha = 0.6
            # f.legend.border_line_color = None
            f.legend.label_height = 12
            f.legend.glyph_height = 12
        f.min_border_left = 0
        f.min_border_top = 3
        f.min_border_bottom = 6
        f.min_border_right = 10
        f.outline_line_color = '#666666'
        # 可选：设置网格线样式
        f.xgrid.grid_line_dash = "dotted"  # 虚线样式
        f.ygrid.grid_line_dash = "dotted"   # 实线样式

    grid = gridplot([
        fig1,
        fig2,
        fig3
        # [fig_price],
        # [fig_vol],
        # [fig_rsi]
    ], 
    ncols=1,
    sizing_mode='stretch_width',
    merge_tools=True,
    toolbar_options=dict(logo=None),
    toolbar_location='right')

    show(grid)

