from math import copysign
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Span, Range1d,LinearAxis, DataRange1d, CustomJS
from bokeh.layouts import gridplot
from bokeh.models import CustomJSTickFormatter, NumeralTickFormatter, DatetimeTickFormatter
from bokeh.transform import factor_cmap

from qtrade.core.broker import Broker

day_colors = {
    'equity': 'darkslateblue',
    'equity_peak': '#8B5DFF',
    'buy_and_hold': 'darkgray',
    'drawdown': 'orangered',
    'bull': 'limegreen',
    'bear': 'tomato',
}

def _plot_equity(source, equity_history, cumulative_returns, buy_and_hold_returns, x_range):
    # Create the top equity curve plot (including maximum drawdown and duration)
    fig_equity = figure(height=120, tools="xpan,xwheel_zoom,reset,save",
                        active_drag='xpan', active_scroll='xwheel_zoom', x_range=x_range)
    r = fig_equity.line('index', 'cumulative_returns', source=source, line_width=1.5, line_alpha=1, color=day_colors['equity'])
    fig_equity.line('index', 'buy_and_hold_returns', source=source, line_width=1.5, line_alpha=1, 
              color=day_colors['buy_and_hold'], line_dash="dotted")
    
    # Add a single HoverTool
    hover = HoverTool(
        tooltips=[
            ("Date", "@datetime{%Y-%m-%d %H:%M:%S}"),
            ("Equity", "@equity{0,0.00}"),
            ("Equity Returns", "@cumulative_returns{0.00%}"),
            ("Buy and Hold Returns", "@buy_and_hold_returns{0.00%}")
        ],
        formatters={
            '@datetime': 'datetime',  # Use datetime to format dates
        },
        visible=False,
        mode='vline',  # Vertical line mode
        renderers=[r]
    )
    fig_equity.add_tools(hover)

    peak_equity_index = cumulative_returns.idxmax()
    peak_equity_value = cumulative_returns[peak_equity_index]
    fig_equity.scatter(peak_equity_index, peak_equity_value, color=day_colors['equity_peak'], size=7, line_color='black', 
                       line_width=0.5, legend_label=f"Peak ({peak_equity_value*100:.0f}%)")

    final_equity_index = cumulative_returns.index[-1]
    final_equity_value = cumulative_returns[final_equity_index]
    fig_equity.scatter(final_equity_index, final_equity_value, color=day_colors['equity'], size=7, line_color='black', 
                       line_width=0.5, legend_label=f"Final ({final_equity_value*100:.0f}%)")

    final_bnh_index = buy_and_hold_returns.index[-1]
    final_bnh_value = buy_and_hold_returns[final_equity_index]
    fig_equity.scatter(final_bnh_index, final_bnh_value, color=day_colors['buy_and_hold'], size=7, 
                 line_color='black', line_width=0.5, legend_label=f"Buy and Hold ({final_bnh_value*100:.0f}%)")

    # Peak equity point
    cumulative_max = equity_history.cummax()
    drawdowns = (equity_history - cumulative_max) / cumulative_max
    
    # Identify peak points (where cumulative max updates)
    dd_max_time = drawdowns.idxmin()
    dd_max_val = cumulative_returns[dd_max_time]
    dd_max_drawdown = drawdowns.min() * 100
    fig_equity.scatter(dd_max_time, dd_max_val, color=day_colors['drawdown'], size=7, line_color='black', line_width=0.5,
                      legend_label=f"Max Drawdown ({dd_max_drawdown:.1f}%)")

    # Maximum drawdown duration (marked with red horizontal line segments on the chart)
    drawdown_flag = drawdowns < 0
    drawdown_periods = drawdown_flag.ne(drawdown_flag.shift()).cumsum()
    drawdown_periods = drawdown_periods[drawdown_flag]
    # Calculate the duration of each drawdown period
    dates = source.data['datetime']
    drawdown_stats = drawdown_periods.groupby(drawdown_periods).agg(
        start=lambda x: x.index[0]-1,
        end=lambda x: x.index[-1],
        duration=lambda x: dates[x.index[-1]] - dates[x.index[0]-1]
    )
    
    # Find the longest drawdown duration
    if not drawdown_stats.empty:
        longest_dd = drawdown_stats.loc[drawdown_stats['duration'].idxmax()]
        max_dd_duration = longest_dd['duration']
        max_dd_start = longest_dd['start']
        max_dd_end = longest_dd['end']
        max_dd_value= cumulative_returns[max_dd_start]
        fig_equity.line([max_dd_start, max_dd_end], max_dd_value,
                    color=day_colors['bear'], line_width=1.5, legend_label=f"Max Dd Dur. ({max_dd_duration})")
        fig_equity.varea(x=cumulative_returns.index, y1=cumulative_returns, y2=cumulative_returns.cummax(), 
                         color=day_colors['bear'], alpha=0.2)

    # if relative_account_value:
    fig_equity.yaxis.formatter = NumeralTickFormatter(format='0,0.[00]%')
    fig_equity.legend.title = 'Equity'
    fig_equity.xaxis.visible = False

    return fig_equity

def _plot_trade(trades, datetime, x_range):
    # Process trade records: mark entry and exit points on the price chart, using different colors for profit and loss
    trade_source = ColumnDataSource(dict(
        index=np.array([trade.exit_index if trade.exit_index else np.nan for trade in trades]),
        datetime=np.array([trade.exit_date for trade in trades]),
        exit_price=np.array([trade.exit_price for trade in trades]),
        size=np.array([trade.size for trade in trades]),
        return_pct=np.array([copysign(1, trade.size) * (trade.exit_price / trade.entry_price - 1) for trade in trades]),
        profit=np.array([trade.profit for trade in trades]),
    ))
    size = np.abs(trade_source.data['size'])
    if len(size) > 0:
        size = np.interp(size, (size.min(), size.max()), (5, 10))
    trade_source.add(size, 'marker_size')
   
    returns_long = np.where(trade_source.data['size'] > 0, trade_source.data['return_pct'], np.nan)
    returns_short = np.where(trade_source.data['size'] < 0, trade_source.data['return_pct'], np.nan)
    trade_source.add(returns_long, 'returns_long')
    trade_source.add(returns_short, 'returns_short')
    
    fig_trade = figure(height=100, tools="xpan,xwheel_zoom,reset,save",
                        active_drag='xpan', active_scroll='xwheel_zoom',
                        x_range=x_range)
    fig_trade.add_layout(Span(location=0, dimension='width', line_color='#666666',
                            line_dash='dashed', line_width=1))
    
    r1 = fig_trade.scatter('index', 'returns_long', source=trade_source, color=day_colors['bull'], marker='triangle',
                           size='marker_size', line_color='black', line_width=0.5, legend_label='Long')
    r2 = fig_trade.scatter('index', 'returns_short', source=trade_source, color=day_colors['bear'], marker='inverted_triangle',
                           size='marker_size', line_color='black', line_width=0.5, legend_label='Short')
    
    tooltips = [
        ("Date", "@datetime{%Y-%m-%d %H:%M:%S}"),
        ("Size", "@size{0,0}"),
        ("Return Pct", "@return_pct{0.00%}"),
        ("Profit", "@profit{0,0.00}")
    ]
    fig_trade.add_tools(HoverTool(
        tooltips=tooltips,
        formatters={"@datetime": "datetime"},
        visible=False,
        mode='vline',
        renderers=[r1]
    ))
    fig_trade.add_tools(HoverTool(
        tooltips=tooltips,
        formatters={"@datetime": "datetime"},
        visible=False,
        mode='vline',
        renderers=[r2]
    ))
    
    fig_trade.legend.title = f'Trades ({len(trades)}) - Total Trade Profit/Loss ({sum(trade.profit for trade in trades):,.2f})'
    fig_trade.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    fig_trade.xaxis.visible = False

    return fig_trade

def _plot_ohlc(source, datetime, trades, orders, plot_volume=True):
    # Map colors for increasing and decreasing values
    inc_values = (source.data['Close'] >= source.data['Open']).astype(np.uint8).astype(str)
    source.add(inc_values, 'inc')
    
    index = source.data['index']
    x_pad = (index[-1] - index[0]) / 20
    y_min = source.data['Close'].min()
    y_max = source.data['Close'].max()
    range_pad = (y_max - y_min) * 0.05
    fig_ohlc = figure(
        height=300, x_axis_type='linear', tools="xpan,xwheel_zoom,reset",active_drag='xpan', active_scroll='xwheel_zoom',
        x_range=Range1d(index[0], index[-1], min_interval=10, bounds=(index[0] - x_pad, index[-1] + x_pad)),
        y_range=DataRange1d(start=y_min - range_pad, end=y_max + range_pad) if plot_volume else DataRange1d(),
    )
    r = fig_ohlc.line('index', 'Close', source=source, line_width=1.5, color='black')
    fig_ohlc.yaxis[0].formatter = NumeralTickFormatter(format="0.[00]")

    if plot_volume:
        # Set additional Y axis range for volume
        fig_ohlc.extra_y_ranges = {"volume": Range1d(start=0, end=source.data['Volume'].mean() * 8)}
        fig_ohlc.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume"), 'right')
        
        # Use factor_cmap to map colors
        color_mapper = factor_cmap('inc', palette=[day_colors['bear'], day_colors['bull']], factors=['0', '1'])

        # Draw volume bars, bound to the second Y axis
        fig_ohlc.vbar(
            x='index',
            top='Volume',
            width=0.8,
            source=source,
            color=color_mapper,
            alpha=0.2,
            y_range_name="volume"
        )
        fig_ohlc.yaxis[1].formatter = NumeralTickFormatter(format="0,0")

    NBSP = '\N{NBSP}' * 2  # Four non-breaking spaces
    
    # Add a single HoverTool
    hover = HoverTool(
        tooltips=[
            ("Date", "@datetime{%Y-%m-%d %H:%M:%S}"),
            ("Price", "@Close{0,0.0[0000]}"),
            ('OHLC', f"O:@Open{{0.2f}}{NBSP}H:@High{{0.2f}}{NBSP}L:@Low{{0.2f}}{NBSP}C:@Close{{0.2f}}"),
            ("Volume", "@Volume{0,0}"),
        ],
        formatters={
            '@datetime': 'datetime',  # Use datetime to format dates
        },
        visible=False,
        mode='vline',  # Vertical line mode
        attachment='right',
        renderers=[r]
    )
    fig_ohlc.add_tools(hover)

    # Format x-axis
    fig_ohlc.xaxis.formatter = CustomJSTickFormatter(
        args=dict(
            axis=fig_ohlc.xaxis[0], 
            formatter=DatetimeTickFormatter(hours='%H:%M', days='%a, %d %b', months='%m/%Y'),
            source=source),
            code='''
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
        '''
        )

    win_trades = [trade for trade in trades if trade.profit > 0]
    lose_trades = [trade for trade in trades if trade.profit <= 0]

    win_trade_source = ColumnDataSource(dict(
        top=np.array([trade.exit_price for trade in win_trades]),
        bottom=np.array([trade.entry_price for trade in win_trades]),
        left=np.array([
            trade.entry_index if trade.entry_index else np.nan 
            for trade in win_trades 
        ]),
        right=np.array([
            trade.exit_index if trade.exit_index else np.nan 
            for trade in win_trades 
        ]),
    ))
    lose_trade_source = ColumnDataSource(dict(
        top=np.array([trade.exit_price for trade in lose_trades]),
        bottom=np.array([trade.entry_price for trade in lose_trades]),
        left=np.array([
            trade.entry_index if trade.entry_index else np.nan 
            for trade in lose_trades 
        ]),
        right=np.array([
            trade.exit_index if trade.exit_index else np.nan 
            for trade in lose_trades 
        ]),
    ))

    fig_ohlc.quad(left='left', right='right', top='top', bottom='bottom', source=win_trade_source, color=day_colors['bull'], 
                  alpha=0.2, legend_label=f'Win Trades ({len(win_trades)})')
    fig_ohlc.quad(left='left', right='right', top='top', bottom='bottom', source=lose_trade_source, color=day_colors['bear'], 
                  alpha=0.2, legend_label=f'Lose Trades ({len(lose_trades)})')

    # Draw buy and sell points
    order_source = ColumnDataSource(dict(
        index=np.array([datetime.get_loc(order._fill_date) for order in orders ]),
        size=np.array([abs(order.size) for order in orders]),
        datetime=np.array([order._fill_date for order in orders]),
        fill_price=np.array([order._fill_price for order in orders]),
        color_mapper=np.where(np.array([order.size for order in orders]) > 0, day_colors['bull'], day_colors['bear']),
        marker_shape=np.where(np.array([order.size for order in orders]) > 0, 'triangle', 'inverted_triangle'),
    ))
    size = np.abs(order_source.data['size'])
    if len(size) > 0:
        size = np.interp(size, (size.min(), size.max()), (6, 12))
    order_source.add(size, 'marker_size')
    r2 = fig_ohlc.scatter('index', 'fill_price', source=order_source, 
                 color='color_mapper', size='marker_size', marker='marker_shape',
                 line_color='black', line_width=0.5)
    order_tooltips = [
        ("Date", "@datetime{%Y-%m-%d %H:%M:%S}"),
        ("Size", "@size{0,0}"),
        ("Fill Price", "@fill_price{0,0.00}")
    ]
    fig_ohlc.add_tools(HoverTool(tooltips=order_tooltips, formatters={"@datetime": "datetime"}, visible=False, mode='vline',
                                 attachment='left', renderers=[r2]))

    fig_ohlc.legend.title = f'{datetime[0]} - {datetime[-1]} ({datetime[-1]-datetime[0]})'
    return fig_ohlc

def _build_asset_source(df: pd.DataFrame, current_time, equity_history, cumulative_returns, buy_and_hold_returns):
    """Build the OHLC ColumnDataSource for one asset, also packing the
    portfolio-level equity / cumulative-return / B&H series so the autoscale
    JS callback (single-asset mode) can read everything off one source."""
    asset_data = df.loc[:current_time].copy(deep=True)
    datetime = asset_data.index.copy(deep=True)
    source = ColumnDataSource(asset_data)
    source.data['index'] = np.arange(len(asset_data))
    source.data['datetime'] = datetime
    source.data['equity'] = equity_history.values
    source.data['cumulative_returns'] = cumulative_returns.values
    source.data['buy_and_hold_returns'] = buy_and_hold_returns.values
    return source, datetime


def plot_with_bokeh(broker: Broker, filename=None):
    """Render a backtest report.

    For single-asset brokers: equity panel, trade-returns scatter, and one
    OHLC panel (with cross-figure autoscale on zoom).

    For multi-asset: portfolio equity vs equal-weighted B&H, a portfolio-wide
    trade-returns scatter, and one OHLC panel per asset stacked vertically.
    """
    if len(broker.data_by_asset) > 1:
        return _plot_multi_asset(broker, filename)
    return _plot_single_asset(broker, filename)


def _plot_single_asset(broker: Broker, filename=None):
    only_asset = next(iter(broker.data_by_asset))
    df = broker.data_by_asset[only_asset]
    plot_volume = 'Volume' in df.columns

    equity_history = broker.equity_history.loc[:broker.current_time].copy(deep=True)
    equity_history.reset_index(drop=True, inplace=True)

    buy_and_hold_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
    buy_and_hold_history = df[buy_and_hold_col].loc[:broker.current_time].copy(deep=True)
    buy_and_hold_history.reset_index(drop=True, inplace=True)

    cumulative_returns = equity_history / equity_history.iloc[0]
    buy_and_hold_returns = buy_and_hold_history / buy_and_hold_history.iloc[0]

    source, datetime = _build_asset_source(
        df, broker.current_time, equity_history, cumulative_returns, buy_and_hold_returns,
    )

    fig_ohlc = _plot_ohlc(source, datetime, broker.closed_trades, broker.filled_orders, plot_volume=plot_volume)
    fig_equity = _plot_equity(source, equity_history, cumulative_returns, buy_and_hold_returns, fig_ohlc.x_range)
    fig_trade = _plot_trade(broker.closed_trades, datetime, fig_ohlc.x_range)
    
    # Create CustomJS callback, pass necessary arguments
    callback_args = dict(
        source=source,
        equity_y_range=fig_equity.y_range,
        ohlc_y_range=fig_ohlc.y_range
    )
    if plot_volume:
        callback_args.update(volume_y_range=fig_ohlc.extra_y_ranges['volume'])

    callback_code = """
    if (!window._qt_scale_range) {
        window._qt_scale_range = function (range, min, max, pad) {
            "use strict";
            if (min !== Infinity && max !== -Infinity) {
                pad = pad ? (max - min) * 0.03 : 0;
                range.start = min - pad;
                range.end = max + pad;
            } else {
                console.error('qtrade: scale range error:', min, max, range);
            }
        };
    }

    clearTimeout(window._qt_autoscale_timeout);

    window._qt_autoscale_timeout = setTimeout(function () {
        "use strict";

        // Get data
        const index = source.data['index'];
        const cumulative_return = source.data['cumulative_returns'];
        const buy_and_hold_returns = source.data['buy_and_hold_returns'];
        const high = source.data['High'];
        const low = source.data['Low'];
        const volume = source.data['Volume'];
        
        // Get the current x_range start and end values
        const x_start = cb_obj.start;
        const x_end = cb_obj.end;
        
        // Global data range
        const total_start = index[0];
        const total_end = index[index.length - 1];
        
        // Determine if the full view is displayed (i.e., x_range covers all data)
        const is_full_view = (x_start <= total_start) && (x_end >= total_end);
        
        if (is_full_view) {
            // If it is the full view, reset y_range to auto range (default behavior of DataRange1d)
            equity_y_range.reset();
            ohlc_y_range.reset();
            if (typeof volume_y_range !== 'undefined') volume_y_range.reset();
        } else {
            // Calculate the index within the current visible range
            let i = Math.floor(x_start);
            let j = Math.ceil(x_end);
            i = Math.max(i, 0);
            j = Math.min(j, index.length - 1);
            
            // Min and max of buy_and_hold_returns and cumulative_return
            let min1 = Infinity;
            let max1 = -Infinity;
            for (let k = i; k <= j; k++) {
                if (buy_and_hold_returns[k] < min1) { min1 = buy_and_hold_returns[k]; }
                if (cumulative_return[k] < min1) { min1 = cumulative_return[k]; }
                if (buy_and_hold_returns[k] > max1) { max1 = buy_and_hold_returns[k]; }
                if (cumulative_return[k] > max1) { max1 = cumulative_return[k]; }
            }
            window._qt_scale_range(equity_y_range, min1, max1, true);
            
            // Min and max of open, high, low, close
            let min2 = Infinity;
            let max2 = -Infinity;
            for (let k = i; k <= j; k++) {
                if (low[k] < min2) { min2 = low[k]; }
                if (high[k] > max2) { max2 = high[k]; }
            }
            window._qt_scale_range(ohlc_y_range, min2, max2, true);
            
            // Average volume * 8 (only when volume axis is present)
            if (typeof volume_y_range !== 'undefined' && volume) {
                let max3 = 0;
                for (let k = i; k <= j; k++) {
                    max3 += volume[k];
                }
                max3 = j - i > 0 ? max3 / (j - i) * 8 : 0;
                window._qt_scale_range(volume_y_range, 0, max3, false);
            }
        }
        
    }, 100);
    """
    
    callback = CustomJS(args=callback_args, code=callback_code)
    fig_ohlc.x_range.js_on_change('start', callback)
    fig_ohlc.x_range.js_on_change('end', callback)

    figs = [fig_equity, fig_trade, fig_ohlc]
    _apply_styling(figs)
    _render(figs, filename)


def _plot_multi_asset(broker: Broker, filename=None):
    """Multi-asset report: portfolio equity + per-asset OHLC panels stacked."""
    equity_history = broker.equity_history.loc[:broker.current_time].copy(deep=True)
    equity_history.reset_index(drop=True, inplace=True)
    cumulative_returns = equity_history / equity_history.iloc[0]

    # Equal-weighted Buy & Hold across all assets.
    bnh_components = []
    for df in broker.data_by_asset.values():
        col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        h = df[col].loc[:broker.current_time].copy(deep=True)
        h.reset_index(drop=True, inplace=True)
        bnh_components.append(h / h.iloc[0])
    buy_and_hold_returns = pd.concat(bnh_components, axis=1).mean(axis=1)

    # First asset drives the shared x_range and equity-panel source.
    first_asset = next(iter(broker.data_by_asset))
    first_df = broker.data_by_asset[first_asset]
    first_source, first_datetime = _build_asset_source(
        first_df, broker.current_time, equity_history, cumulative_returns, buy_and_hold_returns,
    )
    plot_volume = 'Volume' in first_df.columns
    first_trades = [t for t in broker.closed_trades if t.asset == first_asset]
    first_orders = [o for o in broker.filled_orders if o.asset == first_asset]
    first_ohlc = _plot_ohlc(first_source, first_datetime, first_trades, first_orders, plot_volume=plot_volume)
    first_ohlc.title = f"OHLC — {first_asset}"  # type: ignore[assignment]

    fig_equity = _plot_equity(first_source, equity_history, cumulative_returns, buy_and_hold_returns, first_ohlc.x_range)
    fig_equity.title = "Portfolio equity vs equal-weighted Buy & Hold"  # type: ignore[assignment]
    fig_trade = _plot_trade(broker.closed_trades, first_datetime, first_ohlc.x_range)

    asset_figs = [first_ohlc]
    for asset, df in broker.data_by_asset.items():
        if asset == first_asset:
            continue
        source, datetime = _build_asset_source(
            df, broker.current_time, equity_history, cumulative_returns, buy_and_hold_returns,
        )
        plot_volume_i = 'Volume' in df.columns
        a_trades = [t for t in broker.closed_trades if t.asset == asset]
        a_orders = [o for o in broker.filled_orders if o.asset == asset]
        f_ohlc = _plot_ohlc(source, datetime, a_trades, a_orders, plot_volume=plot_volume_i)
        f_ohlc.x_range = first_ohlc.x_range  # link x-axes
        f_ohlc.title = f"OHLC — {asset}"  # type: ignore[assignment]
        asset_figs.append(f_ohlc)

    figs = [fig_equity, fig_trade, *asset_figs]
    _apply_styling(figs)
    _render(figs, filename)


def _apply_styling(figs):
    for f in figs:
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
            f.legend.label_height = 12
            f.legend.glyph_height = 12
        f.min_border_left = 0
        f.min_border_top = 3
        f.min_border_bottom = 6
        f.min_border_right = 10
        f.outline_line_color = '#666666'
        f.xgrid.grid_line_dash = "dotted"
        f.ygrid.grid_line_dash = "dotted"


def _render(figs, filename):
    grid = gridplot(
        figs,
        ncols=1,
        sizing_mode='stretch_width',
        merge_tools=True,
        toolbar_options=dict(logo=None),
        toolbar_location='right',
    )
    show(grid)
    if filename:
        from bokeh.io import output_file, save
        output_file(filename)
        save(grid)
