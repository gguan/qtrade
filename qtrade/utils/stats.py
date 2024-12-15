# src/qtrade/utils/stats.py

import numpy as np
from scipy.stats import gmean
from qtrade.core import Broker, Trade
from datetime import timedelta

def __calculate_basic_metrics(metrics: dict, broker: Broker) -> dict:
    data = broker.data
    metrics['Start'] = data.index[0]
    metrics['End'] = broker.current_time
    metrics['Duration'] = broker.current_time - data.index[0]
    metrics['Start Value'] = broker.equity_history[0]
    metrics['End Value'] = broker.equity_history.loc[broker.current_time]

def __calculate_return_metrics(metrics: dict, broker: Broker) -> None:
    start_value = metrics['Start Value']
    end_value = metrics['End Value']
    total_return = (end_value - start_value) / start_value * 100
    metrics['Total Return [%]'] = total_return

    # Buy & Hold Return
    buy_hold_return = (broker.data['close'][broker.current_time] - broker.data['close'][0]) / broker.data['close'][0] * 100
    metrics['Buy & Hold Return [%]'] = buy_hold_return

    # Annualized Return
    day_returns = broker.equity_history.loc[:broker.current_time].resample('D').last().pct_change().dropna()
    gmean_day_return = gmean(1 + day_returns) - 1
    # For assets tradable on weekends (e.g., crypto), we assume 365 trading days; for stocks, 252 days
    annual_trading_days = float(
        365 if broker.equity_history.loc[:broker.current_time].index.dayofweek.to_series().between(5, 6).mean() > 2/7 * 0.6 else
        252)
    annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
    metrics['Return (Ann.) [%]'] = annualized_return * 100

    # Volatility (Annualized)
    volatility = day_returns.std() * np.sqrt(annual_trading_days) * 100
    metrics['Volatility (Ann.) [%]'] = round(volatility, 2)

def __calculate_risk_metrics(metrics: dict, broker: Broker) -> None:
    # Calculate cumulative max
    cumulative_max = broker.equity_history.cummax()
    drawdowns = (broker.equity_history - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()
    metrics['Max Drawdown [%]'] = max_drawdown * 100

    # Mark drawdown periods
    drawdown_flag = drawdowns < 0
    drawdown_periods = drawdown_flag.ne(drawdown_flag.shift()).cumsum()
    drawdown_periods = drawdown_periods[drawdown_flag]

    # Calculate duration of each drawdown period
    drawdown_durations = drawdown_periods.groupby(drawdown_periods).apply(lambda x: x.index[-1] - x.index[0])
    # Find the longest drawdown duration
    if not drawdown_durations.empty:
        max_dd_duration = drawdown_durations.max()
    else:
        max_dd_duration = np.nan
    metrics['Max Drawdown Duration'] = max_dd_duration

def __calculate_trade_metrics(metrics: dict, broker: Broker) -> None:
    trades: list[Trade] = broker.closed_trades
    total_trades = len(trades)
    wins = [t.profit for t in trades if t.profit > 0]
    losses = [t.profit for t in trades if t.profit <= 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    best_trade = max([t.profit for t in trades], default=0)
    worst_trade = min([t.profit for t in trades], default=0)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    avg_win_duration = (sum([t.exit_date - t.entry_date for t in trades if t.profit > 0], timedelta()) / len([t for t in trades if t.profit > 0])) if wins else timedelta()
    avg_loss_duration = (sum([t.exit_date - t.entry_date for t in trades if t.profit <= 0], timedelta()) / len([t for t in trades if t.profit <= 0])) if losses else timedelta()

    metrics['Total Trades'] = total_trades
    metrics['Win Rate [%]'] = win_rate
    metrics['Best Trade [%]'] = best_trade
    metrics['Worst Trade [%]'] = worst_trade
    metrics['Avg Winning Trade [%]'] = avg_win
    metrics['Avg Losing Trade [%]'] = avg_loss
    metrics['Avg Winning Trade Duration'] = avg_win_duration
    metrics['Avg Losing Trade Duration'] = avg_loss_duration

def __calculate_performance_ratios(metrics: dict, broker: Broker) -> None:
    # Profit Factor
    total_profit = sum([t.profit for t in broker.closed_trades if t.profit > 0])
    total_loss = sum([abs(t.profit) for t in broker.closed_trades if t.profit <= 0])
    profit_factor = total_profit / total_loss if total_loss > 0 else np.nan
    metrics['Profit Factor'] = profit_factor if not np.isnan(profit_factor) else np.nan

    # Expectancy
    expectancy = (total_profit - total_loss) / metrics['Total Trades'] if metrics['Total Trades'] > 0 else np.nan
    metrics['Expectancy'] = expectancy if not np.isnan(expectancy) else np.nan

    # Sharpe Ratio
    daily_returns = broker.equity_history.loc[:broker.current_time].resample('D').last().pct_change().dropna()
    annual_trading_days = float(
        365 if broker.equity_history.loc[:broker.current_time].index.dayofweek.to_series().between(5, 6).mean() > 2/7 * 0.6 else
        252)
    risk_free_rate = 0.0  # Assume risk-free rate is 0; adjust as needed
    if daily_returns.std() != 0:
        sharpe_ratio_value = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(annual_trading_days)
    else:
        sharpe_ratio_value = np.nan
    metrics['Sharpe Ratio'] = sharpe_ratio_value if not np.isnan(sharpe_ratio_value) else np.nan

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if downside_returns.std() != 0:
        sortino_ratio_value = (daily_returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(annual_trading_days)
    else:
        sortino_ratio_value = np.nan
    metrics['Sortino Ratio'] = sortino_ratio_value if not np.isnan(sortino_ratio_value) else np.nan

    # Calmar Ratio
    annualized_return = metrics.get('Return (Ann.) [%]', np.nan)
    max_drawdown = metrics.get('Max Drawdown [%]', np.nan)
    if not np.isnan(annualized_return) and max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = np.nan
    metrics['Calmar Ratio'] = calmar_ratio if not np.isnan(calmar_ratio) else np.nan

    # Omega Ratio
    threshold = 0.0  # Assume threshold is 0
    gains = daily_returns[daily_returns > threshold].sum()
    losses = abs(daily_returns[daily_returns < threshold].sum())
    omega_ratio = gains / losses if losses > 0 else np.nan
    metrics['Omega Ratio'] = omega_ratio if not np.isnan(omega_ratio) else np.nan

def calculate_stats(broker: Broker) -> dict:
    metrics = {}
    __calculate_basic_metrics(metrics, broker)
    __calculate_return_metrics(metrics, broker)
    __calculate_risk_metrics(metrics, broker)
    __calculate_trade_metrics(metrics, broker)
    __calculate_performance_ratios(metrics, broker)
    return metrics

def display_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        print(f"{key:30}: {value}")