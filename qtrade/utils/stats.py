# src/qtrade/utils/stats.py

from datetime import timedelta

import numpy as np
from scipy.stats import gmean

from qtrade.core import Broker, Trade


def __calculate_basic_metrics(metrics: dict, broker: Broker) -> None:
    eq = broker.equity_history
    metrics['Start'] = eq.index[0]
    metrics['End'] = broker.current_time
    metrics['Duration'] = broker.current_time - eq.index[0]
    metrics['Start Value'] = eq.iloc[0]
    metrics['End Value'] = eq.loc[broker.current_time]


def _buy_and_hold_return(broker: Broker) -> float:
    """Equal-weighted buy & hold return across all assets, as a percentage."""
    returns = []
    for df in broker.data_by_asset.values():
        col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        first = df[col].iloc[0]
        last = df[col].loc[broker.current_time]
        returns.append((last - first) / first * 100)
    return float(np.mean(returns)) if returns else 0.0


def __calculate_return_metrics(metrics: dict, broker: Broker) -> None:
    start_value = metrics['Start Value']
    end_value = metrics['End Value']
    total_return = (end_value - start_value) / start_value * 100
    metrics['Total Return [%]'] = total_return

    total_commissions: float = 0.0
    if broker.commission is not None:
        for o in broker.filled_orders:
            if o.fill_price is not None:
                total_commissions += broker.commission.calculate_commission(o.size, o.fill_price)
    metrics['Total Commission Cost[%]'] = total_commissions

    metrics['Buy & Hold Return [%]'] = _buy_and_hold_return(broker)

    # Annualized Return
    eq = broker.equity_history.loc[:broker.current_time]
    day_returns = eq.resample('D').last().pct_change(fill_method=None).dropna()
    gmean_day_return = gmean(1 + day_returns) - 1
    # 365-day calendar for assets that trade weekends (e.g. crypto), 252 otherwise.
    annual_trading_days = float(
        365 if eq.index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * 0.6 else 252
    )
    annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
    metrics['Return (Ann.) [%]'] = annualized_return * 100

    volatility = day_returns.std() * np.sqrt(annual_trading_days) * 100
    metrics['Volatility (Ann.) [%]'] = round(volatility, 2)


def __calculate_risk_metrics(metrics: dict, broker: Broker) -> None:
    cumulative_max = broker.equity_history.cummax()
    drawdowns = (broker.equity_history - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()
    metrics['Max Drawdown [%]'] = max_drawdown * 100

    drawdown_flag = drawdowns < 0
    drawdown_periods = drawdown_flag.ne(drawdown_flag.shift()).cumsum()
    drawdown_periods = drawdown_periods[drawdown_flag]

    drawdown_durations = drawdown_periods.groupby(drawdown_periods).apply(lambda x: x.index[-1] - x.index[0])
    metrics['Max Drawdown Duration'] = drawdown_durations.max() if not drawdown_durations.empty else np.nan


def _trade_metrics(trades: list[Trade]) -> dict:
    """Compute trade-level metrics for an arbitrary list of closed trades."""
    total_trades = len(trades)
    profits = [t.profit for t in trades if t.profit is not None]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    best_trade = max(profits, default=0)
    worst_trade = min(profits, default=0)
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    avg_win_duration = (
        sum([t.exit_date - t.entry_date for t in trades if t.profit is not None and t.profit > 0], timedelta())
        / len(wins)
    ) if wins else timedelta()
    avg_loss_duration = (
        sum([t.exit_date - t.entry_date for t in trades if t.profit is not None and t.profit <= 0], timedelta())
        / len(losses)
    ) if losses else timedelta()

    return {
        'Total Trades': total_trades,
        'Win Rate [%]': win_rate,
        'Best Trade [%]': best_trade,
        'Worst Trade [%]': worst_trade,
        'Avg Winning Trade [%]': avg_win,
        'Avg Losing Trade [%]': avg_loss,
        'Avg Winning Trade Duration': avg_win_duration,
        'Avg Losing Trade Duration': avg_loss_duration,
    }


def __calculate_trade_metrics(metrics: dict, broker: Broker) -> None:
    metrics.update(_trade_metrics(list(broker.closed_trades)))


def __calculate_performance_ratios(metrics: dict, broker: Broker) -> None:
    profits = [t.profit for t in broker.closed_trades if t.profit is not None]
    total_profit = sum(p for p in profits if p > 0)
    total_loss = sum(abs(p) for p in profits if p <= 0)

    profit_factor = total_profit / total_loss if total_loss > 0 else np.nan
    metrics['Profit Factor'] = profit_factor

    expectancy = (total_profit - total_loss) / metrics['Total Trades'] if metrics['Total Trades'] > 0 else np.nan
    metrics['Expectancy'] = expectancy

    eq = broker.equity_history.loc[:broker.current_time]
    daily_returns = eq.resample('D').last().pct_change(fill_method=None).dropna()
    annual_trading_days = float(
        365 if eq.index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * 0.6 else 252
    )
    risk_free_rate = 0.0
    if daily_returns.std() != 0:
        sharpe_ratio_value = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(annual_trading_days)
    else:
        sharpe_ratio_value = np.nan
    metrics['Sharpe Ratio'] = sharpe_ratio_value

    downside_returns = daily_returns[daily_returns < 0]
    if downside_returns.std() != 0:
        sortino_ratio_value = (daily_returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(annual_trading_days)
    else:
        sortino_ratio_value = np.nan
    metrics['Sortino Ratio'] = sortino_ratio_value

    annualized_return = metrics.get('Return (Ann.) [%]', np.nan)
    max_drawdown = metrics.get('Max Drawdown [%]', np.nan)
    if not np.isnan(annualized_return) and max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = np.nan
    metrics['Calmar Ratio'] = calmar_ratio

    threshold = 0.0
    gains = daily_returns[daily_returns > threshold].sum()
    losses_sum = abs(daily_returns[daily_returns < threshold].sum())
    omega_ratio = gains / losses_sum if losses_sum > 0 else np.nan
    metrics['Omega Ratio'] = omega_ratio


def calculate_stats(broker: Broker) -> dict:
    """Compute portfolio-level statistics for a Broker.

    For multi-asset, all aggregate metrics are portfolio-level. The
    'Buy & Hold Return [%]' uses an equal-weighted average across assets.
    For per-asset breakdowns, see :func:`calculate_stats_per_asset`.
    """
    metrics: dict = {}
    __calculate_basic_metrics(metrics, broker)
    __calculate_return_metrics(metrics, broker)
    __calculate_risk_metrics(metrics, broker)
    __calculate_trade_metrics(metrics, broker)
    __calculate_performance_ratios(metrics, broker)
    return metrics


def calculate_stats_per_asset(broker: Broker) -> dict[str, dict]:
    """Per-asset trade-level breakdown of closed trades and buy-and-hold return.

    Note: this only covers trade-level metrics + a per-asset B&H reference.
    Portfolio-level metrics (Sharpe, Drawdown, equity curve) are inherently
    portfolio-wide and only available via :func:`calculate_stats`.
    """
    result: dict[str, dict] = {}
    for asset, position in broker.positions.items():
        trades = list(position.closed_trades)
        per: dict = {'Asset': asset}
        per.update(_trade_metrics(trades))

        df = broker.data_by_asset[asset]
        col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        first = df[col].iloc[0]
        last = df[col].loc[broker.current_time]
        per['Buy & Hold Return [%]'] = (last - first) / first * 100

        result[asset] = per
    return result


def display_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict (key padded, value flush)."""
    for key, value in metrics.items():
        print(f"{key:30}: {value}")
