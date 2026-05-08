"""Tests for contract_multiplier and per-asset margin_ratio."""

import pandas as pd
import pytest

from qtrade.backtest import Backtest, Strategy
from qtrade.core import Broker, NoCommission, Order


def _ohlc(prices, dates):
    return pd.DataFrame(
        {'Open': prices, 'High': [p + 1 for p in prices],
         'Low': [p - 1 for p in prices], 'Close': prices},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Trade-level: multiplier scales profit
# ---------------------------------------------------------------------------


def test_trade_profit_scales_with_multiplier():
    """1 lot of GC (multiplier=100) moving from 2000 → 2001 → $100 profit."""
    from qtrade.core import Trade
    t = Trade(
        entry_price=2000.0,
        entry_date=pd.Timestamp('2024-01-01'),
        entry_index=0,
        size=1,
        multiplier=100,
    )
    closed = t.close(
        size=None,
        exit_price=2001.0,
        exit_date=pd.Timestamp('2024-01-02'),
        exit_index=1,
        exit_reason='signal',
    )
    assert closed.profit == 100.0  # 1 * 100 * (2001 - 2000)
    assert closed.multiplier == 100


def test_trade_default_multiplier_is_one():
    """Stocks (default) — profit equals raw price-difference math."""
    from qtrade.core import Trade
    t = Trade(entry_price=100.0, entry_date=pd.Timestamp('2024-01-01'),
              entry_index=0, size=10)
    closed = t.close(None, 110.0, pd.Timestamp('2024-01-02'), 1, 'signal')
    assert closed.profit == 100.0  # 10 * 1 * 10
    assert closed.multiplier == 1.0


# ---------------------------------------------------------------------------
# Broker single-asset: futures behave like a stock × multiplier
# ---------------------------------------------------------------------------


@pytest.fixture
def gc_data():
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return _ohlc([2000, 2001, 2002, 2003, 2004], dates)


def test_single_asset_futures_pnl(gc_data):
    """Buy 1 GC contract @ 2000, hold to 2004 close → $400 profit (4 * 100)."""
    broker = Broker(gc_data, cash=20_000, commission=NoCommission(),
                    margin_ratio=0.05, trade_on_close=True,
                    contract_multiplier=100)
    broker.process_bar(gc_data.index[0])
    broker.place_orders(Order(size=1))  # 1 contract
    for ts in gc_data.index[1:]:
        broker.process_bar(ts)
    broker.close_all_positions()

    closed = broker.closed_trades
    assert len(closed) == 1
    assert closed[0].profit == pytest.approx(400.0)  # 1 contract * 100 * (2004 - 2000)
    assert broker.cash == pytest.approx(20_400.0)


def test_single_asset_futures_margin(gc_data):
    """Margin for 1 GC @ 2000, ratio=0.05, multiplier=100 → 1 * 100 * 2000 * 0.05 = $10,000."""
    broker = Broker(gc_data, cash=20_000, commission=NoCommission(),
                    margin_ratio=0.05, trade_on_close=True,
                    contract_multiplier=100)
    broker.process_bar(gc_data.index[0])
    broker.place_orders(Order(size=1))

    # available_margin = equity − used_margin = 20000 − 10000 = 10000
    assert broker.available_margin == pytest.approx(10_000.0)


def test_single_asset_futures_rejects_overleveraged_order(gc_data):
    """Buying 3 contracts at 2000 needs $30k margin — only have $20k cash."""
    broker = Broker(gc_data, cash=20_000, commission=NoCommission(),
                    margin_ratio=0.05, trade_on_close=True,
                    contract_multiplier=100)
    broker.process_bar(gc_data.index[0])
    broker.place_orders(Order(size=3))  # 3 * 100 * 2000 * 0.05 = $30k > $20k

    assert len(broker.filled_orders) == 0
    assert len(broker.closed_orders) == 1
    assert broker.closed_orders[0]._close_reason == "Insufficient margin"


# ---------------------------------------------------------------------------
# Multi-asset: stocks + futures of different categories
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_portfolio_data():
    """4-asset universe: AAPL stock, ES (S&P futures, mult=50), GC (gold, mult=100), CL (crude, mult=1000)."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return {
        'AAPL': _ohlc([180, 181, 182, 183, 184], dates),
        'ES':   _ohlc([5000, 5010, 5020, 5030, 5040], dates),
        'GC':   _ohlc([2000, 2001, 2002, 2003, 2004], dates),
        'CL':   _ohlc([80, 80.5, 81, 81.5, 82], dates),
    }


def test_per_asset_multiplier_dict(mixed_portfolio_data):
    broker = Broker(
        mixed_portfolio_data,
        cash=200_000,
        commission=NoCommission(),
        margin_ratio={'AAPL': 1.0, 'ES': 0.05, 'GC': 0.05, 'CL': 0.10},
        trade_on_close=True,
        contract_multiplier={'AAPL': 1, 'ES': 50, 'GC': 100, 'CL': 1000},
    )
    assert broker.multiplier_by_asset == {'AAPL': 1.0, 'ES': 50.0, 'GC': 100.0, 'CL': 1000.0}
    assert broker.margin_ratio_by_asset == {'AAPL': 1.0, 'ES': 0.05, 'GC': 0.05, 'CL': 0.10}


def test_per_asset_pnl_uses_correct_multiplier(mixed_portfolio_data):
    """1 AAPL share + 1 GC contract; AAPL ↑$4 = +$4, GC ↑$4 = +$400."""
    broker = Broker(
        mixed_portfolio_data,
        cash=200_000,
        commission=NoCommission(),
        margin_ratio={'AAPL': 1.0, 'ES': 0.05, 'GC': 0.05, 'CL': 0.10},
        trade_on_close=True,
        contract_multiplier={'AAPL': 1, 'ES': 50, 'GC': 100, 'CL': 1000},
    )
    broker.process_bar(mixed_portfolio_data['AAPL'].index[0])
    broker.place_orders(Order(size=1, asset='AAPL'))
    broker.place_orders(Order(size=1, asset='GC'))
    for ts in mixed_portfolio_data['AAPL'].index[1:]:
        broker.process_bar(ts)
    broker.close_all_positions()

    aapl_trades = broker.positions['AAPL'].closed_trades
    gc_trades = broker.positions['GC'].closed_trades
    assert aapl_trades[0].profit == pytest.approx(4.0)     # 1 * 1 * (184 - 180)
    assert gc_trades[0].profit == pytest.approx(400.0)     # 1 * 100 * (2004 - 2000)


def test_dict_validates_keys(mixed_portfolio_data):
    """Per-asset dict must have keys exactly matching the data."""
    with pytest.raises(ValueError, match=r"missing|extra"):
        Broker(
            mixed_portfolio_data,
            cash=100_000,
            commission=NoCommission(),
            margin_ratio=1.0,
            trade_on_close=True,
            contract_multiplier={'AAPL': 1, 'GC': 100},  # missing ES, CL
        )


def test_per_asset_margin_ratio(mixed_portfolio_data):
    """Stock margin = 1.0 (no leverage), futures margin = 0.05 (20x)."""
    broker = Broker(
        mixed_portfolio_data,
        cash=100_000,
        commission=NoCommission(),
        margin_ratio={'AAPL': 1.0, 'ES': 0.05, 'GC': 0.05, 'CL': 0.10},
        trade_on_close=True,
        contract_multiplier={'AAPL': 1, 'ES': 50, 'GC': 100, 'CL': 1000},
    )
    broker.process_bar(mixed_portfolio_data['AAPL'].index[0])
    broker.place_orders(Order(size=10, asset='AAPL'))     # full margin: 10 * 1 * 180 * 1.0 = $1800
    broker.place_orders(Order(size=1, asset='ES'))        # leveraged: 1 * 50 * 5000 * 0.05 = $12,500

    # Used margin = 1800 + 12500 = 14300; equity = 100000 (no PnL yet)
    # available_margin = 100000 − 14300 = 85700
    assert broker.available_margin == pytest.approx(85_700.0)


# ---------------------------------------------------------------------------
# End-to-end Backtest with a realistic mixed portfolio
# ---------------------------------------------------------------------------


class _BuyOneOfEach(Strategy):
    def prepare(self): pass

    def on_bar_close(self):
        for asset in self.assets:
            if self.positions[asset].size == 0:
                self.buy(asset, size=1)


def test_backtest_mixed_portfolio_endtoend(mixed_portfolio_data):
    bt = Backtest(
        mixed_portfolio_data,
        _BuyOneOfEach,
        cash=200_000,
        commission=NoCommission(),
        margin_ratio={'AAPL': 1.0, 'ES': 0.05, 'GC': 0.05, 'CL': 0.10},
        trade_on_close=True,
        contract_multiplier={'AAPL': 1, 'ES': 50, 'GC': 100, 'CL': 1000},
    )
    bt.run()

    by_asset = {t.asset: t for t in bt.broker.closed_trades}
    # AAPL bought day 1 close=181, exit @ end (close=184) → 1 * 1 * 3 = $3
    # ES   bought day 1 close=5010, exit @ 5040 → 1 * 50 * 30 = $1,500
    # GC   bought day 1 close=2001, exit @ 2004 → 1 * 100 * 3 = $300
    # CL   bought day 1 close=80.5, exit @ 82 → 1 * 1000 * 1.5 = $1,500
    assert by_asset['AAPL'].profit == pytest.approx(3.0)
    assert by_asset['ES'].profit == pytest.approx(1500.0)
    assert by_asset['GC'].profit == pytest.approx(300.0)
    assert by_asset['CL'].profit == pytest.approx(1500.0)


def test_default_multiplier_is_unchanged_for_stocks(gc_data):
    """No contract_multiplier kwarg → behaves exactly like before (multiplier=1)."""
    broker = Broker(gc_data, cash=20_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    broker.process_bar(gc_data.index[0])
    broker.place_orders(Order(size=10))
    broker.process_bar(gc_data.index[1])
    broker.close_all_positions()
    # Same as a stock backtest: 10 * 1 * (2001 - 2000) when sold at next bar...
    # actually entry @ 2000, exit @ 2001 → 10 * 1 = 10
    assert broker.closed_trades[0].profit == pytest.approx(10.0)
