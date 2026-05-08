"""Tests for the high-level Contract / contracts= API and qtrade.data helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from qtrade.backtest import Backtest, Strategy
from qtrade.contracts import (
    Contract,
    ES_CME,
    GC_COMEX,
    STOCK_CASH,
    STOCK_REGT,
)
from qtrade.core import NoCommission


def _ohlc(prices, dates):
    return pd.DataFrame(
        {'Open': prices, 'High': [p + 1 for p in prices],
         'Low': [p - 1 for p in prices], 'Close': prices},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Contract dataclass
# ---------------------------------------------------------------------------


def test_contract_defaults_match_stock_cash():
    """Default Contract() has the same numeric profile as STOCK_CASH."""
    c = Contract()
    assert (c.multiplier, c.margin_ratio) == (STOCK_CASH.multiplier, STOCK_CASH.margin_ratio)


def test_contract_is_frozen():
    """Should be immutable so users can share built-ins safely."""
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        STOCK_CASH.multiplier = 999  # type: ignore[misc]


def test_builtin_contracts_match_known_specs():
    assert STOCK_CASH.multiplier == 1 and STOCK_CASH.margin_ratio == 1.0
    assert STOCK_REGT.multiplier == 1 and STOCK_REGT.margin_ratio == 0.5
    assert ES_CME.multiplier == 50 and ES_CME.margin_ratio == 0.05
    assert GC_COMEX.multiplier == 100 and GC_COMEX.margin_ratio == 0.05


def test_user_can_define_custom_contract():
    """Users aren't required to use the built-in catalog."""
    AU_SHFE = Contract(multiplier=1000, margin_ratio=0.08, name="SHFE Gold")
    assert AU_SHFE.multiplier == 1000
    assert AU_SHFE.margin_ratio == 0.08
    assert "SHFE Gold" in repr(AU_SHFE)


# ---------------------------------------------------------------------------
# Backtest contracts= API: defaults
# ---------------------------------------------------------------------------


@pytest.fixture
def stock_data():
    return pd.DataFrame(
        {'Open': [100, 102, 104], 'High': [101, 103, 105],
         'Low': [99, 101, 103], 'Close': [100, 102, 104]},
        index=pd.date_range('2024-01-01', periods=3, freq='D'),
    )


class _BuyOnce(Strategy):
    """Buys 1 unit of every asset on the first bar where it's flat."""
    def prepare(self): pass
    def on_bar_close(self):
        for asset in self.assets:
            if self.positions[asset].size == 0:
                self.buy(asset, size=1)


def test_no_contracts_arg_defaults_to_stock_cash(stock_data):
    """Plain stock backtest — no contracts kwarg, no margin_ratio kwarg."""
    bt = Backtest(stock_data, _BuyOnce, cash=10_000,
                  commission=NoCommission(), trade_on_close=True)
    bt.run()
    # Multiplier 1, margin 1.0 — same numbers as before this PR.
    assert bt.broker.multiplier_by_asset == {"default": 1.0}
    assert bt.broker.margin_ratio_by_asset == {"default": 1.0}


def test_contracts_single_contract_applies_to_all(stock_data):
    """Pass a single Contract — applies to every asset."""
    bt = Backtest(stock_data, _BuyOnce, cash=10_000, commission=NoCommission(),
                  trade_on_close=True, contracts=STOCK_REGT)
    assert bt.broker.margin_ratio_by_asset == {"default": 0.5}


# ---------------------------------------------------------------------------
# Multi-asset
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_data():
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    return {
        "AAPL": _ohlc([180, 181, 182], dates),
        "GC=F": _ohlc([2000, 2001, 2002], dates),
        "ES=F": _ohlc([5000, 5010, 5020], dates),
    }


def test_partial_contracts_dict_fills_missing_with_stock_cash(mixed_data):
    """Only specifying futures should leave AAPL on the STOCK_CASH default."""
    bt = Backtest(mixed_data, _BuyOnce, cash=200_000, commission=NoCommission(),
                  trade_on_close=True,
                  contracts={"GC=F": GC_COMEX, "ES=F": ES_CME})
    assert bt.broker.multiplier_by_asset["AAPL"] == 1.0
    assert bt.broker.margin_ratio_by_asset["AAPL"] == 1.0  # STOCK_CASH default
    assert bt.broker.multiplier_by_asset["GC=F"] == 100
    assert bt.broker.margin_ratio_by_asset["GC=F"] == 0.05
    assert bt.broker.multiplier_by_asset["ES=F"] == 50


def test_contracts_dict_with_extra_keys_raises(mixed_data):
    """Catching typos in asset symbols — silently dropping them produces
    wrong-but-plausible backtests."""
    with pytest.raises(ValueError, match=r"contracts dict has keys not in data"):
        Backtest(mixed_data, _BuyOnce, cash=100_000, commission=NoCommission(),
                 trade_on_close=True,
                 contracts={'AAPL': STOCK_CASH, 'GC': GC_COMEX})  # typo: GC vs GC=F


def test_contracts_dict_for_single_asset_must_use_default_key(stock_data):
    """Single-asset data is keyed as 'default' internally; using a real ticker
    name in the contracts dict is a typo, not a feature."""
    with pytest.raises(ValueError, match=r"contracts dict has keys not in data"):
        Backtest(stock_data, _BuyOnce, cash=10_000, commission=NoCommission(),
                 trade_on_close=True,
                 contracts={'AAPL': STOCK_CASH})  # should be contracts=STOCK_CASH


def test_contracts_with_explicit_margin_raises(mixed_data):
    """Mixing the high-level and low-level APIs is rejected."""
    with pytest.raises(ValueError, match=r"either `contracts=` OR"):
        Backtest(mixed_data, _BuyOnce, cash=100_000, commission=NoCommission(),
                 trade_on_close=True,
                 contracts={"AAPL": STOCK_CASH},
                 margin_ratio=0.5)


def test_contracts_with_explicit_multiplier_raises(mixed_data):
    with pytest.raises(ValueError, match=r"either `contracts=` OR"):
        Backtest(mixed_data, _BuyOnce, cash=100_000, commission=NoCommission(),
                 trade_on_close=True,
                 contracts={"AAPL": STOCK_CASH},
                 contract_multiplier={"AAPL": 1, "GC=F": 100, "ES=F": 50})


def test_contracts_pnl_matches_low_level_api(mixed_data):
    """contracts= and the low-level API should produce identical results."""
    bt_high = Backtest(mixed_data, _BuyOnce, cash=200_000, commission=NoCommission(),
                       trade_on_close=True,
                       contracts={
                           "AAPL": STOCK_CASH,
                           "GC=F": GC_COMEX,
                           "ES=F": ES_CME,
                       })
    bt_high.run()

    bt_low = Backtest(mixed_data, _BuyOnce, cash=200_000, commission=NoCommission(),
                      trade_on_close=True,
                      margin_ratio={"AAPL": 1.0, "GC=F": 0.05, "ES=F": 0.05},
                      contract_multiplier={"AAPL": 1, "GC=F": 100, "ES=F": 50})
    bt_low.run()

    high_profits = sorted(t.profit for t in bt_high.broker.closed_trades)
    low_profits = sorted(t.profit for t in bt_low.broker.closed_trades)
    assert high_profits == pytest.approx(low_profits)


# ---------------------------------------------------------------------------
# qtrade.data
# ---------------------------------------------------------------------------


def test_align_indexes_intersects_indexes():
    from qtrade.data import align_indexes
    dates_a = pd.date_range('2024-01-01', periods=5, freq='D')
    dates_b = pd.date_range('2024-01-03', periods=5, freq='D')
    data = {
        "A": pd.DataFrame({'Close': range(5)}, index=dates_a),
        "B": pd.DataFrame({'Close': range(5)}, index=dates_b),
    }
    aligned = align_indexes(data)
    assert len(aligned["A"]) == len(aligned["B"]) == 3   # overlap is 3 days
    assert aligned["A"].index.equals(aligned["B"].index)


def test_align_indexes_empty_dict():
    from qtrade.data import align_indexes
    assert align_indexes({}) == {}


def test_from_yfinance_raises_clear_error_when_yfinance_missing(monkeypatch):
    """When yfinance is not importable, surface a friendly error."""
    import qtrade.data
    import sys

    # Force the import inside from_yfinance to fail.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"qtrade-lib\[data\]"):
        qtrade.data.from_yfinance(["AAPL"])
