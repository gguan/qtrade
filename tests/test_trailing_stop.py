"""Tests for SL/TP setters, update_exit_levels, and trailing-stop orders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtrade.backtest.backtest import Backtest
from qtrade.backtest.strategy import Strategy
from qtrade.core import Order, Trade


# ---------------------------------------------------------------------------
# 1. SL/TP setters and update_exit_levels (mid-trade modification)
# ---------------------------------------------------------------------------


def _make_trade(size=10, sl=None, tp=None):
    return Trade(
        entry_price=100.0,
        entry_date=pd.Timestamp("2024-01-01"),
        entry_index=0,
        size=size,
        sl=sl,
        tp=tp,
    )


def test_sl_tp_setters_modify_open_trade():
    t = _make_trade(sl=95.0, tp=110.0)
    t.sl = 97.5
    t.tp = 115.0
    assert t.sl == 97.5
    assert t.tp == 115.0


def test_sl_tp_setters_can_clear_to_none():
    t = _make_trade(sl=95.0, tp=110.0)
    t.sl = None
    t.tp = None
    assert t.sl is None
    assert t.tp is None


def test_update_exit_levels_changes_only_what_was_passed():
    t = _make_trade(sl=95.0, tp=110.0)
    t.update_exit_levels(sl=97.0)        # tp untouched
    assert (t.sl, t.tp) == (97.0, 110.0)
    t.update_exit_levels(tp=120.0)       # sl untouched
    assert (t.sl, t.tp) == (97.0, 120.0)


def test_update_exit_levels_explicit_none_clears():
    t = _make_trade(sl=95.0, tp=110.0)
    t.update_exit_levels(sl=None)        # explicit None → clear
    assert t.sl is None
    assert t.tp == 110.0                 # still untouched


def test_modified_sl_triggers_at_new_level_in_backtest():
    """End-to-end: change SL mid-backtest, exit fires at the new level."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Bar 0: 100 (warmup, skipped). Bar 1: open at 100, then climb to 105…
    close = np.array([100, 100, 102, 104, 105, 103, 99, 96, 95, 94], dtype=float)
    df = pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": 1000,
    }, index=dates)

    class _RatchetingSL(Strategy):
        """Buy on bar 1 with SL=90, then ratchet SL up to 100 once price>=104."""

        def prepare(self):
            self._bought = False
            self._ratcheted = False

        def on_bar_close(self):
            if not self._bought:
                self.buy(size=10, sl=90.0)
                self._bought = True
                return
            if not self._ratcheted and self.data["Close"].iloc[-1] >= 104:
                # Move SL up — should now fire when low <= 100
                for trade in self.active_trades:
                    trade.update_exit_levels(sl=100.0)
                self._ratcheted = True

    bt = Backtest(df, _RatchetingSL, cash=10_000)
    bt.run()
    closed = bt.broker.closed_trades
    assert len(closed) == 1
    # SL=100 triggered when low dipped below it (bar at 103 → low ~102.5 not
    # triggered; next bar 99 → low 98.5 triggers at 100). exit_reason = 'sl'.
    assert closed[0].exit_reason == "sl"
    assert closed[0].exit_price == 100.0


# ---------------------------------------------------------------------------
# 2. Trailing-stop validation
# ---------------------------------------------------------------------------


def test_trail_percent_and_amount_mutually_exclusive_on_order():
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        Order(size=10, trail_percent=0.05, trail_amount=2.0)


def test_trail_percent_must_be_positive():
    with pytest.raises(ValueError, match=r"trail_percent must be > 0"):
        Order(size=10, trail_percent=0)
    with pytest.raises(ValueError, match=r"trail_percent must be > 0"):
        Order(size=10, trail_percent=-0.01)


def test_trail_amount_must_be_positive():
    with pytest.raises(ValueError, match=r"trail_amount must be > 0"):
        Order(size=10, trail_amount=0)
    with pytest.raises(ValueError, match=r"trail_amount must be > 0"):
        Trade(100.0, pd.Timestamp("2024-01-01"), 0, 10, trail_amount=-1.0)


def test_trail_validation_also_applies_on_trade():
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        Trade(100.0, pd.Timestamp("2024-01-01"), 0, 10,
              trail_percent=0.05, trail_amount=2.0)


# ---------------------------------------------------------------------------
# 3. Trailing-stop ratchet behavior (long)
# ---------------------------------------------------------------------------


def test_long_trail_seeds_initial_sl_at_open():
    """Initial trail SL = entry * (1 - trail_percent), available immediately."""
    t = Trade(100.0, pd.Timestamp("2024-01-01"), 0, 10, trail_percent=0.05)
    # Before any bar runs, SL is already seeded from entry price as HWM.
    assert t.sl == pytest.approx(95.0)
    assert t.trail_high == 100.0


def test_long_trail_ratchets_up_only_on_new_highs():
    t = Trade(100.0, pd.Timestamp("2024-01-01"), 0, 10, trail_percent=0.05)
    # Initial: sl=95, trail_high=100.
    t._update_trailing_stop(bar_high=102, bar_low=99)
    assert t.trail_high == 102
    assert t.sl == pytest.approx(96.9)
    # Lower-high bar: SL stays put.
    t._update_trailing_stop(bar_high=101, bar_low=99)
    assert t.trail_high == 102                # unchanged
    assert t.sl == pytest.approx(96.9)        # unchanged
    # New high again: SL bumps up.
    t._update_trailing_stop(bar_high=110, bar_low=105)
    assert t.trail_high == 110
    assert t.sl == pytest.approx(104.5)


def test_long_trail_amount_uses_absolute_distance():
    t = Trade(100.0, pd.Timestamp("2024-01-01"), 0, 10, trail_amount=2.0)
    assert t.sl == pytest.approx(98.0)
    t._update_trailing_stop(bar_high=110, bar_low=105)
    assert t.sl == pytest.approx(108.0)


def test_long_trail_respects_explicit_sl_floor():
    """Explicit SL acts as a floor — trail can only RAISE the stop above it."""
    t = Trade(100.0, pd.Timestamp("2024-01-01"), 0, 10,
              sl=98.0, trail_percent=0.05)
    # Initial trail-implied = 95, but explicit sl=98 wins.
    assert t.sl == 98.0
    t._update_trailing_stop(bar_high=101, bar_low=100)
    # Trail-implied now 95.95, still below 98 → keep 98.
    assert t.sl == 98.0
    t._update_trailing_stop(bar_high=110, bar_low=105)
    # Trail-implied now 104.5, above 98 → tighten to 104.5.
    assert t.sl == pytest.approx(104.5)


# ---------------------------------------------------------------------------
# 4. Trailing-stop ratchet behavior (short)
# ---------------------------------------------------------------------------


def test_short_trail_seeds_initial_sl_above_entry():
    t = Trade(100.0, pd.Timestamp("2024-01-01"), 0, -10, trail_percent=0.05)
    assert t.sl == pytest.approx(105.0)
    assert t.trail_low == 100.0


def test_short_trail_ratchets_down_on_new_lows():
    t = Trade(100.0, pd.Timestamp("2024-01-01"), 0, -10, trail_percent=0.05)
    t._update_trailing_stop(bar_high=101, bar_low=98)
    assert t.trail_low == 98
    assert t.sl == pytest.approx(102.9)
    # Higher-low bar: SL stays put.
    t._update_trailing_stop(bar_high=100, bar_low=99)
    assert t.trail_low == 98
    assert t.sl == pytest.approx(102.9)
    # New low: SL tightens.
    t._update_trailing_stop(bar_high=92, bar_low=90)
    assert t.trail_low == 90
    assert t.sl == pytest.approx(94.5)


# ---------------------------------------------------------------------------
# 5. End-to-end: trailing stop in a Backtest
# ---------------------------------------------------------------------------


def test_long_trail_stop_locks_in_profit_in_backtest():
    """Price runs up, then reverses — trailing stop should exit at a profit."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Bar 1 fill at open ~100, climbs to high 120 by bar 5, then crashes.
    close = np.array([100, 100, 105, 110, 115, 120, 110, 100, 95, 90], dtype=float)
    df = pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": 1000,
    }, index=dates)

    class _TrailBuy(Strategy):
        def prepare(self):
            self._bought = False
        def on_bar_close(self):
            if not self._bought:
                # Enter long with 10% trailing stop. As price climbs, SL ratchets
                # up. After the crash starts, SL fires before reaching breakeven.
                self.buy(size=10, trail_percent=0.10)
                self._bought = True

    bt = Backtest(df, _TrailBuy, cash=10_000)
    bt.run()
    closed = bt.broker.closed_trades
    assert len(closed) == 1
    assert closed[0].exit_reason == "sl"
    # Peak high was ~120.5, trail SL = 120.5 * 0.9 = 108.45.
    # Exit at 108.45 vs entry ~100 → profit > 0.
    assert closed[0].exit_price == pytest.approx(108.45, rel=0.01)
    assert closed[0].profit > 0


def test_trail_metadata_preserved_on_closed_trade():
    """After exit, the closed-trade record still reports trail_percent."""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    close = np.array([100, 100, 110, 120, 130, 110, 100, 90], dtype=float)
    df = pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": 1000,
    }, index=dates)

    class _S(Strategy):
        def prepare(self):
            self._bought = False
        def on_bar_close(self):
            if not self._bought:
                self.buy(size=5, trail_percent=0.05)
                self._bought = True

    bt = Backtest(df, _S, cash=10_000)
    bt.run()
    closed = bt.broker.closed_trades[0]
    assert closed.trail_percent == 0.05
    assert closed.trail_high is not None
    # HWM should have ratcheted up at least to the highest close (130) — its
    # bar high was 130.5.
    assert closed.trail_high >= 130


def test_strategy_trail_keyword_forwards_to_order():
    """Strategy.buy / sell pass trail_percent through to the underlying Order."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
        "Volume": 1000,
    }, index=dates)

    captured: dict = {}

    class _S(Strategy):
        def prepare(self):
            self._fired = False
        def on_bar_close(self):
            if not self._fired:
                self.buy(size=10, trail_percent=0.03)
                self._fired = True
                captured["pending"] = self.pending_orders

    bt = Backtest(df, _S, cash=10_000)
    bt.run()
    # After run(), trail metadata shows up on the open Trade.
    assert any(t.trail_percent == 0.03 for t in bt.broker.closed_trades + tuple(
        t for pos in bt.broker.positions.values() for t in pos.active_trades
    ))
