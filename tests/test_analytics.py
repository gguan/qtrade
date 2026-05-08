"""Tests for qtrade.analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from qtrade.analytics import (
    entries_by_month,
    entries_by_weekday,
    hold_duration_distribution,
    win_loss_feature_comparison,
)


def _make_trades_df():
    """Hand-rolled trade-history DataFrame matching Broker.get_trade_history."""
    return pd.DataFrame({
        "Asset": ["AAPL"] * 4,
        "Type": ["Long", "Long", "Long", "Long"],
        "Size": [10, -20, 5, 15],
        "Entry Price": [100.0, 200.0, 50.0, 80.0],
        "Exit Price": [110.0, 195.0, 55.0, 75.0],
        # Mix Mondays (2024-01-01), Tuesdays, etc. across two months.
        "Entry Time": pd.to_datetime([
            "2024-01-01",  # Monday
            "2024-01-02",  # Tuesday
            "2024-02-05",  # Monday
            "2024-02-06",  # Tuesday
        ]),
        "Exit Date": pd.to_datetime([
            "2024-01-03",  # 2 days
            "2024-01-02",  # same day (0 days, intra)
            "2024-02-08",  # 3 days
            "2024-02-07",  # 1 day
        ]),
        "Profit": [100.0, -100.0, 25.0, -75.0],   # 2 winners, 2 losers
        "Tag": [None, None, None, None],
        "Exit Reason": ["close", "close", "close", "close"],
        "Duration": [
            pd.Timedelta(days=2),
            pd.Timedelta(days=0),
            pd.Timedelta(days=3),
            pd.Timedelta(days=1),
        ],
    })


# ---------------------------------------------------------------------------
# hold_duration_distribution
# ---------------------------------------------------------------------------


def test_hold_duration_returns_3_groups_with_describe_columns():
    df = _make_trades_df()
    out = hold_duration_distribution(df)
    assert list(out.index) == ["all", "winners", "losers"]
    expected_cols = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
    assert list(out.columns) == expected_cols
    assert out.loc["all", "count"] == 4
    # Mean of [48, 0, 72, 24] hours = 36
    assert out.loc["all", "mean"] == pytest.approx(36.0)
    # Winners: bars 0 (48h) and 2 (72h) → mean 60h
    assert out.loc["winners", "mean"] == pytest.approx(60.0)
    # Losers: bars 1 (0h) and 3 (24h) → mean 12h
    assert out.loc["losers", "mean"] == pytest.approx(12.0)


def test_hold_duration_by_outcome_false_returns_just_all():
    out = hold_duration_distribution(_make_trades_df(), by_outcome=False)
    assert list(out.index) == ["all"]


def test_hold_duration_handles_empty_input():
    out = hold_duration_distribution(pd.DataFrame())
    assert out.empty
    assert "count" in out.columns


def test_hold_duration_accepts_broker_via_duck_typing():
    """Anything with .get_trade_history() works."""
    class FakeBroker:
        def get_trade_history(self):
            return _make_trades_df()

    out = hold_duration_distribution(FakeBroker())
    assert out.loc["all", "count"] == 4


# ---------------------------------------------------------------------------
# entries_by_weekday / entries_by_month
# ---------------------------------------------------------------------------


def test_entries_by_weekday_count_orders_calendar_and_zero_fills():
    s = entries_by_weekday(_make_trades_df(), metric="count")
    assert list(s.index) == [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday",
    ]
    # 2 Monday entries (Jan 1, Feb 5), 2 Tuesday entries
    assert s["Monday"] == 2
    assert s["Tuesday"] == 2
    # No entries on Wednesday → reindexed to 0
    assert s["Wednesday"] == 0


def test_entries_by_weekday_profit_sum():
    s = entries_by_weekday(_make_trades_df(), metric="profit_sum")
    # Mondays: +100 + 25 = 125
    assert s["Monday"] == pytest.approx(125.0)
    # Tuesdays: -100 + -75 = -175
    assert s["Tuesday"] == pytest.approx(-175.0)


def test_entries_by_weekday_win_rate():
    s = entries_by_weekday(_make_trades_df(), metric="win_rate")
    # Mondays: both winners → 1.0
    assert s["Monday"] == pytest.approx(1.0)
    # Tuesdays: both losers → 0.0
    assert s["Tuesday"] == pytest.approx(0.0)
    # Empty buckets → NaN for win_rate / profit_mean (float divisions)
    assert pd.isna(s["Wednesday"])


def test_entries_by_month_orders_calendar():
    s = entries_by_month(_make_trades_df(), metric="count")
    assert list(s.index)[:3] == ["January", "February", "March"]
    assert s["January"] == 2
    assert s["February"] == 2
    assert s["March"] == 0


def test_entries_by_period_rejects_invalid_metric():
    with pytest.raises(ValueError, match=r"metric must be one of"):
        entries_by_weekday(_make_trades_df(), metric="not_a_metric")


# ---------------------------------------------------------------------------
# win_loss_feature_comparison
# ---------------------------------------------------------------------------


def test_win_loss_feature_comparison_compares_default_features():
    out = win_loss_feature_comparison(_make_trades_df())
    expected = ["winners_mean", "losers_mean", "diff", "winners_median", "losers_median"]
    assert list(out.columns) == expected
    # Default features (Size pooled by abs)
    assert set(out.index) == {"Size", "Entry Price", "Exit Price", "Duration (h)"}
    # Winners' Size: |10|, |5| → mean 7.5; Losers': |20|, |15| → mean 17.5
    assert out.loc["Size", "winners_mean"] == pytest.approx(7.5)
    assert out.loc["Size", "losers_mean"] == pytest.approx(17.5)
    assert out.loc["Size", "diff"] == pytest.approx(-10.0)
    # Duration in hours: winners (48 + 72)/2 = 60, losers (0 + 24)/2 = 12
    assert out.loc["Duration (h)", "winners_mean"] == pytest.approx(60.0)
    assert out.loc["Duration (h)", "losers_mean"] == pytest.approx(12.0)


def test_win_loss_feature_comparison_extra_features():
    df = _make_trades_df()
    df["My Feature"] = [1.0, 2.0, 3.0, 4.0]
    out = win_loss_feature_comparison(df, extra_features=["My Feature"])
    assert "My Feature" in out.index
    # Winners are rows 0 (1.0) and 2 (3.0) → mean 2.0; losers 1 (2.0), 3 (4.0) → mean 3.0
    assert out.loc["My Feature", "winners_mean"] == pytest.approx(2.0)
    assert out.loc["My Feature", "losers_mean"] == pytest.approx(3.0)


def test_win_loss_feature_comparison_rejects_unknown_feature():
    with pytest.raises(ValueError, match=r"Feature 'nope' not in"):
        win_loss_feature_comparison(_make_trades_df(), extra_features=["nope"])


def test_win_loss_feature_comparison_handles_empty():
    out = win_loss_feature_comparison(pd.DataFrame())
    assert out.empty


def test_analytics_rejects_unknown_input_type():
    with pytest.raises(TypeError, match=r"Expected a qtrade.core.broker.Broker"):
        hold_duration_distribution(42)  # type: ignore[arg-type]
