"""Trade-level analytics — slice closed trades by time, duration, and outcome.

This module turns the raw trade history into questions you actually want
answered after a backtest:

- *How long do my winners hold vs my losers?*
  → :func:`hold_duration_distribution`
- *Are my entries clustered on Mondays? In specific months?*
  → :func:`entries_by_weekday`, :func:`entries_by_month`
- *What separates winning trades from losers — bigger size, higher entry
  price, longer hold?*
  → :func:`win_loss_feature_comparison`

Every function takes either a :class:`~qtrade.core.broker.Broker` (so you can
pass ``bt.broker`` directly) or the trade-history ``DataFrame`` produced by
:meth:`Broker.get_trade_history`. They return plain pandas objects so you
can ``.plot()`` them inline, hand them to seaborn, or render them yourself.

Example:

.. code-block:: python

    from qtrade.analytics import (
        hold_duration_distribution,
        entries_by_weekday,
        win_loss_feature_comparison,
    )

    bt.run()
    print(hold_duration_distribution(bt.broker))
    print(entries_by_weekday(bt.broker))
    print(win_loss_feature_comparison(bt.broker))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qtrade.core.broker import Broker


# Public type alias: anything that resolves to a trade-history DataFrame.
TradesInput = Union["Broker", pd.DataFrame]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_trades_df(source: TradesInput) -> pd.DataFrame:
    """Normalize input → trade-history DataFrame.

    Accepts a Broker (calls ``get_trade_history``) or the DataFrame itself.
    Returns a copy so callers can mutate freely.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    # Avoid runtime import of Broker (TYPE_CHECKING above); duck-type instead.
    if hasattr(source, "get_trade_history"):
        return source.get_trade_history().copy()
    raise TypeError(
        "Expected a qtrade.core.broker.Broker or a trade-history DataFrame; "
        f"got {type(source).__name__}"
    )


def _duration_hours(td: pd.Timedelta | None) -> float:
    """Convert a Timedelta to hours (float). NaT / None → nan."""
    if td is None or pd.isna(td):
        return float("nan")
    return td.total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hold_duration_distribution(
    source: TradesInput,
    by_outcome: bool = True,
) -> pd.DataFrame:
    """Summary statistics for trade hold times, optionally split by win/loss.

    Returns count, mean, median, std, min, 25%, 50%, 75%, and max — same
    shape as ``DataFrame.describe()``. Durations are reported in hours so
    intraday and daily strategies share one scale.

    Args:
        source: A :class:`Broker` or the DataFrame from
            :meth:`Broker.get_trade_history`.
        by_outcome: If ``True`` (default), one row per group: ``"all"``,
            ``"winners"``, ``"losers"``. If ``False``, a single ``"all"`` row.

    Returns:
        DataFrame indexed by group with columns ``count, mean, median, std,
        min, 25%, 50%, 75%, max`` — all in hours. Empty input → empty
        DataFrame with the same columns.

    Example:
        >>> hold_duration_distribution(bt.broker)
                  count    mean  median     std  ...     max
        all        42.0  18.50    12.0   14.20  ...    72.0
        winners    25.0  22.10    18.0   13.80  ...    72.0
        losers     17.0  13.20     8.0   12.50  ...    48.0
    """
    df = _to_trades_df(source)
    cols = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]

    if df.empty:
        return pd.DataFrame(columns=cols)

    df["_hours"] = df["Duration"].map(_duration_hours)

    def _row(sub: pd.DataFrame) -> dict:
        h = sub["_hours"].dropna()
        if h.empty:
            return dict.fromkeys(cols, float("nan")) | {"count": 0}
        q = h.quantile([0.25, 0.5, 0.75])
        return {
            "count": int(h.size),
            "mean": h.mean(),
            "median": h.median(),
            "std": h.std(ddof=0) if h.size > 1 else 0.0,
            "min": h.min(),
            "25%": q.loc[0.25],
            "50%": q.loc[0.5],
            "75%": q.loc[0.75],
            "max": h.max(),
        }

    rows = {"all": _row(df)}
    if by_outcome:
        rows["winners"] = _row(df[df["Profit"] > 0])
        rows["losers"] = _row(df[df["Profit"] < 0])

    return pd.DataFrame.from_dict(rows, orient="index")[cols]


def entries_by_weekday(
    source: TradesInput,
    metric: str = "count",
) -> pd.Series:
    """Tally or average trade-level metric by entry weekday.

    Useful for spotting calendar bias: does your strategy fire 80% of its
    entries on Monday? Are Friday entries net negative?

    Args:
        source: Broker or trade-history DataFrame.
        metric: One of:

            - ``"count"`` (default): number of entries per weekday.
            - ``"profit_sum"``: total ``Profit`` per weekday.
            - ``"profit_mean"``: average ``Profit`` per weekday.
            - ``"win_rate"``: fraction of trades closing positive (0.0–1.0).

    Returns:
        Series indexed by weekday name (``"Monday"`` … ``"Sunday"``),
        in calendar order. Weekdays with zero entries are included as 0
        (or NaN for ``profit_mean`` / ``win_rate``).

    Example:
        >>> entries_by_weekday(bt.broker, metric="count")
        Monday       12
        Tuesday       8
        Wednesday    11
        ...
    """
    return _entries_by_period(source, period="weekday", metric=metric)


def entries_by_month(
    source: TradesInput,
    metric: str = "count",
) -> pd.Series:
    """Tally or average trade-level metric by entry calendar month.

    Same shape as :func:`entries_by_weekday` but bucketed by month — useful
    for seasonal strategies (e.g. "sell in May") or for spotting that all
    your edge came from Q4 last year.

    Args:
        source: Broker or trade-history DataFrame.
        metric: One of ``"count"``, ``"profit_sum"``, ``"profit_mean"``,
            ``"win_rate"``. See :func:`entries_by_weekday`.

    Returns:
        Series indexed by month name (``"January"`` … ``"December"``),
        in calendar order.
    """
    return _entries_by_period(source, period="month", metric=metric)


_VALID_METRICS = ("count", "profit_sum", "profit_mean", "win_rate")
_WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _entries_by_period(
    source: TradesInput,
    *,
    period: str,
    metric: str,
) -> pd.Series:
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"metric must be one of {_VALID_METRICS}, got {metric!r}"
        )

    df = _to_trades_df(source)

    if period == "weekday":
        order = _WEEKDAY_ORDER
        bucket_name = "Weekday"
    elif period == "month":
        order = _MONTH_ORDER
        bucket_name = "Month"
    else:
        raise ValueError(f"unknown period {period!r}")

    if df.empty:
        return pd.Series(
            [0 if metric == "count" else float("nan")] * len(order),
            index=order,
            name=metric,
        )

    entry = pd.to_datetime(df["Entry Time"])
    if period == "weekday":
        df["_bucket"] = entry.dt.day_name()
    else:
        df["_bucket"] = entry.dt.month_name()

    if metric == "count":
        s = df["_bucket"].value_counts()
        s = s.reindex(order, fill_value=0)
    elif metric == "profit_sum":
        s = df.groupby("_bucket")["Profit"].sum().reindex(order, fill_value=0.0)
    elif metric == "profit_mean":
        s = df.groupby("_bucket")["Profit"].mean().reindex(order)
    else:  # win_rate
        s = (
            df.assign(_win=(df["Profit"] > 0).astype(float))
            .groupby("_bucket")["_win"]
            .mean()
            .reindex(order)
        )

    s.name = metric
    s.index.name = bucket_name
    return s


def win_loss_feature_comparison(
    source: TradesInput,
    extra_features: list[str] | None = None,
) -> pd.DataFrame:
    """Compare summary stats of winning vs losing trades.

    Helps answer "what's different about my losers?" — bigger size, longer
    hold, lower entry price, etc. Useful for filter design (e.g. "drop
    trades where size > 2× median").

    Default features compared: ``Size``, ``Entry Price``, ``Exit Price``,
    duration in hours. Pass extra column names from the trade-history
    DataFrame via ``extra_features``.

    Args:
        source: Broker or trade-history DataFrame.
        extra_features: Additional column names to summarize (must be
            present in the trade-history DataFrame).

    Returns:
        DataFrame indexed by feature, columns ``winners_mean``,
        ``losers_mean``, ``diff`` (= winners − losers), ``winners_median``,
        ``losers_median``. ``Size`` and any explicit-size column compare
        ``abs(size)`` so longs and shorts pool by magnitude.

    Example:
        >>> win_loss_feature_comparison(bt.broker)
                       winners_mean  losers_mean       diff  winners_median  ...
        Size                  85.20        92.10      -6.90           80.00  ...
        Entry Price          172.40       175.80      -3.40          170.50  ...
        Duration (h)          22.10        13.20       8.90           18.00  ...
    """
    df = _to_trades_df(source)
    cols = ["winners_mean", "losers_mean", "diff", "winners_median", "losers_median"]

    if df.empty:
        return pd.DataFrame(columns=cols)

    df["_size_abs"] = df["Size"].abs()
    df["Duration (h)"] = df["Duration"].map(_duration_hours)

    features = ["_size_abs", "Entry Price", "Exit Price", "Duration (h)"]
    feature_labels = {"_size_abs": "Size"}
    if extra_features:
        for f in extra_features:
            if f not in df.columns:
                raise ValueError(
                    f"Feature {f!r} not in trade-history columns: {list(df.columns)}"
                )
            features.append(f)

    winners = df[df["Profit"] > 0]
    losers = df[df["Profit"] < 0]

    rows = {}
    for f in features:
        label = feature_labels.get(f, f)
        w_vals = winners[f].dropna()
        l_vals = losers[f].dropna()
        rows[label] = {
            "winners_mean": w_vals.mean() if not w_vals.empty else float("nan"),
            "losers_mean": l_vals.mean() if not l_vals.empty else float("nan"),
            "diff": (
                (w_vals.mean() - l_vals.mean())
                if not w_vals.empty and not l_vals.empty
                else float("nan")
            ),
            "winners_median": w_vals.median() if not w_vals.empty else float("nan"),
            "losers_median": l_vals.median() if not l_vals.empty else float("nan"),
        }

    return pd.DataFrame.from_dict(rows, orient="index")[cols]


def _maybe_round(x: float, n: int = 4) -> float:
    """Round if finite, else return as-is. Avoids FloatingPointError on NaN."""
    if x is None or not np.isfinite(x):
        return x
    return round(x, n)


__all__ = [
    "entries_by_month",
    "entries_by_weekday",
    "hold_duration_distribution",
    "win_loss_feature_comparison",
]
