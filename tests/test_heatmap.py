"""Tests for qtrade.utils.heatmap."""

from __future__ import annotations

import pytest
from bokeh.plotting import figure

from qtrade.utils.heatmap import plot_optimization_heatmap, results_to_dataframe


def _make_results():
    """Toy 3×3 grid with one clear best (n1=10, n2=30) → Sharpe 2.5."""
    return [
        {"params": {"n1": 5, "n2": 20}, "stats": {"Sharpe Ratio": 0.8, "Total Return [%]": 5}},
        {"params": {"n1": 5, "n2": 30}, "stats": {"Sharpe Ratio": 1.0, "Total Return [%]": 8}},
        {"params": {"n1": 5, "n2": 40}, "stats": {"Sharpe Ratio": 0.6, "Total Return [%]": 3}},
        {"params": {"n1": 10, "n2": 20}, "stats": {"Sharpe Ratio": 1.5, "Total Return [%]": 12}},
        {"params": {"n1": 10, "n2": 30}, "stats": {"Sharpe Ratio": 2.5, "Total Return [%]": 20}},
        {"params": {"n1": 10, "n2": 40}, "stats": {"Sharpe Ratio": 1.2, "Total Return [%]": 9}},
        {"params": {"n1": 15, "n2": 20}, "stats": {"Sharpe Ratio": 1.0, "Total Return [%]": 6}},
        {"params": {"n1": 15, "n2": 30}, "stats": {"Sharpe Ratio": 1.8, "Total Return [%]": 14}},
        {"params": {"n1": 15, "n2": 40}, "stats": {"Sharpe Ratio": 0.9, "Total Return [%]": 4}},
    ]


def test_results_to_dataframe_flattens_params_and_stats():
    df = results_to_dataframe(_make_results())
    assert {"n1", "n2", "Sharpe Ratio", "Total Return [%]"}.issubset(df.columns)
    assert len(df) == 9


def test_results_to_dataframe_handles_empty():
    assert results_to_dataframe([]).empty


def test_plot_heatmap_returns_bokeh_figure(tmp_path):
    out = tmp_path / "heatmap.html"
    fig = plot_optimization_heatmap(
        _make_results(),
        x="n1",
        y="n2",
        metric="Sharpe Ratio",
        filename=str(out),
        show_plot=False,
    )
    assert isinstance(fig, figure().__class__)
    # File written and non-empty
    assert out.exists() and out.stat().st_size > 0
    # Title falls back to "<metric> (<x> × <y>)"
    assert "Sharpe Ratio" in fig.title.text


def test_plot_heatmap_accepts_dataframe_input():
    df = results_to_dataframe(_make_results())
    fig = plot_optimization_heatmap(
        df, x="n1", y="n2", metric="Sharpe Ratio", show_plot=False,
    )
    # Categorical x_range and y_range = the unique sorted values as strings
    assert set(fig.x_range.factors) == {"5", "10", "15"}
    assert set(fig.y_range.factors) == {"20", "30", "40"}


def test_plot_heatmap_rejects_unknown_columns():
    with pytest.raises(ValueError, match=r"'nope' not found"):
        plot_optimization_heatmap(
            _make_results(), x="nope", y="n2", metric="Sharpe Ratio", show_plot=False,
        )


def test_plot_heatmap_rejects_empty_results():
    with pytest.raises(ValueError, match=r"empty"):
        plot_optimization_heatmap(
            [], x="n1", y="n2", metric="Sharpe Ratio", show_plot=False,
        )


def test_plot_heatmap_rejects_all_nan_metric():
    """If every grid cell is NaN, color mapping is meaningless — bail out."""
    bad = [
        {"params": {"n1": 1, "n2": 1}, "stats": {"M": float("nan")}},
        {"params": {"n1": 2, "n2": 2}, "stats": {"M": float("nan")}},
    ]
    with pytest.raises(ValueError, match=r"no finite values"):
        plot_optimization_heatmap(bad, x="n1", y="n2", metric="M", show_plot=False)


def test_plot_heatmap_aggregates_duplicates_via_aggfunc():
    """When the same (x, y) appears twice with different metric, aggfunc combines."""
    results = [
        {"params": {"n1": 5, "n2": 20, "extra": 1}, "stats": {"Sharpe Ratio": 1.0}},
        {"params": {"n1": 5, "n2": 20, "extra": 2}, "stats": {"Sharpe Ratio": 3.0}},
    ]
    fig = plot_optimization_heatmap(
        results, x="n1", y="n2", metric="Sharpe Ratio",
        aggfunc="mean", show_plot=False,
    )
    # No assertion on internal data; just ensure no crash and non-empty axes.
    assert fig.x_range.factors == ["5"]
    assert fig.y_range.factors == ["20"]


def test_plot_heatmap_via_backtest_method(ohlc_data_trending):
    """Backtest.plot_heatmap delegates to plot_optimization_heatmap."""
    from qtrade.backtest.backtest import Backtest
    from qtrade.backtest.strategy import Strategy

    class _Noop(Strategy):
        def prepare(self):
            pass
        def on_bar_close(self):
            pass

    bt = Backtest(ohlc_data_trending, _Noop)
    fig = bt.plot_heatmap(
        _make_results(), x="n1", y="n2", metric="Sharpe Ratio",
        show_plot=False,
    )
    assert isinstance(fig, figure().__class__)
