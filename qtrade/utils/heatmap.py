"""Optimization heatmap — visualize :meth:`Backtest.optimize` grid output.

Turn the ``all_results`` list returned by ``Backtest.optimize`` into a
2D heatmap so you can see at a glance *where* the parameter landscape is
flat vs sharp, and whether the "best" point is on a plateau (robust) or a
spike (likely overfit).

Example:

.. code-block:: python

    best_params, best_stats, results = bt.optimize(
        maximize='Sharpe Ratio',
        n1=range(5, 30, 5),
        n2=range(20, 80, 10),
    )

    from qtrade.utils.heatmap import plot_optimization_heatmap
    plot_optimization_heatmap(results, x='n1', y='n2', metric='Sharpe Ratio',
                              filename='heatmap.html')
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    PrintfTickFormatter,
)
from bokeh.palettes import RdYlGn11
from bokeh.plotting import figure


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten ``all_results`` into a DataFrame: one row per param combo.

    Columns: every key in ``params`` plus every key in ``stats``. Useful on
    its own for ``df.sort_values('Sharpe Ratio', ascending=False).head()`` or
    for handing to seaborn/plotly directly.
    """
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        row = dict(r["params"])
        # Stats values may be Series / numbers / NaT; pandas handles them.
        for k, v in r["stats"].items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def plot_optimization_heatmap(
    results: list[dict[str, Any]] | pd.DataFrame,
    x: str,
    y: str,
    metric: str,
    *,
    aggfunc: str = "mean",
    title: str | None = None,
    palette: tuple[str, ...] | None = None,
    filename: str | None = None,
    show_plot: bool = True,
    width: int = 700,
    height: int = 500,
):
    """Render a 2D heatmap of an optimization grid.

    Args:
        results: Either the ``all_results`` list returned by
            :meth:`Backtest.optimize`, or a DataFrame with the same shape
            (one row per param combo, columns = params + stats).
        x: Parameter name to lay out along the x-axis.
        y: Parameter name to lay out along the y-axis.
        metric: Stats key to color by (e.g. ``"Sharpe Ratio"``,
            ``"Total Return [%]"``, ``"Max Drawdown [%]"``).
        aggfunc: How to aggregate when ``x`` / ``y`` are not unique
            (after marginalizing over other params). Passed to
            :meth:`pandas.pivot_table`. Default ``"mean"``.
        title: Optional figure title. Defaults to ``f"{metric} ({x} × {y})"``.
        palette: Optional Bokeh palette tuple (color list). Defaults to
            ``RdYlGn11`` (red → yellow → green; high = green, good for
            most metrics. Reverse it for "lower is better" metrics).
        filename: If provided, save the figure as a standalone HTML file.
        show_plot: If True (default), open the figure in a browser tab via
            :func:`bokeh.plotting.show`. Set False for tests / headless.
        width, height: Figure pixel dimensions.

    Returns:
        The Bokeh ``figure`` object — useful for composing into a larger
        layout or for further customization.

    Raises:
        ValueError: if ``results`` is empty, ``x`` / ``y`` / ``metric`` are
            missing from the data, or the grid has fewer than 2 unique
            values along either axis.
    """
    if isinstance(results, list):
        df = results_to_dataframe(results)
    else:
        df = results.copy()

    if df.empty:
        raise ValueError("Cannot plot heatmap: results is empty.")

    for col in (x, y, metric):
        if col not in df.columns:
            raise ValueError(
                f"{col!r} not found in results. Available columns: "
                f"{sorted(df.columns)}"
            )

    # dropna=False keeps NaN-only cells in the pivot so the "no finite values"
    # check below catches them with a clearer error than the empty-pivot one.
    pivot = df.pivot_table(
        index=y, columns=x, values=metric, aggfunc=aggfunc, dropna=False,
    )
    if pivot.shape[0] < 1 or pivot.shape[1] < 1:
        raise ValueError(
            "Heatmap needs at least one row and one column after pivoting."
        )

    # Build a long-form CDS: one row per cell, with x/y as strings (so
    # categorical axes line up cleanly even when params are numeric).
    x_values = [str(v) for v in pivot.columns]
    y_values = [str(v) for v in pivot.index]

    long = pivot.stack(future_stack=True).reset_index()
    long.columns = [y, x, metric]
    long["_x"] = long[x].astype(str)
    long["_y"] = long[y].astype(str)

    finite = long[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError(
            f"Metric {metric!r} has no finite values across the grid."
        )
    vmin = float(finite.min())
    vmax = float(finite.max())
    # Avoid degenerate (vmin==vmax) which makes the color mapper unhappy.
    if vmin == vmax:
        vmax = vmin + 1e-9

    if palette is None:
        palette = RdYlGn11
    mapper = LinearColorMapper(palette=palette, low=vmin, high=vmax)

    source = ColumnDataSource(long)

    fig = figure(
        title=title or f"{metric} ({x} x {y})",
        x_range=x_values,
        y_range=y_values,
        x_axis_label=x,
        y_axis_label=y,
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
    )
    fig.toolbar.logo = None  # type: ignore[attr-defined]

    fig.rect(
        x="_x",
        y="_y",
        width=1.0,
        height=1.0,
        source=source,
        fill_color={"field": metric, "transform": mapper},
        line_color=None,
    )

    fig.add_tools(
        HoverTool(
            tooltips=[
                (x, "@_x"),
                (y, "@_y"),
                (metric, f"@{{{metric}}}{{0.000}}"),
            ]
        )
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(desired_num_ticks=8),
        formatter=PrintfTickFormatter(format="%.2f"),
        label_standoff=6,
        border_line_color=None,
        location=(0, 0),
    )
    fig.add_layout(color_bar, "right")

    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.xaxis.major_label_orientation = 0.7

    if show_plot:
        from bokeh.plotting import show as bokeh_show

        bokeh_show(fig)
    if filename:
        from bokeh.io import output_file, save

        output_file(filename)
        save(fig)

    return fig


__all__ = ["plot_optimization_heatmap", "results_to_dataframe"]
