"""Single-file HTML backtest report — stats + charts + trades, all in one.

:func:`build_html_report` (also exposed as :meth:`Backtest.export_report`)
takes a finished backtest and writes a self-contained HTML file you can
email, attach to a Slack message, or commit to a results folder. It bundles:

1. **Header** — strategy class, date range, starting cash.
2. **Stats table** — everything :func:`qtrade.utils.calculate_stats` returns.
3. **Equity / OHLC charts** — the same Bokeh layout :func:`plot_with_bokeh`
   produces.
4. **Per-asset stats** — when run on a multi-asset broker.
5. **Trade analytics** — hold-duration distribution and win/loss feature
   comparison from :mod:`qtrade.analytics`.
6. **Trade history** — the full trade-by-trade DataFrame.

Usage:

.. code-block:: python

    bt = Backtest(data, MyStrategy, cash=100_000)
    bt.run()
    bt.export_report("backtest_report.html")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from qtrade.utils.stats import calculate_stats, calculate_stats_per_asset

if TYPE_CHECKING:
    from qtrade.core.broker import Broker


# Lightweight CSS — embedded inline so the report is truly single-file.
_REPORT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 Helvetica, Arial, sans-serif;
    margin: 24px auto;
    max-width: 1200px;
    color: #222;
    line-height: 1.5;
}
h1 { margin: 0 0 4px 0; font-size: 22px; }
h2 { margin-top: 32px; font-size: 18px; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; }
h3 { margin-top: 24px; font-size: 15px; color: #555; }
.meta { color: #666; font-size: 13px; margin-bottom: 24px; }
.kvtable, .data-table { border-collapse: collapse; font-size: 13px; }
.kvtable td, .data-table td, .data-table th {
    padding: 4px 12px; border-bottom: 1px solid #eee;
}
.kvtable td:first-child { color: #555; min-width: 200px; }
.kvtable td:last-child { font-variant-numeric: tabular-nums;
                         text-align: right; min-width: 120px; }
.data-table th { background: #f7f7f7; text-align: left; }
.data-table td { font-variant-numeric: tabular-nums; }
.data-table tr:nth-child(even) td { background: #fafafa; }
.profit-pos { color: #1a7f37; }
.profit-neg { color: #cf222e; }
.section { margin-bottom: 8px; }
.charts { margin-top: 8px; }
.tables-wrap {
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
    align-items: flex-start;
}
"""


# ---------------------------------------------------------------------------
# Internal HTML builders
# ---------------------------------------------------------------------------


def _format_value(v) -> str:
    """Best-effort scalar → string for the stats / KV tables."""
    if v is None:
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, pd.Timedelta):
        # e.g. 3 days 04:30:00 → "3d 4h"
        days = v.days
        hours, rem = divmod(v.seconds, 3600)
        if days:
            return f"{days}d {hours}h"
        if hours:
            return f"{hours}h"
        minutes = rem // 60
        return f"{minutes}m"
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _kv_table(d: dict, css_class: str = "kvtable") -> str:
    """Render a 2-column "name → value" table from a dict."""
    rows = "".join(
        f"<tr><td>{k}</td><td>{_format_value(v)}</td></tr>" for k, v in d.items()
    )
    return f'<table class="{css_class}">{rows}</table>'


def _trade_history_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Render trade history with profit colored green/red and a row cap."""
    if df.empty:
        return "<p><em>No trades.</em></p>"

    # Make a display copy with formatted values; preserve raw Profit for
    # the css class.
    display = df.copy()
    truncated = len(display) > max_rows
    if truncated:
        display = display.head(max_rows)

    cols = list(display.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)

    body_rows = []
    for _, row in display.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            css = ""
            if c == "Profit" and pd.notna(val):
                css = " class=\"profit-pos\"" if val > 0 else (
                    " class=\"profit-neg\"" if val < 0 else ""
                )
            cells.append(f"<td{css}>{_format_value(val)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    note = (
        f"<p><em>Showing first {max_rows} of {len(df)} trades.</em></p>"
        if truncated
        else ""
    )
    return (
        '<table class="data-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
        f"{note}"
    )


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Generic DataFrame → styled HTML table."""
    if df.empty:
        return "<p><em>(empty)</em></p>"
    cols = list(df.columns)
    header = "<th></th>" + "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{_format_value(row[c])}</td>" for c in cols)
        body_rows.append(f"<tr><th>{idx}</th>{cells}</tr>")
    return (
        '<table class="data-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _bokeh_layout_html(broker: Broker) -> tuple[str, str]:
    """Render the existing plot_with_bokeh layout into (script, div).

    Importing inside the function keeps Bokeh out of the import path for
    callers who don't render reports.
    """
    from bokeh.embed import components

    from qtrade.utils.plot_bokeh import (
        _plot_multi_asset_layout,  # type: ignore[attr-defined]
        _plot_single_asset_layout,
    )

    if len(broker.data_by_asset) > 1:
        grid = _plot_multi_asset_layout(broker)
    else:
        grid = _plot_single_asset_layout(broker)
    return components(grid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_html_report(
    broker: Broker,
    output_path: str | Path,
    *,
    title: str = "QTrade Backtest Report",
    strategy_name: str | None = None,
) -> Path:
    """Write a single self-contained HTML report for a finished backtest.

    Args:
        broker: A finished :class:`Broker` (post-``Backtest.run``).
        output_path: Where to write the HTML file. Parent directories
            are created if missing.
        title: Document ``<title>`` and main heading.
        strategy_name: Optional label shown in the header. Falls back to
            the broker's strategy class name if available.

    Returns:
        The absolute path to the written file (``Path``).
    """
    from bokeh.embed import components
    from bokeh.resources import CDN

    from qtrade.analytics import (
        hold_duration_distribution,
        win_loss_feature_comparison,
    )
    from qtrade.utils.plot_bokeh import (
        _plot_multi_asset_layout,
        _plot_single_asset_layout,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Top-level stats
    stats = calculate_stats(broker)

    # 2. Header meta
    is_multi = len(broker.data_by_asset) > 1
    first_idx = next(iter(broker.data_by_asset.values())).index
    date_range = f"{first_idx[0].date()} → {first_idx[-1].date()}"
    header_meta = {
        "Strategy": strategy_name or "—",
        "Mode": "Multi-asset portfolio" if is_multi else "Single-asset",
        "Assets": ", ".join(sorted(broker.data_by_asset.keys())),
        "Period": date_range,
        "Starting cash": f"{stats.get('Initial Cash', '—'):,.2f}"
        if isinstance(stats.get("Initial Cash"), (int, float)) else "—",
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 3. Render Bokeh layout
    if is_multi:
        grid = _plot_multi_asset_layout(broker)
    else:
        grid = _plot_single_asset_layout(broker)
    chart_script, chart_div = components(grid)

    # 4. Per-asset breakdown (multi-asset only)
    per_asset_html = ""
    if is_multi:
        try:
            per_asset = calculate_stats_per_asset(broker)
            # dict[str, dict] → DataFrame with assets as columns
            per_asset_df = pd.DataFrame(per_asset)
            per_asset_html = (
                "<h2>Per-asset stats</h2>"
                + _df_to_html_table(per_asset_df)
            )
        except Exception:
            per_asset_html = ""

    # 5. Trade analytics
    try:
        duration_df = hold_duration_distribution(broker)
        winloss_df = win_loss_feature_comparison(broker)
    except Exception:
        duration_df = pd.DataFrame()
        winloss_df = pd.DataFrame()

    analytics_html = (
        '<div class="tables-wrap">'
        '<div><h3>Hold duration (hours)</h3>'
        + _df_to_html_table(duration_df)
        + "</div>"
        '<div><h3>Winners vs losers</h3>'
        + _df_to_html_table(winloss_df)
        + "</div></div>"
    )

    # 6. Trade history
    trade_history = broker.get_trade_history()

    # 7. Compose final HTML
    bokeh_css = "\n".join(f'<link rel="stylesheet" href="{u}">' for u in CDN.css_files)
    bokeh_js = "\n".join(f'<script src="{u}"></script>' for u in CDN.js_files)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
{bokeh_css}
{bokeh_js}
<style>{_REPORT_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{date_range} &middot; generated {header_meta['Generated']}</div>

<div class="tables-wrap">
  <div class="section">
    <h2>Run info</h2>
    {_kv_table(header_meta)}
  </div>
  <div class="section">
    <h2>Performance</h2>
    {_kv_table(stats)}
  </div>
</div>

<h2>Charts</h2>
<div class="charts">{chart_div}</div>

{per_asset_html}

<h2>Trade analytics</h2>
{analytics_html}

<h2>Trade history</h2>
{_trade_history_table(trade_history)}

{chart_script}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path.resolve()


__all__ = ["build_html_report"]
