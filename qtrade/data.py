"""Data loading helpers — convert common API outputs into Backtest-ready dicts.

Currently supports:

- :func:`from_yfinance` — multi-ticker download via the optional ``yfinance``
  dependency (install with ``pip install qtrade-lib[data]``).
- :func:`align_indexes` — utility to intersect indexes across a dict of
  DataFrames so ``Backtest`` accepts them as a multi-asset input.
"""

from __future__ import annotations

import pandas as pd


def from_yfinance(
    symbols: str | list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download multiple tickers via ``yfinance`` and return a Backtest-ready dict.

    All DataFrames are aligned on the **intersection** of their DatetimeIndexes —
    so the returned dict can be passed straight to ``Backtest(data=...)`` for
    multi-asset portfolios.

    Args:
        symbols: One ticker (string) or a list of tickers
            (e.g. ``["AAPL", "ES=F", "GC=F"]``).
        start, end: Optional date strings (``"YYYY-MM-DD"``) passed to yfinance.
        interval: yfinance interval — ``"1d"`` (default), ``"1h"``, ``"5m"``, etc.
        auto_adjust: Pass-through to yfinance for split / dividend adjustment.

    Returns:
        ``dict[str, pd.DataFrame]`` keyed by symbol with OHLCV columns
        and aligned DatetimeIndex.

    Raises:
        ImportError: if ``yfinance`` isn't installed.
        ValueError: if any ticker returns no data.

    Example:
        >>> from qtrade.data import from_yfinance
        >>> data = from_yfinance(["AAPL", "GC=F"], "2023-01-01", "2024-01-01")
        >>> sorted(data.keys())
        ['AAPL', 'GC=F']
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for qtrade.data.from_yfinance. "
            "Install it with: pip install qtrade-lib[data]"
        ) from e

    if isinstance(symbols, str):
        symbols = [symbols]

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.download(
            sym,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            multi_level_index=False,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for symbol '{sym}'")
        data[sym] = df.dropna()

    return align_indexes(data)


def align_indexes(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reindex every DataFrame on the intersection of all their indexes.

    ``Backtest`` requires every asset in a multi-asset run to share the same
    ``DatetimeIndex``. Use this helper when your data sources don't already
    line up — e.g. one ticker has a holiday the other doesn't.

    Args:
        data: ``dict[str, pd.DataFrame]`` keyed by asset symbol.

    Returns:
        Same shape as ``data`` but every DataFrame restricted to the
        intersection index.
    """
    if not data:
        return data
    common = None
    for df in data.values():
        common = df.index if common is None else common.intersection(df.index)
    return {sym: df.loc[common] for sym, df in data.items()}


__all__ = ["align_indexes", "from_yfinance"]
