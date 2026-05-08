"""Data loading helpers — convert common API outputs into Backtest-ready dicts.

Currently supports:

- :func:`from_yfinance` — multi-ticker download via the optional ``yfinance``
  dependency (install with ``pip install qtrade-lib[data]``).
- :func:`from_akshare_stock_a` — Chinese A-share daily/weekly/monthly OHLCV
  via the optional ``akshare`` dependency (install with
  ``pip install qtrade-lib[cn]``).
- :func:`from_akshare_futures` — Chinese futures (main-contract continuous)
  daily OHLCV via ``akshare``.
- :func:`align_indexes` — utility to intersect indexes across a dict of
  DataFrames so ``Backtest`` accepts them as a multi-asset input.
"""

from __future__ import annotations

import pandas as pd

# Standard OHLCV column shape that Backtest expects. All adapters
# normalize to this — capitalized, English, in this order.
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


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


# ---------------------------------------------------------------------------
# AKShare adapters (Chinese markets)
# ---------------------------------------------------------------------------

# AKShare returns DataFrames with Chinese column names; map to OHLCV.
_AKSHARE_STOCK_COLS = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
}
_AKSHARE_FUTURES_COLS = {
    "date": "Date",       # futures_main_sina uses lowercase English already
    "日期": "Date",
    "open": "Open",
    "开盘价": "Open",
    "high": "High",
    "最高价": "High",
    "low": "Low",
    "最低价": "Low",
    "close": "Close",
    "收盘价": "Close",
    "volume": "Volume",
    "成交量": "Volume",
}


def _import_akshare():
    """Lazy import with a friendly hint pointing at the [cn] extra."""
    try:
        import akshare as ak

        return ak
    except ImportError as e:
        raise ImportError(
            "akshare is required for qtrade.data.from_akshare_*. "
            "Install it with: pip install qtrade-lib[cn]"
        ) from e


def _to_akshare_date(s: str | None) -> str | None:
    """Accept ``"YYYY-MM-DD"`` or ``"YYYYMMDD"``; AKShare wants ``"YYYYMMDD"``."""
    if s is None:
        return None
    return s.replace("-", "")


def _normalize_ohlcv(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns, set DatetimeIndex on ``Date``, sort, keep OHLCV only."""
    df = df.rename(columns=column_map)
    if "Date" not in df.columns:
        raise ValueError(
            f"Adapter did not produce a 'Date' column; got {list(df.columns)}"
        )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    keep = [c for c in _OHLCV if c in df.columns]
    if not {"Open", "High", "Low", "Close"}.issubset(keep):
        raise ValueError(
            f"Missing OHLC columns after normalization; got {list(df.columns)}"
        )
    return df[keep].astype(float).dropna()


def from_akshare_stock_a(
    symbols: str | list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "daily",
    adjust: str = "qfq",
) -> dict[str, pd.DataFrame]:
    """Download Chinese A-share stocks via ``akshare``.

    Wraps ``akshare.stock_zh_a_hist`` and returns a Backtest-ready dict with
    standard OHLCV columns and a sorted DatetimeIndex. All DataFrames are
    aligned on the intersection of their indexes.

    Args:
        symbols: A-share stock code (string) or list of codes — e.g.
            ``"600519"`` (Kweichow Moutai), ``"000001"`` (Ping An Bank),
            ``["600519", "000001"]``. Use the bare 6-digit code without
            an exchange prefix.
        start, end: Date strings. Either ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` —
            both forms are accepted.
        period: ``"daily"`` (default), ``"weekly"``, or ``"monthly"``.
        adjust: Price adjustment mode. ``"qfq"`` (前复权, default — best for
            backtests), ``"hfq"`` (后复权), or ``""`` (no adjustment, raw prices).

    Returns:
        ``dict[str, pd.DataFrame]`` keyed by stock code with columns
        ``Open, High, Low, Close, Volume`` and a tz-naive DatetimeIndex.

    Raises:
        ImportError: if ``akshare`` isn't installed.
        ValueError: if any symbol returns no data.

    Example:
        >>> from qtrade.data import from_akshare_stock_a
        >>> data = from_akshare_stock_a(
        ...     ["600519", "000001"], "2024-01-01", "2024-12-31"
        ... )
        >>> sorted(data.keys())
        ['000001', '600519']
    """
    ak = _import_akshare()

    if isinstance(symbols, str):
        symbols = [symbols]
    start_ak = _to_akshare_date(start) or "19700101"
    end_ak = _to_akshare_date(end) or "20991231"

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = ak.stock_zh_a_hist(
            symbol=sym,
            period=period,
            start_date=start_ak,
            end_date=end_ak,
            adjust=adjust,
        )
        if df is None or df.empty:
            raise ValueError(f"No data returned for A-share symbol '{sym}'")
        data[sym] = _normalize_ohlcv(df, _AKSHARE_STOCK_COLS)

    return align_indexes(data)


def from_akshare_futures(
    symbols: str | list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download Chinese futures main-contract continuous OHLCV via ``akshare``.

    Wraps ``akshare.futures_main_sina`` (Sina's main-contract continuous series),
    which is the standard data source for Chinese futures backtests. Returns a
    Backtest-ready dict with standard OHLCV columns, aligned on the index
    intersection.

    Args:
        symbols: Futures main-contract code (string) or list of codes — e.g.
            ``"AU0"`` (gold), ``"RB0"`` (rebar), ``"CU0"`` (copper),
            ``"IF0"`` (CSI 300 index futures). The trailing ``0`` denotes the
            main-contract continuous series.
        start, end: Date strings. Either ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` —
            both forms are accepted.

    Returns:
        ``dict[str, pd.DataFrame]`` keyed by futures code with columns
        ``Open, High, Low, Close, Volume`` and a tz-naive DatetimeIndex.

    Raises:
        ImportError: if ``akshare`` isn't installed.
        ValueError: if any symbol returns no data.

    Example:
        >>> from qtrade.contracts import Contract
        >>> from qtrade.data import from_akshare_futures
        >>>
        >>> data = from_akshare_futures(["AU0", "RB0"], "2024-01-01", "2024-12-31")
        >>> # Then for a futures backtest:
        >>> # SHFE_AU = Contract(multiplier=1000, margin_ratio=0.08)
        >>> # SHFE_RB = Contract(multiplier=10,   margin_ratio=0.10)
        >>> # bt = Backtest(data, MyStrat, contracts={"AU0": SHFE_AU, "RB0": SHFE_RB})
    """
    ak = _import_akshare()

    if isinstance(symbols, str):
        symbols = [symbols]
    start_ak = _to_akshare_date(start) or "19700101"
    end_ak = _to_akshare_date(end) or "20991231"

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = ak.futures_main_sina(
            symbol=sym,
            start_date=start_ak,
            end_date=end_ak,
        )
        if df is None or df.empty:
            raise ValueError(f"No data returned for futures symbol '{sym}'")
        data[sym] = _normalize_ohlcv(df, _AKSHARE_FUTURES_COLS)

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


__all__ = [
    "align_indexes",
    "from_akshare_futures",
    "from_akshare_stock_a",
    "from_yfinance",
]
