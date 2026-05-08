"""Tests for the AKShare adapters in qtrade.data.

We don't hit the AKShare network here — we monkeypatch the akshare module
to return fixed DataFrames matching its real-world column shape, then check
that our adapters normalize them correctly.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest


@pytest.fixture
def fake_akshare(monkeypatch):
    """Inject a stub `akshare` module exposing the two functions we wrap."""
    fake = types.ModuleType("akshare")

    def stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        # AKShare returns Chinese column names; mimic that shape.
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame({
            "日期": dates.strftime("%Y-%m-%d"),
            "股票代码": symbol,
            "开盘": [10.0, 10.5, 11.0, 11.2, 11.5],
            "收盘": [10.5, 11.0, 11.2, 11.5, 12.0],
            "最高": [10.6, 11.1, 11.3, 11.6, 12.1],
            "最低": [9.9, 10.4, 10.9, 11.1, 11.4],
            "成交量": [1000, 1100, 1200, 1300, 1400],
            "成交额": [10500.0, 11500.0, 12500.0, 13500.0, 14500.0],
        })

    def futures_main_sina(symbol, start_date, end_date):
        # Sina futures returns lowercase English column names.
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": [400.0, 405.0, 410.0, 412.0],
            "high": [406.0, 411.0, 415.0, 418.0],
            "low": [398.0, 403.0, 408.0, 410.0],
            "close": [405.0, 410.0, 412.0, 416.0],
            "volume": [10000, 11000, 12000, 13000],
            "hold": [50000, 51000, 52000, 53000],  # 持仓量, ignored
        })

    fake.stock_zh_a_hist = stock_zh_a_hist
    fake.futures_main_sina = futures_main_sina
    monkeypatch.setitem(sys.modules, "akshare", fake)
    return fake


def test_from_akshare_stock_a_normalizes_chinese_columns(fake_akshare):
    from qtrade.data import from_akshare_stock_a

    data = from_akshare_stock_a("600519", "2024-01-01", "2024-01-05")

    assert list(data.keys()) == ["600519"]
    df = data["600519"]
    # Chinese columns translated to OHLCV
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # Index is parsed as DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    # Sorted ascending
    assert df.index.is_monotonic_increasing
    # Values came through as floats
    assert df["Close"].iloc[0] == 10.5
    assert df["Volume"].iloc[-1] == 1400


def test_from_akshare_stock_a_accepts_string_or_list(fake_akshare):
    from qtrade.data import from_akshare_stock_a

    s = from_akshare_stock_a("000001", "2024-01-01", "2024-01-05")
    multi = from_akshare_stock_a(["000001", "600519"], "2024-01-01", "2024-01-05")

    assert set(s.keys()) == {"000001"}
    assert set(multi.keys()) == {"000001", "600519"}
    # Multi-asset call aligns indexes (fake returns identical 5-day range)
    assert multi["000001"].index.equals(multi["600519"].index)


def test_from_akshare_stock_a_accepts_yyyymmdd_date_format(
    monkeypatch, fake_akshare
):
    from qtrade.data import from_akshare_stock_a

    captured = {}
    real = fake_akshare.stock_zh_a_hist

    def spy(symbol, period, start_date, end_date, adjust):
        captured["start"] = start_date
        captured["end"] = end_date
        return real(symbol, period, start_date, end_date, adjust)

    monkeypatch.setattr(fake_akshare, "stock_zh_a_hist", spy)
    from_akshare_stock_a("600519", "20240101", "20240105")
    # Both forms reach AKShare in the YYYYMMDD shape it expects.
    assert captured == {"start": "20240101", "end": "20240105"}

    captured.clear()
    from_akshare_stock_a("600519", "2024-01-01", "2024-01-05")
    assert captured == {"start": "20240101", "end": "20240105"}


def test_from_akshare_stock_a_raises_on_empty_response(monkeypatch, fake_akshare):
    from qtrade.data import from_akshare_stock_a

    monkeypatch.setattr(
        fake_akshare,
        "stock_zh_a_hist",
        lambda **kw: pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="No data returned"):
        from_akshare_stock_a("999999", "2024-01-01", "2024-01-05")


def test_from_akshare_futures_normalizes_columns(fake_akshare):
    from qtrade.data import from_akshare_futures

    data = from_akshare_futures(["AU0", "RB0"], "2024-01-01", "2024-01-04")

    for sym in ("AU0", "RB0"):
        df = data[sym]
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
    assert data["AU0"].index.equals(data["RB0"].index)


def test_from_akshare_raises_clear_error_when_akshare_missing(monkeypatch):
    """No akshare installed → friendly ImportError with install hint."""
    import qtrade.data

    monkeypatch.setitem(sys.modules, "akshare", None)
    with pytest.raises(ImportError, match=r"qtrade-lib\[cn\]"):
        qtrade.data.from_akshare_stock_a("600519")
    with pytest.raises(ImportError, match=r"qtrade-lib\[cn\]"):
        qtrade.data.from_akshare_futures("AU0")
