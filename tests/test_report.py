"""Tests for qtrade.utils.report.build_html_report and Backtest.export_report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtrade.backtest.backtest import Backtest
from qtrade.backtest.strategy import Strategy
from qtrade.utils.report import build_html_report


class _BuyOnce(Strategy):
    """Buy 5 shares on the second bar, hold to the end."""

    def prepare(self):
        self._bought = False

    def on_bar_close(self):
        if not self._bought:
            self.buy(size=5)
            self._bought = True


@pytest.fixture
def trending_data():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    close = np.linspace(100, 119, 20)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(20, 1000),
        },
        index=dates,
    )


def test_export_report_writes_self_contained_html(tmp_path, trending_data):
    bt = Backtest(trending_data, _BuyOnce, cash=10_000)
    bt.run()
    out = tmp_path / "report.html"
    written = bt.export_report(str(out))

    assert out.exists()
    text = out.read_text(encoding="utf-8")

    # Has expected structural sections
    for section in (
        "<title>QTrade Backtest Report</title>",
        "<h2>Performance</h2>",
        "<h2>Charts</h2>",
        "<h2>Trade analytics</h2>",
        "<h2>Trade history</h2>",
    ):
        assert section in text, f"missing section: {section!r}"

    # Strategy class name is auto-filled into the header
    assert "_BuyOnce" in text
    # Bokeh JS / CSS bundled via CDN
    assert "bokeh" in text.lower()
    # Returned path is the resolved absolute path
    assert written == str(out.resolve())


def test_export_report_supports_custom_title_and_strategy_name(tmp_path, trending_data):
    bt = Backtest(trending_data, _BuyOnce, cash=10_000)
    bt.run()
    out = tmp_path / "custom.html"
    bt.export_report(str(out), title="Q1 Backtest", strategy_name="MyStrat v3")
    text = out.read_text(encoding="utf-8")
    assert "<title>Q1 Backtest</title>" in text
    assert "<h1>Q1 Backtest</h1>" in text
    assert "MyStrat v3" in text


def test_build_html_report_with_multi_asset(tmp_path):
    dates = pd.date_range("2024-01-01", periods=15, freq="D")

    def make(prices):
        return pd.DataFrame({
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": np.full(15, 1000),
        }, index=dates)

    data = {
        "AAPL": make(np.linspace(100, 115, 15)),
        "MSFT": make(np.linspace(300, 320, 15)),
    }

    class _MultiBuyOnce(Strategy):
        def prepare(self):
            self._bought = False

        def on_bar_close(self):
            if not self._bought:
                for asset in self.assets:
                    self.buy(asset, size=2)
                self._bought = True

    bt = Backtest(data, _MultiBuyOnce, cash=100_000)
    bt.run()

    out = tmp_path / "multi.html"
    build_html_report(bt.broker, out, strategy_name="Multi")
    text = out.read_text(encoding="utf-8")

    # Multi-asset header note
    assert "Multi-asset portfolio" in text
    # Per-asset section appears
    assert "<h2>Per-asset stats</h2>" in text
    # Both asset symbols mentioned
    assert "AAPL" in text and "MSFT" in text


def test_export_report_creates_parent_directory(tmp_path, trending_data):
    bt = Backtest(trending_data, _BuyOnce, cash=10_000)
    bt.run()
    out = tmp_path / "nested" / "subdir" / "report.html"
    bt.export_report(str(out))
    assert out.exists()
