"""Smoke tests for qtrade.utils.plot_bokeh.

Verifies the function builds without errors against the current bokeh version
and that the source DataFrame fields the JS callback / HoverTool reference
actually exist (catches case-mismatch regressions like the one fixed in
the bokeh 3.9 upgrade)."""

from unittest.mock import patch

import pytest

from qtrade.core import Broker, NoCommission
from qtrade.utils.plot_bokeh import plot_with_bokeh


@pytest.fixture
def broker_with_volume(ohlc_data_long):
    broker = Broker(ohlc_data_long, cash=10_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    for ts in ohlc_data_long.index:
        broker.process_bar(ts)
    broker._open_trade(entry_price=100.0, entry_date=ohlc_data_long.index[0], size=10)
    broker.close_all_positions()
    return broker


def test_plot_with_bokeh_runs_without_error(broker_with_volume):
    with patch('qtrade.utils.plot_bokeh.show'):
        plot_with_bokeh(broker_with_volume)


def test_plot_with_bokeh_writes_html_when_filename_given(broker_with_volume, tmp_path):
    output = tmp_path / "report.html"
    with patch('qtrade.utils.plot_bokeh.show'):
        plot_with_bokeh(broker_with_volume, filename=str(output))
    assert output.exists()
    assert output.stat().st_size > 0
    text = output.read_text()
    # The HoverTool tooltip and JS callback both reference capitalized OHLCV
    # column names — make sure the file we generated actually contains them.
    assert 'High' in text
    assert 'Low' in text
    assert 'Volume' in text


def test_plot_with_bokeh_runs_without_volume_column(ohlc_data_trending):
    """Bug #6/#7: plot_volume detection was buggy (lowercase) and the JS
    callback unconditionally referenced extra_y_ranges['volume']. Should now
    handle data without a Volume column gracefully."""
    no_vol = ohlc_data_trending.drop(columns=['Volume'])
    from qtrade.core import Broker, NoCommission
    broker = Broker(no_vol, cash=10_000, commission=NoCommission(),
                    margin_ratio=1.0, trade_on_close=True)
    for ts in no_vol.index:
        broker.process_bar(ts)
    with patch('qtrade.utils.plot_bokeh.show'):
        plot_with_bokeh(broker)  # would crash before the fix
