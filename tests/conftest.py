"""Shared fixtures for test modules."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlc_data_long():
    """40 trading days of OHLC data — wide enough for the TradingEnv default window."""
    dates = pd.date_range('2024-01-01', periods=40, freq='D')
    rng = np.random.default_rng(seed=42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
    return pd.DataFrame(
        {
            'Open': close - 0.5,
            'High': close + 1.0,
            'Low': close - 1.0,
            'Close': close,
            'Volume': 1000 + rng.integers(0, 100, size=len(dates)),
            'Feature1': rng.normal(0, 1, size=len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def ohlc_data_trending():
    """20 trading days of strictly upward-trending OHLC data — predictable pnl."""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    close = np.linspace(100, 119, 20)
    return pd.DataFrame(
        {
            'Open': close - 0.5,
            'High': close + 1.0,
            'Low': close - 1.0,
            'Close': close,
            'Volume': np.full(20, 1000),
        },
        index=dates,
    )
