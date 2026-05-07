"""Tests for qtrade.env.environment.TradingEnv."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')  # noqa: E402  Headless backend for CI

from qtrade.env.environment import TradingEnv  # noqa: E402


@pytest.fixture
def env(ohlc_data_long):
    e = TradingEnv(
        data=ohlc_data_long,
        cash=10_000,
        margin_ratio=1.0,
        trade_on_close=True,
        window_size=5,
        max_steps=10,
        random_start=False,
        render_mode='human',
    )
    yield e
    e.close()


def test_env_reset_returns_observation_and_info(env):
    obs, info = env.reset()
    assert obs.shape == (5, 1)  # one feature column from fixture
    assert info == {}


def test_env_step_returns_5_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # long
    assert obs.shape == (5, 1)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {'equity', 'unrealized_pnl', 'cumulative_return', 'position',
            'total_trades', 'trades_profit', 'avg_trade_duration', 'is_success'} <= info.keys()


def test_env_step_max_steps_truncates(env):
    env.reset()
    truncated = False
    for _ in range(env.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(2)  # always close
        if terminated or truncated:
            break
    assert truncated or terminated


def test_env_position_property(env):
    env.reset()
    env.step(0)  # buy
    assert env.position.size != 0


def test_env_data_truncated_to_current_time(env):
    env.reset()
    env.step(0)
    assert env.data.index[-1] == env.current_time


def test_env_get_trade_history_is_dataframe(env):
    env.reset()
    for _ in range(3):
        env.step(0)
    env.step(2)  # close
    df = env.get_trade_history()
    assert isinstance(df, pd.DataFrame)


def test_env_random_start_picks_within_bounds(ohlc_data_long):
    e = TradingEnv(
        data=ohlc_data_long,
        cash=10_000,
        window_size=5,
        max_steps=10,
        random_start=True,
        render_mode='human',
    )
    e.reset(seed=123)
    assert 5 <= e.start_idx <= len(ohlc_data_long) - 10
    e.close()
