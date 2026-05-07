"""Tests for qtrade.env.rewards."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qtrade.core import NoCommission
from qtrade.env.rewards import DefaultReward, RewardScheme


def _make_trade(*, size, entry_price, exit_price, exit_date):
    return SimpleNamespace(
        size=size,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_date=exit_date,
        is_long=size > 0,
    )


def _make_env(*, current_time, closed_trades, commission=None):
    return SimpleNamespace(
        current_time=current_time,
        closed_trades=closed_trades,
        commission=commission if commission is not None else NoCommission(),
    )


def test_reward_scheme_is_abstract():
    with pytest.raises(TypeError):
        RewardScheme()


def test_reward_zero_when_no_trades_closed_this_step():
    now = pd.Timestamp('2024-01-05')
    env = _make_env(current_time=now, closed_trades=[])
    assert DefaultReward().get_reward(env) == 0


def test_reward_ignores_trades_closed_other_steps():
    now = pd.Timestamp('2024-01-05')
    earlier = pd.Timestamp('2024-01-04')
    env = _make_env(
        current_time=now,
        closed_trades=[_make_trade(size=10, entry_price=100, exit_price=110, exit_date=earlier)],
    )
    assert DefaultReward().get_reward(env) == 0


def test_reward_long_profitable_trade_positive():
    now = pd.Timestamp('2024-01-05')
    env = _make_env(
        current_time=now,
        closed_trades=[_make_trade(size=1, entry_price=100, exit_price=110, exit_date=now)],
    )
    reward = DefaultReward().get_reward(env)
    # NoCommission: cost = log(1 - 0/110) = 0, profit = log(110/100)
    assert reward == pytest.approx(np.log(110 / 100))


def test_reward_short_profitable_trade_positive():
    now = pd.Timestamp('2024-01-05')
    env = _make_env(
        current_time=now,
        closed_trades=[_make_trade(size=-1, entry_price=110, exit_price=100, exit_date=now)],
    )
    reward = DefaultReward().get_reward(env)
    # ratio = 100/110, profit = log(2 - ratio)
    assert reward == pytest.approx(np.log(2 - 100 / 110))


def test_reward_reset_does_not_raise():
    DefaultReward().reset()
