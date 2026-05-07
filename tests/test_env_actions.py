"""Tests for qtrade.env.actions."""

from types import SimpleNamespace

import pytest
from gymnasium.spaces import Discrete

from qtrade.env.actions import ActionScheme, DefaultAction


def _env_with_position_size(size: int):
    return SimpleNamespace(position=SimpleNamespace(size=size))


def test_action_scheme_is_abstract():
    with pytest.raises(TypeError):
        ActionScheme()


def test_default_action_space():
    assert DefaultAction().action_space == Discrete(3)


def test_action_long_from_flat():
    orders = DefaultAction().get_orders(0, _env_with_position_size(0))
    assert len(orders) == 1
    assert orders[0].size == 1


def test_action_long_reverses_short():
    orders = DefaultAction().get_orders(0, _env_with_position_size(-1))
    assert len(orders) == 1
    assert orders[0].size == 2


def test_action_long_when_already_long_is_noop():
    assert DefaultAction().get_orders(0, _env_with_position_size(5)) == []


def test_action_short_from_flat():
    orders = DefaultAction().get_orders(1, _env_with_position_size(0))
    assert len(orders) == 1
    assert orders[0].size == -1


def test_action_short_reverses_long():
    orders = DefaultAction().get_orders(1, _env_with_position_size(1))
    assert len(orders) == 1
    assert orders[0].size == -2


def test_action_short_when_already_short_is_noop():
    assert DefaultAction().get_orders(1, _env_with_position_size(-3)) == []


def test_action_close_long_position():
    orders = DefaultAction().get_orders(2, _env_with_position_size(7))
    assert len(orders) == 1
    assert orders[0].size == -7


def test_action_close_short_position():
    orders = DefaultAction().get_orders(2, _env_with_position_size(-4))
    assert len(orders) == 1
    assert orders[0].size == 4


def test_action_close_when_flat_is_noop():
    assert DefaultAction().get_orders(2, _env_with_position_size(0)) == []


def test_action_unknown_returns_empty():
    assert DefaultAction().get_orders(99, _env_with_position_size(0)) == []
