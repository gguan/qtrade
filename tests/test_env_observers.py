"""Tests for qtrade.env.observers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box

from qtrade.env.observers import DefaultObserver, ObserverScheme


def test_observer_scheme_is_abstract():
    with pytest.raises(TypeError):
        ObserverScheme()


def test_default_observer_space_shape():
    obs = DefaultObserver(window_size=5, features=['a', 'b', 'c'])
    space = obs.observation_space
    assert isinstance(space, Box)
    assert space.shape == (5, 3)
    assert space.dtype == np.float32


def test_default_observer_get_observation_returns_window():
    df = pd.DataFrame(
        {'a': np.arange(10, dtype=float), 'b': np.arange(10, 20, dtype=float)}
    )
    env = SimpleNamespace(data=df)
    obs = DefaultObserver(window_size=4, features=['a', 'b']).get_observation(env)
    assert obs.shape == (4, 2)
    assert obs.dtype == np.float32
    # Last row should be the most recent values
    assert obs[-1, 0] == pytest.approx(9.0)
    assert obs[-1, 1] == pytest.approx(19.0)
