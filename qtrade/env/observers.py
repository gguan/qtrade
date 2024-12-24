from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np

class ObserverScheme(ABC):
    """Abstract base class for observation schemes."""

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Defines the observation space.

        Returns:
            Space: The observation space of the environment.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_observation(self, env: 'TradingEnv') -> Any: # type: ignore
        """Generates an observation from the environment.

        Args:
            env (TradingEnv): The trading environment instance.

        Returns:
            Any: The observation data.
        """
        raise NotImplementedError()

class DefaultObserver(ObserverScheme):
    """Default observer that returns the recent window of specified features."""

    def __init__(self, window_size: int, features: list[str]):
        """Initializes the DefaultObserver.

        Args:
            window_size (int): The size of the observation window.
            features (list[str]): List of feature names to include in the observation.
        """
        super().__init__()
        self.window_size = window_size
        self.features = features

    @property
    def observation_space(self) -> Space:
        """Defines the observation space as a Box space.

        Returns:
            Space: A Box space with shape (window_size, number of features).
        """
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.features)),
            dtype=np.float32
        )

    def get_observation(self, env: 'TradingEnv') -> Any: # type: ignore
        """Retrieves the observation from the environment data.

        Args:
            env (TradingEnv): The trading environment instance.

        Returns:
            np.ndarray: The observation array containing recent feature data.
        """
        obs = env.data[self.features].iloc[-self.window_size:].values.astype(np.float32)
        return obs

