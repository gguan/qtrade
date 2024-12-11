from abc import ABC, abstractmethod
from typing import Any, List

import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np

class ObserverScheme(ABC):
    
    @property
    @abstractmethod
    def observation_space(self) -> Space:
        raise NotImplementedError()

    @abstractmethod
    def get_observation(self, env: 'TradingEnv') -> Any: # type: ignore
        raise NotImplementedError()
    

class DefaultObserver(ObserverScheme):

    def __init__(self, window_size: int = 10, features: List[str] = ['close']):
        super().__init__()
        self.window_size = window_size
        self.features = features
    
    @property
    def observation_space(self) -> Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, len(self.features)), dtype=np.float32)
    
    def get_observation(self, env: 'TradingEnv') -> Any: # type: ignore
        obs = env.data[self.features].iloc[-self.window_size:].values.astype(np.float32)
        return obs
    