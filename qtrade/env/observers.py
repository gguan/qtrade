from abc import ABC, abstractmethod
from typing import Any, List, Optional

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
    
    @property
    def observation_space(self) -> Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, len(timeseries_features)), dtype=np.float32)
    
    def get_observation(self, env: 'TradingEnv') -> Any: # type: ignore
        return np.array([[env.position.size]])
    