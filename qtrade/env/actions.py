from abc import ABC, abstractmethod
from typing import Any, List

import gymnasium as gym
from gymnasium.spaces import Space

from qtrade.core import Order

class ActionScheme(ABC):
    
    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()


    @abstractmethod
    def get_orders(self, action: Any, env: 'TradingEnv') -> List[Order]: # type: ignore
        """Returns a list of orders to be executed based on the action."""
        raise NotImplementedError()
    

class DefaultAction(ActionScheme):
    
    @property
    def action_space(self) -> Space:
        # Action 0 = Long, 1 = Short, 2 = Empty
        return gym.spaces.Discrete(3)
    
    def get_orders(self, action: int, env: 'TradingEnv') -> List[Order]: # type: ignore
        if action == 0:
            if env.position.size == 0:
                return [Order(size=1)]
            elif env.position.size < 0:
                return [Order(size=2)]
            else:
                return []
        elif action == 1:
            if env.position.size == 0:
                return [Order(size=-1)]
            elif env.position.size > 0:
                return [Order(size=-2)]
            else:
                return []
        else:
            if env.position.size != 0:
                return [Order(size=-env.position.size)]
            else:
                return []
        
