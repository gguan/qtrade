from abc import ABC, abstractmethod
from datetime import timedelta
import numpy as np
import pandas as pd

class RewardScheme(ABC):
    @abstractmethod
    def get_reward(self, env: 'TradingEnv') -> float:
        """Calculate the reward based on the current environment state."""
        pass

    def reset(self) -> None:
        """Resets the reward scheme."""
        pass

class DefaultReward(RewardScheme):
    def get_reward(self, env: 'TradingEnv') -> float:
        reward = 0
        for trade in env.closed_trades:
            cost = np.log(1-env.commission)
            if trade.is_long:
                reward += np.log(trade.exit_price / trade.entry_price) + cost
            else:
                reward += np.log(2 - trade.exit_price/ trade.entry_price) + cost
        return reward
    
