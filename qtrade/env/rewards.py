from abc import ABC, abstractmethod
import numpy as np

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
        curr_price = env.data.close.iloc[-1]
        prev_price = env.data.close.iloc[-2]

        if env.position.size > 0:
            reward += np.log(curr_price / prev_price)
        if env.position.size < 0:
            reward += np.log(2 - curr_price / prev_price) 

        return reward
    
