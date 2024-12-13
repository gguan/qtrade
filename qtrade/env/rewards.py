from abc import ABC, abstractmethod
import numpy as np

class RewardScheme(ABC):

    @abstractmethod
    def get_reward(self, env: 'TradingEnv') -> float: # type: ignore
        """Calculate the reward based on the current environment state."""
        pass

    def reset(self) -> None:
        """Resets the reward scheme."""
        pass

class DefaultReward(RewardScheme):

    def get_reward(self, env: 'TradingEnv') -> float: # type: ignore
        step_reward = 0
        for trade in env.closed_trades:
            if trade.exit_date == env.current_time:
                cost = np.log(1-env.commission.calculate_commission(trade.size, trade.entry_price)/trade.exit_price)
                ratio = trade.exit_price / trade.entry_price
                if trade.is_long:
                    profit = np.log(ratio) 
                else:    
                    profit = np.log(2 - ratio)
                step_reward += profit + cost

        return step_reward
    
