import os
import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


class EvalWithInfoCallback(EvalCallback):
    """
    Custom evaluation callback for tracking trading metrics during evaluation.
    Extends EvalCallback to include additional info tracking capabilities.

    Args:
        eval_env: The environment used for evaluation
        best_model_save_path: Path to save the best model
        log_path: Path for logging evaluation results
        info_keywords: List of keywords to track from environment info dict
        eval_freq: Evaluate the agent every n steps
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment during evaluation
        verbose: Verbosity level
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path,
        log_path=None,
        info_keywords=None,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose
        )
        self.info_keywords = info_keywords if info_keywords is not None else []

    def _on_step(self) -> bool:
        """
        Evaluation method called every n steps.
        Performs evaluation and logs metrics to TensorBoard.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run evaluation and get results
            mean_reward, std_reward, infos = evaluate_policy_with_infos(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic
            )
            
            # Log to TensorBoard if logger exists
            if self.logger is not None:
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                
                # Initialize dictionary for info keywords
                info_values = {key: [] for key in self.info_keywords}

                # Collect values for each info keyword from episode infos
                for info in infos:
                    for key in self.info_keywords:
                        if key in info:
                            info_values[key].append(info[key])

                # Calculate and log averages for each info keyword
                for key, values in info_values.items():
                    if values:
                        avg_value = np.mean(values)
                        self.logger.record(f"eval/mean_{key}", avg_value)
                            
                self.logger.dump(self.num_timesteps)

            # Print evaluation results if verbose
            if self.verbose > 0:
                print(f"=== Evaluation at step {self.num_timesteps} ===")
                print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                for key, values in info_values.items():
                    if values:
                        print(f"Mean {key}: {np.mean(values):.2f}")
                
            # Save best model if current performance is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    filename = f"best_model_step_{self.num_timesteps}"
                    self.model.save(os.path.join(self.best_model_save_path, filename))
                    print(f"Saving new best model at step {self.num_timesteps} as {filename}")
                    
        return True


def evaluate_policy_with_infos(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float, list[dict[str, Any]]], tuple[list[float], list[int], list[dict[str, Any]]]]:
    """
    Evaluates a policy by running it for n episodes and returns results with additional info.

    Args:
        model: The RL agent to evaluate (must implement predict method)
        env: The environment to evaluate in
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        callback: Optional callback function for additional checks
        reward_threshold: Minimum expected reward per episode
        return_episode_rewards: Whether to return detailed episode data
        warn: Whether to warn about missing Monitor wrapper

    Returns:
        Either (mean_reward, std_reward, episode_infos) or
        (episode_rewards, episode_lengths, episode_infos) if return_episode_rewards is True
    """
    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_infos = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    episode_infos.append(info)
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_infos
    return mean_reward, std_reward, episode_infos
