# Gym Trading Environment

Qtrade provides a highly customizable Gym trading environment to facilitate research on reinforcement learning in trading.

## Initialize Gym Environment

The following example demonstrates how to create a basic trading environment. For advanced customization of Actions, Rewards, and Observers, please refer to [Customizing Trading Environment Guide](customize_environment.md).

```python
import yfinance as yf
import talib as ta
from qtrade.env import TradingEnv
from qtrade.core.commission import PercentageCommission

# Download historical gold futures data
data = yf.download(
    "GC=F", 
    start="2023-01-01", 
    end="2024-01-01", 
    interval="1d",
    multi_level_index=False
)

# Calculate technical indicators
df['Rsi'] = ta.rsi(df['Close'], length=14)  # 14-period RSI
df['Diff'] = df['Close'].diff()             # Price difference
df.dropna(inplace=True)

commission = PercentageCommission(0.001)     # 0.1% commission per trade

# Initialize trading environment
env = TradingEnv(
    data=df, 
    cash=3000,                # Initial capital
    max_steps=550,           # Maximum steps per episode
    commission=commission,    # Commission scheme
)
```

The example above uses the `DefaultObserver`, which includes all columns except OHLCV by default.

## Training

We'll use stable-baselines3 (sb3) for training our trading agent. First, install the library:

```bash
$ pip install stable-baselines3
```

Then train the model using PPO (Proximal Policy Optimization):

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
```

## Evaluation

After training, evaluate the model's performance:

```python
obs, _ = env.reset()
for _ in range(400):
    env.render('human')           # render live trading
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# print result stats
env.show_stats()
# plot a result chart 
env.plot()
```

You can visualize the trading process using `env.render('human')`. For recording purposes, you can save the renders as a video using sb3's [VecVideoRecorder](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#record-a-video) wrapper.

![Trading Environment Render](../_static/render_rgb.gif)
