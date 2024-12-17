# Gym Trading Environment

Qtrade provides a highly customizable Gym trading environment to facilitate research on reinforcement learning in trading.

## Initializing the Gym Environment

In this example, we create a simple trading environment. For advanced usage, please refer to the guide on how to define custom Actions, Rewards, and Observers.

```python
import yfinance as yf
import talib as ta
from qtrade.env import TradingEnv
from qtrade.core.commission import PercentageCommission

# Download daily gold data
data = yf.download(
    "GC=F", 
    start="2023-01-01", 
    end="2024-01-01", 
    interval="1d", 
    multi_level_index=False
)

# Add indicators
df['Rsi'] = ta.rsi(df['Close'], length=14)
df['Diff'] = df['Close'].diff()
df.dropna(inplace=True)

features = ['Rsi', 'Diff', 'Close']
commission = PercentageCommission(0.001)

env = TradingEnv(
    data=df, 
    cash=3000,
    window_size=10, 
    features=features, 
    max_steps=550,  # Maximum steps per episode
    commission=commission, 
)
```

## Training

Here we use the popular stable-baselines3 library to train a policy. First, install the sb3 library:

```bash
pip install stable-baselines3
```

```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
```

## Evaluation

Now let's evaluate our trained model.

```python
obs, _ = env.reset()
for _ in range(400):
    env.render('human')  # Render live trading
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated, break

# Display result statistics
env.show_stats()
# Plot result chart
env.plot()
```

You can watch live trading by our model using `env.render('human')`. You can also save the renders to a video using the sb3 [VecVideoRecorder](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#record-a-video) wrapper.

![Trading Environment Render](../_static/render_rgb.gif)
