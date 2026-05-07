# Customizing Trading Environment

`TradingEnv` is composed of three pluggable pieces:

| Scheme | Decides | Default |
|---|---|---|
| `ActionScheme` | The agent's action space and how an action turns into orders | `DefaultAction`: 3-discrete (long / short / flat) |
| `ObserverScheme` | The observation space and what the agent sees each step | `DefaultObserver`: trailing-window of selected feature columns |
| `RewardScheme` | The scalar reward returned by `env.step` | `DefaultReward`: log-return of trades closed this step, minus commission |

Subclass the abstract base for any of them and pass your instance to
`TradingEnv(...)`:

```python
from qtrade.env import TradingEnv

env = TradingEnv(
    data=df,
    cash=10_000,
    action_scheme=MyActions(),
    observer_scheme=MyObservations(),
    reward_scheme=MyReward(),
)
```

If you don't pass one, the corresponding default is used. You can mix
custom and default schemes freely.

---

## ActionScheme

Two methods to implement:

- `action_space` — a `gymnasium.spaces.Space` describing what the agent
  can output.
- `get_orders(action, env) -> list[Order]` — translate the chosen action
  into actual `Order` objects to place. Return `[]` for "do nothing."

`env` exposes `env.position`, `env.equity`, `env.data`, `env.current_time`,
and the underlying `env._broker` if you need more.

### Example: position-sized actions

The default action scheme uses fixed `size=1`. Here's one that lets the
agent pick a position scale (small / medium / large) along with direction:

```python
import gymnasium as gym
from gymnasium.spaces import Space
from qtrade.env import ActionScheme
from qtrade.core import Order

class TieredAction(ActionScheme):
    """6-discrete: {flat, small_long, large_long, small_short, large_short, hold}."""

    SIZES = {"small": 5, "large": 20}

    @property
    def action_space(self) -> Space:
        return gym.spaces.Discrete(6)

    def get_orders(self, action: int, env) -> list[Order]:
        target = {
            0: 0,                            # flat
            1: +self.SIZES["small"],
            2: +self.SIZES["large"],
            3: -self.SIZES["small"],
            4: -self.SIZES["large"],
            5: env.position.size,            # hold
        }[action]
        delta = target - env.position.size
        return [Order(size=delta)] if delta != 0 else []
```

### Example: continuous (Box) action space

Useful for PPO / SAC if you want the agent to pick a target position
fraction directly.

```python
import numpy as np
from gymnasium.spaces import Box
from qtrade.env import ActionScheme
from qtrade.core import Order

class ContinuousAllocation(ActionScheme):
    """Single float in [-1, 1] — fraction of max size, sign = side."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size

    @property
    def action_space(self):
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def get_orders(self, action, env):
        target = int(round(float(action[0]) * self.max_size))
        delta = target - env.position.size
        return [Order(size=delta)] if delta != 0 else []
```

---

## ObserverScheme

Two methods to implement:

- `observation_space` — a `gymnasium.spaces.Space`. Make sure its `shape`
  / `dtype` match what `get_observation` actually returns.
- `get_observation(env) -> np.ndarray | dict[str, np.ndarray]` — pull
  whatever you want out of `env.data`, `env.position`, `env._broker`,
  etc., shaped to match `observation_space`.

The default observer returns a trailing window of feature columns.
You can subclass to return raw OHLCV, technical indicators, position
state, or a `Dict` space combining several modalities.

### Example: include position state

A common need is letting the agent know its current position size and
unrealized PnL alongside market data:

```python
import numpy as np
import gymnasium as gym
from qtrade.env import ObserverScheme

class WindowedObsWithPosition(ObserverScheme):
    def __init__(self, window_size: int, features: list[str]):
        self.window_size = window_size
        self.features = features

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "market": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, len(self.features)),
                dtype=np.float32,
            ),
            "position": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32,
            ),
        })

    def get_observation(self, env):
        market = (
            env.data[self.features]
            .iloc[-self.window_size:]
            .values.astype(np.float32)
        )
        position = np.array(
            [env.position.size, env._broker.unrealized_pnl],
            dtype=np.float32,
        )
        return {"market": market, "position": position}
```

> A `Dict` observation space requires a `MultiInputPolicy` in
> `stable-baselines3`. For most off-the-shelf policies, stick to a flat
> `Box`.

### Example: returns instead of prices

Raw price series are non-stationary. Most RL papers feed log-returns:

```python
import numpy as np
import gymnasium as gym
from qtrade.env import ObserverScheme

class LogReturnsObserver(ObserverScheme):
    def __init__(self, window_size: int):
        self.window_size = window_size

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size,), dtype=np.float32,
        )

    def get_observation(self, env):
        closes = env.data['Close'].iloc[-(self.window_size + 1):].values
        returns = np.diff(np.log(closes)).astype(np.float32)
        return returns
```

---

## RewardScheme

One required method, one optional:

- `get_reward(env) -> float` — the scalar reward for the current step.
- `reset()` — optional. Called from `env.reset()`. Use it to clear any
  per-episode state your reward function tracks.

The default rewards realized log-returns of trades closed this step
(minus commission). Three other patterns come up a lot:

### Example: equity-based reward (every step)

```python
from qtrade.env import RewardScheme

class EquityChangeReward(RewardScheme):
    """Reward = change in equity since the previous step."""

    def __init__(self):
        self._prev_equity = None

    def reset(self):
        self._prev_equity = None

    def get_reward(self, env) -> float:
        equity = env._broker.equity
        if self._prev_equity is None:
            self._prev_equity = equity
            return 0.0
        delta = equity - self._prev_equity
        self._prev_equity = equity
        return float(delta)
```

### Example: differential Sharpe ratio

Rewards risk-adjusted returns rather than raw PnL — see Moody & Saffell (1998):

```python
from qtrade.env import RewardScheme

class DifferentialSharpe(RewardScheme):
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self._A = 0.0   # EMA of return
        self._B = 0.0   # EMA of squared return
        self._prev_equity = None

    def reset(self):
        self._A = self._B = 0.0
        self._prev_equity = None

    def get_reward(self, env) -> float:
        equity = env._broker.equity
        if self._prev_equity is None or self._prev_equity == 0:
            self._prev_equity = equity
            return 0.0
        r = (equity - self._prev_equity) / self._prev_equity
        self._prev_equity = equity

        dA = r - self._A
        dB = r * r - self._B
        denom = (self._B - self._A ** 2) ** 1.5
        d_sharpe = (self._B * dA - 0.5 * self._A * dB) / denom if denom > 1e-9 else 0.0

        self._A += self.eta * dA
        self._B += self.eta * dB
        return float(d_sharpe)
```

### Example: drawdown penalty (composes another reward)

A risk-aware reward that wraps any base scheme and adds a penalty
proportional to current drawdown:

```python
from qtrade.env import RewardScheme

class DrawdownPenalty(RewardScheme):
    def __init__(self, base: RewardScheme, penalty: float = 1.0):
        self.base = base
        self.penalty = penalty

    def reset(self):
        self.base.reset()

    def get_reward(self, env) -> float:
        eq = env._broker.equity_history.loc[:env.current_time]
        peak = eq.cummax().iloc[-1]
        drawdown = (eq.iloc[-1] - peak) / peak  # ≤ 0
        return self.base.get_reward(env) + self.penalty * float(drawdown)
```

---

## Putting it together

```python
import yfinance as yf
from qtrade.env import TradingEnv
from qtrade.core import PercentageCommission

data = yf.download("GC=F", start="2023-01-01", end="2024-01-01",
                   interval="1d", multi_level_index=False)
data['returns'] = data['Close'].pct_change().fillna(0)

env = TradingEnv(
    data=data,
    cash=10_000,
    commission=PercentageCommission(0.001),
    action_scheme=TieredAction(),
    observer_scheme=WindowedObsWithPosition(
        window_size=10, features=['returns'],
    ),
    reward_scheme=DifferentialSharpe(eta=0.01),
    window_size=10,
    max_steps=200,
)

obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

For training the resulting environment with `stable-baselines3`, see
[Gym Trading Environment](trading_environment.md).
