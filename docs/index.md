---
hide-toc: true
firstpage:
lastpage:
---

# QTrade

QTrade is a simple, modular, and highly customizable trading interface capable of handling backtesting, reinforcement learning tasks.

With features ranging from traditional signal-based strategies to reinforcement learning-driven approaches, QTrade allows traders to focus on developing and testing strategies without the burden of implementation details.

## Key Features
- **Backtesting**: Single-asset or multi-asset / portfolio strategies with shared cash and per-asset positions.
- **Reinforcement Learning**: A highly customizable Gymnasium environment for training and testing AI-driven trading agents.

```{toctree}
:caption: User Guide
:maxdepth: 1
:hidden:

guide/getting_started
guide/concepts
guide/multi_asset
guide/stops
guide/walk_forward
guide/commissions
guide/plot
guide/data_sources
guide/analytics
guide/reports
guide/trading_environment
guide/customize_environment
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

guide/stats_glossary
guide/faq
```

```{toctree}
:caption: API
:hidden:

api/core
api/backtest/index
api/env
```

```{toctree}
:caption: Development
:hidden:

GitHub <https://github.com/gguan/qtrade>
```

---

## How to Use

1. **Install QTrade**
   Follow the instructions in the [Installation Guide](#installation) to set up QTrade in your Python environment.

2. **Explore Tutorials**
   Learn to create strategies, backtest them, and use gym trading environment by following the [Getting Started](guide/getting_started.md) and the [Trading Environment](guide/trading_environment.md).

3. **API Reference**
   Dive deeper into QTrade's core components with the [API Reference](api/core.md).

---

(installation)=
## Installation

QTrade can be installed with [pip](https://pip.pypa.io):

```bash
$ pip install qtrade-lib
```

QTrade ships several optional extras for things you don't always need:

```bash
$ pip install "qtrade-lib[rl]"     # RL training utilities (stable-baselines3)
$ pip install "qtrade-lib[data]"   # yfinance loader for US tickers
$ pip install "qtrade-lib[cn]"     # akshare loader for Chinese A-shares + futures
```

Alternatively, you can obtain the latest source code from [GitHub](https://github.com/gguan/qtrade):

```bash
$ git clone https://github.com/gguan/qtrade.git
$ cd qtrade
$ pip install .
```

To run the example code:

```bash
$ python examples/simple_strategy.py
```

---

## Usage

The [User Guide](guide/getting_started.md) is the place to learn how to use the library and accomplish common tasks.

The API Reference covers the [core](api/core.md), [backtest](api/backtest/index.md), and [environment](api/env.md) modules.
