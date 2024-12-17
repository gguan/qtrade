---
hide-toc: true
firstpage:
lastpage:
---

# QTrade

QTrade is a simple, modular, and highly customizable trading interface capable of handling backtesting, reinforcement learning tasks.

With features ranging from traditional signal-based strategies to reinforcement learning-driven approaches, QTrade allows traders to focus on developing and testing strategies without the burden of implementation details.

## Key Features
- **Backtesting**: Efficient simulation of trading strategies on historical data.
- **Reinforcement Learning**: Provides a highly customizable Gym environment for training and testing AI-driven trading agents.

```{toctree}
:caption: User Guide
:maxdepth: 2
:hidden:

guide/getting_started
guide/installation
guide/tutorials
customisation/index
reference/index
recommendations
```

```{toctree}
:caption: API
:hidden:

api/core
api/backtest
api/env
```

```{toctree}
:caption: Development
:hidden:

GitHub <https://github.com/gguan/qtrade>
kitchen-sink/index
stability
changelog
Contribute to the Docs <https://github.com/gguan/qtrade/blob/main/docs/README.md>
```

---

## How to Use

1. **Install QTrade**  
   Follow the instructions in the [Installation Guide](guide/installation.md) to set up QTrade in your Python environment.

2. **Explore Tutorials**  
   Learn to create strategies, backtest them, and execute live trades by following the [Getting Started Guide](guide/getting_started.md) and the [Tutorials](guide/tutorials.md).

3. **API Reference**  
   Dive deeper into QTrade's core components with the [API Reference](reference/index.md).

4. **Get Involved**  
   Contribute to the development or documentation through the links in the [Development Section](#development).

---

## Installing

QTrade can be installed with [pip](https://pip.pypa.io):

```bash
$ pip install qtrade-lib
```

Alternatively, you can obtain the latest source code from [GitHub](https://github.com/gguan/qtrade):

```bash
$ git clone https://github.com/gguan/qtrade.git
$ cd qtrade
$ pip install .
```

To run the example code:

```bash
$ pip install -r examples/requirements.txt
$ python examples/simple_strategy.py
```

---

## Usage

The [User Guide](guide/getting_started.md) is the place to learn how to use the library and accomplish common tasks. For more advanced customization, refer to the [Customization Guide](customisation/index.md).

The [Reference Documentation](reference/index.md) provides API-level documentation.
