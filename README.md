[![PyPI version](https://img.shields.io/pypi/v/qtrade-lib.svg)](https://pypi.org/project/qtrade-lib/)
[![Python versions](https://img.shields.io/pypi/pyversions/qtrade-lib.svg)](https://pypi.org/project/qtrade-lib/)
[![CI](https://github.com/gguan/qtrade/actions/workflows/ci.yml/badge.svg)](https://github.com/gguan/qtrade/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue.svg)](https://gguan.github.io/qtrade/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# QTrade

QTrade is a small, modular Python library for backtesting trading
strategies and training reinforcement learning agents on financial data.

## Features

- **Single-asset and portfolio backtesting**: pass one DataFrame for
  single-asset, or a `dict[str, DataFrame]` for multi-asset / portfolio
  strategies sharing one cash pool.
- **Walk-forward optimization**: rolling-window training + immediate
  out-of-sample test, the standard antidote to in-sample overfitting.
- **RL trading environment**: a [Gymnasium](https://gymnasium.farama.org/)
  `TradingEnv` with pluggable Action / Observation / Reward schemes,
  ready to drop into `stable-baselines3` and friends.
- **Bokeh reports**: equity, drawdown, per-asset OHLC panels, trade
  scatter — saved as a single self-contained HTML.
- **Curated metrics**: Sharpe, Sortino, Calmar, Omega, max drawdown
  duration, win rate, profit factor, etc.

See [COMPARISON.md](COMPARISON.md) for an honest take on where QTrade
fits next to backtrader / vectorbt / Backtesting.py.

## Installation

```bash
pip install qtrade-lib
```

Optional extras:

```bash
pip install "qtrade-lib[rl]"   # adds stable-baselines3 for RL evaluation utils
```

To work from source:

```bash
git clone https://github.com/gguan/qtrade.git
cd qtrade
pip install -e ".[dev]"
pre-commit install
pytest
```

## Quickstart

```python
import yfinance as yf
from qtrade.backtest import Strategy, Backtest

class SmaCrossover(Strategy):
    n1 = 5
    n2 = 20

    def prepare(self):
        for df in self._data.values():
            df['sma1'] = df['Close'].rolling(self.n1).mean()
            df['sma2'] = df['Close'].rolling(self.n2).mean()

    def on_bar_close(self):
        s1, s2 = self.data['sma1'], self.data['sma2']
        if s1.iloc[-2] < s2.iloc[-2] and s1.iloc[-1] > s2.iloc[-1]:
            self.buy()
        elif s1.iloc[-2] > s2.iloc[-2] and s1.iloc[-1] < s2.iloc[-1]:
            self.close()

data = yf.download("GC=F", start="2023-01-01", end="2024-01-01",
                   interval="1d", multi_level_index=False)

bt = Backtest(data, SmaCrossover, cash=10_000, trade_on_close=True)
bt.run()
bt.show_stats()
bt.plot()
```

For a multi-asset / walk-forward example, see
[`examples/portfolio_strategy.py`](examples/portfolio_strategy.py).

## Documentation

- [User Guide](https://gguan.github.io/qtrade/guide/getting_started.html)
- [Multi-asset / portfolio backtests](https://gguan.github.io/qtrade/guide/multi_asset.html)
- [RL trading environment](https://gguan.github.io/qtrade/guide/trading_environment.html)
- [API reference](https://gguan.github.io/qtrade/api/core.html)
- [Releasing](RELEASING.md)
- [Changelog](CHANGELOG.md)

## Requirements

- Python ≥ 3.10
- Runtime dependencies declared in `pyproject.toml` (numpy, pandas, scipy,
  matplotlib, bokeh, gymnasium, mplfinance, tqdm).

## References

This project is inspired by:

- [backtesting.py](https://github.com/kernc/backtesting.py) — single-asset backtest API design
- [tensortrade](https://github.com/tensortrade-org/tensortrade) — RL trading framework

## License

MIT — see [LICENSE](LICENSE).
