# QTrade

A powerful quantitative trading framework for financial markets.

## Features

- Backtesting engine
- Market data components
- Trading environment simulation
- Strategy development tools

## Installation

### From Source

```bash
git clone https://github.com/yourusername/qtrade.git
cd qtrade
pip install -e .
```

### Run Example

```bash
python examples/backtest/simply_strategy.py
```


### Requirements

- Python >= 3.7
- Dependencies listed in requirements.txt

## Project Structure

```
qtrade/
├── qtrade/              # Main package
│   ├── backtest/       # Backtesting engine
│   ├── components/     # Trading components
│   └── env/           # Trading environment
├── tests/              # Unit tests
├── examples/           # Example scripts
└── docs/              # Documentation
```

## Usage

Basic example:

```python
from qtrade import Backtest
from qtrade.components import Strategy

# Your trading logic here
```

## Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.