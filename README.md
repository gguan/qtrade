[![CI Status](https://github.com/gguan/qtrade/actions/workflows/ci.yml/badge.svg)](https://github.com/gguan/qtrade/actions)
[![Python](https://img.shields.io/pypi/pyversions/qtrade-lib.svg)](https://badge.fury.io/py/qtrade-lib)
[![PyPI version](https://badge.fury.io/py/qtrade-lib.svg)](https://badge.fury.io/py/qtrade-lib)
![Coverage](https://img.shields.io/badge/coverage-87%25-green)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# QTrade

QTrade is a simple, modular, and highly customizable trading interface capable of handling backtesting, reinforcement learning tasks.

## Features

- Backtesting engine
- Gym Trading environment simulation

## Installation

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

### Run Example

To run the example code from repository:

```bash
$ pip install -r examples/requirements.txt
$ python examples/simple_strategy.py
```

### Requirements

- Python >= 3.8
- Dependencies listed in requirements.txt


### Usage

The [User Guide](guide/getting_started.md) is the place to learn how to use the library and accomplish common tasks. For more advanced customization, refer to the [Customization Guide](customisation/index.md).

The [Reference Documentation](reference/index.md) provides API-level documentation.


## References

This project is inspired by following projects.

- https://github.com/tensortrade-org/tensortrade
- https://github.com/kernc/backtesting.py


## License

This project is licensed under the MIT License - see the LICENSE file for details.