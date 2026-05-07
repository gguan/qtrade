---
title: Strategy
---

# Strategy

```{eval-rst}
.. autoclass:: qtrade.backtest.Strategy
```

## Lifecycle methods
```{eval-rst}
.. automethod:: qtrade.backtest.Strategy.prepare
.. automethod:: qtrade.backtest.Strategy.on_bar_close
```

## Order placement
```{eval-rst}
.. automethod:: qtrade.backtest.Strategy.buy
.. automethod:: qtrade.backtest.Strategy.sell
.. automethod:: qtrade.backtest.Strategy.close
```

## Read accessors

Single-asset shorthand: `data`, `position`, `equity`, `unrealized_pnl`,
`active_trades`, `closed_trades`, `pending_orders`.

Multi-asset accessors: `assets` (list of symbols), `positions` (dict),
`data_by_asset` (dict of truncated DataFrames). The single-asset shorthand
properties (`data`, `position`) raise `AttributeError` with a clear message
when the strategy is running on more than one asset.