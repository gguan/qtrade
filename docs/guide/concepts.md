# Concepts

QTrade is a small library, but understanding how its components fit
together makes both the API and the source code easier to navigate.
This page is the mental model.

## The pieces

```
        ┌─────────────────────────────────────────────────┐
        │                    Backtest                     │
        │  (orchestrates a full run; iterates the bars)   │
        └────────────────────┬────────────────────────────┘
                             │ owns
                             ▼
        ┌─────────────────────────────────────────────────┐
        │                    Strategy                     │
        │  user code: prepare(), on_bar_close()           │
        │  reads market data, places Orders via the Broker│
        └────────────────────┬────────────────────────────┘
                             │ places orders on
                             ▼
        ┌─────────────────────────────────────────────────┐
        │                     Broker                      │
        │  fills Orders into Trades, tracks cash, equity, │
        │  margin, SL/TP, equity history                  │
        └──────────┬──────────────────────────┬───────────┘
                   │ holds                     │ records
                   ▼                           ▼
        ┌──────────────────┐         ┌─────────────────────┐
        │   Position       │         │   Order / Trade     │
        │ active + closed  │         │   filled / closed   │
        │ Trade objects    │         │   queues            │
        └──────────────────┘         └─────────────────────┘
```

For RL workflows, replace `Backtest` with `TradingEnv` — same `Broker`
underneath, just driven by Gymnasium's `step()` instead of a Python
`for` loop.

## What each component does

### Broker

The simulation core. You don't usually instantiate it directly — both
`Backtest` and `TradingEnv` create one for you.

The Broker:
- Holds the OHLCV data (single asset or `dict[str, DataFrame]` for portfolios)
- Tracks `cash`, computes `equity`, `unrealized_pnl`, `available_margin`
- Maintains queues of pending / executing / filled / closed orders
- Triggers SL / TP exits per bar
- Owns one `Position` per asset
- Records the portfolio-level `equity_history` Series

### Order

A trade *intention* — what you'd like to do. Created with a signed size
(positive = buy, negative = sell), optional `limit` / `stop` / `sl` /
`tp` / `tag`, and an `asset` symbol. Orders go through the broker's
queues and either fill or get rejected (typically for insufficient margin).

### Trade

A *filled* position. Each Order that fills opens (or partially closes)
a Trade. While open, the Trade has `entry_price`, `entry_date`, `size`,
optional `sl` / `tp` (modifiable mid-trade — see [Stops](stops.md)),
and optional trailing-stop state. When the Trade is closed (by an
opposite-side order, an SL/TP trigger, or end-of-backtest), it gets
`exit_price`, `exit_date`, `profit`, and an `exit_reason`.

### Position

A holder for trades on one asset. `active_trades` are open; `closed_trades`
are closed. `position.size` is the net signed size across all active
trades on that asset.

For multi-asset, the Broker holds `positions: dict[str, Position]` —
one per asset. `broker.position` (singular) is a convenience accessor
that only works in single-asset mode.

### Strategy

The piece you write. Subclass `qtrade.backtest.Strategy` and implement:

- `prepare()` — runs once before the bar loop. Add indicators here.
- `on_bar_close()` — called on every bar. Place orders, manage positions.

Inside `on_bar_close()` you have:
- `self.data` (single-asset) / `self.data_by_asset[asset]` (multi-asset)
  — the OHLCV slice up to the current bar.
- `self.position` / `self.positions[asset]` — current holdings.
- `self.equity`, `self.unrealized_pnl` — portfolio-level snapshots.
- `self.buy(...)`, `self.sell(...)`, `self.close(...)` — order helpers.

### Backtest

The driver. Given `data`, a `Strategy` class, cash, commission, and
margin settings, it:

1. Constructs a `Broker` and a `Strategy` instance.
2. Calls `strategy.prepare()`.
3. Loops over bars, calling `broker.process_bar(ts)` then
   `strategy.on_bar_close()`.
4. Closes open positions at the end.

Plus utilities: `optimize`, `walk_forward_optimize`, `show_stats`,
`get_trade_history`, `plot`.

### TradingEnv (RL)

A Gymnasium environment wrapping the same `Broker`. Composes three
pluggable schemes:

| Scheme | Decides |
|---|---|
| `ActionScheme` | Action space + how an action becomes Orders |
| `ObserverScheme` | Observation space + what the agent sees |
| `RewardScheme` | The scalar reward each step |

See [Customizing Trading Environment](customize_environment.md) for the
full story.

## Lifecycle of a backtest

For a typical bar, in this order:

1. `Backtest.run` advances to bar *i*.
2. `Broker.process_bar(ts)` fires:
   - Drop cancelled / rejected orders from the pending queue.
   - Process executing orders (queued for fill at the next bar's open).
   - Check SL / TP on every active trade — close any that triggered.
   - Process pending stop / limit orders.
   - Update `equity_history` with the new bar's equity.
3. `Strategy.on_bar_close()` runs — you can read `self.data` (truncated
   to *i*) and place new orders. New orders fill immediately if
   `trade_on_close=True`, otherwise queue for next bar's open.

After the final bar, `Broker.close_all_positions()` flushes everything
at the last bar's close.

## Key design choices (and why)

- **Event-loop, not vectorized.** Strategy code reads like real trading
  logic (`if x then buy`). Easier to write, easier to debug. Slower
  than vectorbt for huge sweeps — see [COMPARISON.md](https://github.com/gguan/qtrade/blob/main/COMPARISON.md).
- **No look-ahead by construction.** `self.data` is sliced to
  `[:current_time]`. You literally cannot read future bars from inside
  `on_bar_close()`.
- **One cash pool, per-asset positions.** Multi-asset backtests get
  realistic portfolio behavior (margin shared across assets) without
  having to manage N independent brokers.
- **Strategy + RL share one Broker.** Same accounting code, same SL/TP
  logic, same fill semantics. An RL agent in `TradingEnv` is governed
  by exactly the same rules as a `Strategy` in `Backtest`.

## Where to read next

- [Getting Started](getting_started.md) — the canonical Strategy /
  Backtest workflow on a single asset.
- [Multi-asset / portfolio backtests](multi_asset.md) — passing dict data
  and using the multi-asset Strategy API.
- [Walk-forward optimization](walk_forward.md) — sample-out parameter
  selection.
- [Stats glossary](stats_glossary.md) — what every metric means.
- [Trading environment](trading_environment.md) — the RL workflow.
- [FAQ](faq.md) — common pitfalls.
