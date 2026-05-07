# FAQ

Common questions and pitfalls. If you hit something not covered here,
[open an issue](https://github.com/gguan/qtrade/issues).

## Backtesting

### Why does my strategy underperform Buy & Hold?

Most simple strategies do. Buy & Hold extracts the asset's underlying
trend with zero commission and zero timing risk. To beat it your
strategy needs to either:

1. **Avoid drawdowns** — sit out the worst periods (timing).
2. **Add diversification** — multi-asset reduces variance even with
   B&H-like returns per asset.
3. **Use leverage** — explicit via `margin_ratio` or position sizing.
4. **Trade with edge** — short-side gains, mean reversion, etc.

If your strategy has a positive Sharpe but loses to B&H, it might still
be valuable as a portfolio component (uncorrelated with passive long).

### How do I avoid look-ahead bias?

QTrade prevents the most common form by construction: `self.data` (and
`self.data_by_asset[asset]`) is sliced to `[:current_time]`, so you
*cannot* read future bars from inside `on_bar_close`.

Where look-ahead can still creep in:

- **In `prepare()`**, you have access to the full DataFrame. If you
  compute an indicator that uses future information (e.g.
  `df['Close'].shift(-1)`), `on_bar_close` will read it.
- **Limit / stop fills** assume the price actually traded at your
  level inside the bar. For thinly-traded assets this overstates
  fill probability.
- **`trade_on_close=True`** lets a market order fill at the current
  bar's close — the same bar your signal was computed from. For some
  signals this is realistic (close-of-bar signals), for others it's
  effectively cheating. If unsure, set `trade_on_close=False` and
  fills happen at the next bar's open.

### What's the difference between `trade_on_close=True` and `False`?

| Setting | Market order placed in `on_bar_close()` for bar N fills at... |
|---|---|
| `True` | Bar N's **close** (same bar) |
| `False` | Bar N+1's **open** |

`True` is convenient for backtests where your signal genuinely arrives
at end-of-bar (e.g. you compute on close prices). `False` is the
realistic setting if your signal could in practice take a tick to act
on — slightly conservative, recommended unless you have a reason.

### My strategy fires no trades. Why?

A few common causes:

1. **Indicators are NaN at the start** — the first `window_size` bars
   are warmup. `Backtest._start_idx` skips them automatically; if your
   data is shorter than the warmup, you get zero trades.
2. **`available_margin` is 0** — `margin_ratio < 1` lets you take
   leveraged positions, but `available_margin` shrinks fast as you
   stack trades. Lower the size or raise the ratio.
3. **`size=0` quietly does nothing** — `Order(size=0)` raises now (good!),
   but if you compute `size = available_margin // close` with too-small
   cash, default `Strategy.buy()` resolves to size 0 and Order
   construction errors.

### Why does my SL/TP fire at a price different from what I set?

The SL/TP triggers when the bar's `low` (for long stops) or `high` (for
long take-profits) crosses your level. The synthetic exit is recorded
*at your SL/TP level*, not at the bar's close. So if you set sl=99 and
the bar dips to low=98, your fill is recorded at 99, even though the
bar closed at 102.

> Caveat: ordinary stop *orders* (placed via `Order(stop=...)`) currently
> fill at the bar's close, not the trigger price — a known fidelity gap
> tracked as a future fix. SL/TP attached to a Trade work as described.

## Optimization

### Why does my strategy look great in `optimize()` but bad live?

In-sample overfitting. `optimize()` picks the params that look best on
the same data they're tested on — which is a textbook recipe for
fitting noise.

Use `walk_forward_optimize` instead — see
[Walk-forward optimization](walk_forward.md). Each parameter set is
evaluated only on data the strategy was never tuned on.

If walk-forward Sharpe is much lower than `optimize` Sharpe, that gap
is your overfit. Common; expected; treat the walk-forward number as
the realistic one.

### How do I pick the `maximize` metric?

- **Total Return [%]**: simple, but punishes nothing for risk.
  Strategies with huge drawdowns can win.
- **Sharpe Ratio**: most common default. Good for normal-ish return
  distributions.
- **Sortino Ratio**: better for asymmetric / fat-tailed returns
  (trend-followers).
- **Calmar Ratio**: rewards smooth equity curves; good if you care
  about drawdown experience.

Start with Sharpe unless you have a specific reason.

### My grid search is too slow.

Two paths:

1. **Reduce grid size** — many strategies have one or two truly
   important parameters; ranges of 4–6 values per param are usually
   enough. Don't sweep everything at fine resolution.
2. **Use vectorbt instead** — for thousands of param combinations,
   QTrade's event loop is the wrong tool. See
   [COMPARISON.md](https://github.com/gguan/qtrade/blob/main/COMPARISON.md).

## Multi-asset

### How do I align data from different sources?

QTrade requires every asset's DataFrame to share an identical
`DatetimeIndex`. Use a pandas join on the indexes before passing in:

```python
common = aapl.index.intersection(msft.index)
data = {"AAPL": aapl.loc[common], "MSFT": msft.loc[common]}
```

For data with gaps (corporate actions, illiquid days) you might prefer
union + forward-fill — but **be careful**: forward-filled bars give
the impression of liquidity that wasn't there.

### Can I have different commission per asset?

Not currently. `Commission` is a single instance shared across the
broker. To approximate per-asset cost, wrap your strategy logic to
add asset-specific slippage manually before placing orders.

### Why does `self.position` raise in multi-asset mode?

Because there's no single Position to return — you have one per asset.
Use `self.positions["AAPL"]` (dict access) instead. Same for
`self.data` → `self.data_by_asset["AAPL"]`.

## Reinforcement Learning

### My agent learns to do nothing (reward = 0).

The default `RewardScheme` only rewards realized PnL on bars where a
trade closed. If your agent never closes trades, reward stays 0 and
the gradient signal is too sparse.

Try one of:

- An equity-based reward that fires every bar (see
  [Customizing Trading Environment](customize_environment.md)).
- An action scheme that forces flat at episode end (so trades close
  for sure).
- A higher episode count — sparse reward + long episodes is hard.

### What does `info` contain in `env.step()` returns?

Per-step dict from `TradingEnv.step()`:

| Key | Meaning |
|---|---|
| `equity` | `broker.equity` |
| `unrealized_pnl` | `broker.unrealized_pnl` |
| `cumulative_return` | `broker.cumulative_returns` (multiplicative since start) |
| `position` | `broker.position.size` (signed) |
| `total_trades` | `len(broker.closed_trades)` |
| `trades_profit` | Sum of all closed trade profits |
| `avg_trade_duration` | Mean exit_index − entry_index in bars |
| `is_success` | `trades_profit > 0` |

`is_success` is intended for SB3's success-rate logging in
`EvalCallback`.

### `random_start=True` errors with a cryptic message.

Fixed in v0.3.0 — now raises `ValueError` if
`len(data) <= window_size + max_steps`. The default `max_steps=3000`
needs at least ~3000 bars of data; lower it for shorter datasets.

## Misc

### How do I save the backtest result for later?

There's no built-in serializer, but the broker holds everything. Pickle
works as a quick-and-dirty save:

```python
import pickle
with open("run.pkl", "wb") as f:
    pickle.dump(bt.broker, f)
```

For something less brittle (cross-version safe), pull what you need
into DataFrames first: `bt.broker.equity_history`,
`bt.get_trade_history()`, `calculate_stats(bt.broker)`.

### Where do I report bugs?

[https://github.com/gguan/qtrade/issues](https://github.com/gguan/qtrade/issues).
Minimal repro + the QTrade version (`qtrade-lib --version` doesn't
exist; use `pip show qtrade-lib`) helps a lot.
