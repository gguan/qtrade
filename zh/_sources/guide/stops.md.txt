# Stops: SL, TP, and trailing

QTrade supports three flavors of position-exit logic:

1. **Static SL / TP** — fixed levels set at order time. Fires when
   `low <= sl` (long) or `high >= tp` (long); symmetric for shorts.
2. **Mid-trade modification** — change SL / TP on an open Trade and the
   next bar checks against the new level.
3. **Trailing stop** — Broker auto-bumps the SL each bar based on the
   high-water (long) / low-water (short) mark. The stop only ever
   ratchets in the trade's favor.

## Static SL / TP at order time

```python
self.buy(size=10, sl=95.0, tp=110.0)
```

If the next bar's low touches 95 the trade exits at 95 (`exit_reason='sl'`);
if the high touches 110 it exits at 110 (`exit_reason='tp'`). SL is checked
before TP within a single bar.

## Modify SL / TP mid-trade

`Trade.sl` and `Trade.tp` are now settable, so you can ratchet a stop after
a profit threshold or shift a target further out:

```python
class RatchetingStop(Strategy):
    def on_bar_close(self):
        for trade in self.active_trades:
            if trade.is_long and self.data['Close'].iloc[-1] > trade.entry_price * 1.05:
                # Up 5% — move SL to breakeven, lock in zero loss
                trade.sl = trade.entry_price
```

For both at once, use `update_exit_levels()`. It uses a sentinel so the
difference between "leave alone" and "clear" is unambiguous:

```python
trade.update_exit_levels(sl=110)         # bump SL only
trade.update_exit_levels(tp=None)        # explicitly CLEAR the TP
trade.update_exit_levels(sl=110, tp=130) # both at once
```

The Broker's per-bar SL/TP check reads the current values fresh, so any
mutation in `on_bar_close()` takes effect immediately on the next bar.

## Trailing stop

For the standard "lock in profit as price runs in your favor" pattern, use
the Broker-managed trailing stop instead of writing the bookkeeping yourself:

```python
self.buy(size=10, trail_percent=0.05)    # 5% trailing stop
self.sell(size=10, trail_amount=2.0)     # $2 trailing stop on a short
```

How it works:

- **Long**: each bar, `trail_high = max(trail_high, bar_high)`. The SL is
  recomputed as `trail_high * (1 - trail_percent)` (or `trail_high - trail_amount`).
  The stop only moves up.
- **Short**: symmetric. `trail_low = min(trail_low, bar_low)`, SL =
  `trail_low * (1 + trail_percent)`. Only moves down.

The `trail_high` / `trail_low` is seeded with `entry_price` at trade open,
so the initial trail SL is already in effect on bar 1.

### Trail + explicit SL together

You can pass both. Whichever is **tighter** (better for the trader) wins,
and the trail still only ratchets in your favor:

```python
self.buy(size=10, sl=98.0, trail_percent=0.05)
```

- Initial trail-implied SL = `entry * 0.95 = 95`. Explicit `sl=98` is
  tighter → effective SL is 98.
- Price climbs from 100 to 110: trail-implied = 104.5. Now tighter than
  98 → effective SL becomes 104.5.

### Validation

`trail_percent` and `trail_amount` are mutually exclusive and must be > 0.
Both are **keyword-only** parameters on `Order`, `Trade`, `Strategy.buy`,
and `Strategy.sell`.

```python
self.buy(size=10, trail_percent=0.05, trail_amount=2.0)  # ValueError
self.buy(size=10, trail_percent=0)                       # ValueError
```

## When the trail update fires (timing)

Within the per-bar pipeline:

```
process_bar(ts):
    process_executing_orders   # market orders fill at this bar's open
    check_sl_tp                # exit triggers using the OLD SL level
    update_trailing_stops      # bump SL using THIS bar's high/low
    process_pending_orders     # stop / limit orders triggered today
```

The trail update runs *after* the SL check, so the new tighter level
applies starting on the **next** bar. This is the conservative bar-level
convention — within one OHLC bar we don't know if the low (which would
trigger the OLD SL) came before or after the high (which would tighten
the trail). Assuming the unfavorable extreme came first matches how
`__check_sl_tp` already treats single-bar extremes.

## Inspecting trail state after exit

The `Trade` record preserves trailing metadata even after the position
closes — useful for post-run analysis:

```python
for t in bt.broker.closed_trades:
    if t.trail_percent is not None:
        print(f"Trail-stopped at {t.exit_price}, peak was {t.trail_high}")
```

## What's NOT modeled

- **Trailing TP / take-profit ratcheting.** Trail only manages SL. To
  ratchet a TP you'd write the loop yourself with `update_exit_levels(tp=...)`.
- **Volatility-based trails (chandelier exit, ATR stop).** Need ATR data,
  which you'd compute in `prepare()` and apply via mid-trade `trade.sl = ...`.
- **Time-in-trade exits.** Compare `self.current_time - trade.entry_date`
  in `on_bar_close()` and call `self.close()` if too long.
- **Maker-side stop orders.** All exits fill at the trigger price (or
  worse on a gap, for pure stop orders). No queue-position modeling.
