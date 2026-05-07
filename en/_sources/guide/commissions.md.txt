# Commissions and slippage

QTrade ships four `Commission` implementations covering the patterns
most strategies need. All live in `qtrade.core.commission` and share
the abstract base `Commission`.

```python
from qtrade.core import (
    NoCommission,            # default: 0
    PercentageCommission,    # X% of order value
    FixedCommission,         # flat fee per order
    SlippageCommission,      # X% slippage on order value
)
```

Pass an instance to `Backtest(..., commission=...)` or
`TradingEnv(..., commission=...)`. `commission=None` is equivalent to
`NoCommission()`.

## The interface

`Commission.calculate_commission(order_size: int, fill_price: float) -> float`

Returns the commission in account currency for a single fill. The
Broker subtracts it from `cash` at fill time. Sign of `order_size` is
preserved — use `abs(order_size)` inside if you want symmetric fees on
buys and sells.

To roll your own:

```python
from qtrade.core.commission import Commission

class TieredCommission(Commission):
    """0.1% under $10k, 0.05% above (per fill)."""

    def calculate_commission(self, order_size, fill_price):
        notional = abs(order_size) * fill_price
        rate = 0.001 if notional < 10_000 else 0.0005
        return notional * rate
```

## Built-in implementations

### NoCommission

```python
NoCommission()
```

Returns `0.0` always. The default if you don't pass anything. Use it
for clean comparisons against Buy & Hold, or when modeling commission-free
brokers.

### PercentageCommission

```python
PercentageCommission(percentage=0.001)   # 0.1%
```

Charges `abs(order_size) * fill_price * percentage`.

The standard model for **stocks** (commissions quoted as bps of notional)
and **crypto spot** (taker fees are typically 0.05–0.1%).

### FixedCommission

```python
FixedCommission(fixed_fee=1.0)   # $1 per fill
```

Returns a flat fee regardless of size or price. Realistic for **futures**
(where the exchange charges per-contract) and for **discount brokers**
that quote a flat ticket charge.

For futures with multiple commission components (exchange + clearing +
broker), sum them up: `FixedCommission(fixed_fee=2.50)`.

### SlippageCommission

```python
SlippageCommission(slippage_percentage=0.0005)   # 5 bps
```

Mathematically identical to `PercentageCommission`, but the intent is
different — this models the *price impact* of crossing the spread,
not a venue fee. Use it when:

- You want to add slippage **on top of** an explicit commission.
- You want the cost to scale with notional but separate from "fees" in
  reporting.

```python
# Crypto spot: 0.1% taker fee + 0.05% slippage
class CryptoCost(Commission):
    def __init__(self):
        self.fee = PercentageCommission(0.001)
        self.slip = SlippageCommission(0.0005)
    def calculate_commission(self, order_size, fill_price):
        return self.fee.calculate_commission(order_size, fill_price) \
             + self.slip.calculate_commission(order_size, fill_price)
```

## Picking the right model

| Asset class | Typical setup |
|---|---|
| US equities (retail) | `NoCommission()` (most brokers are zero now) + small `SlippageCommission` |
| US equities (institutional) | `PercentageCommission(0.0005)` (~5 bps blended) |
| Futures | `FixedCommission(2.5)` per contract + slippage |
| FX | `PercentageCommission(0.0001)` (1 pip on majors) |
| Crypto spot | `PercentageCommission(0.001)` (taker) or `(0.0005)` (maker) |
| Crypto perps | `PercentageCommission(0.0005)` taker + funding (modeled separately) |

If unsure, **run the backtest both with and without commission**. The
gap tells you how cost-sensitive your strategy is — high-turnover
strategies often look great pre-cost and lose money post-cost.

## What's not modeled

- **Maker/taker asymmetry.** All fills are charged at one rate. To
  model maker fees on limit orders, write a custom `Commission` and
  branch on whether the order has a `limit` price.
- **Spread.** The Broker fills market orders at `Close` (or next
  `Open`). The bid/ask spread isn't modeled — `SlippageCommission` is
  the closest approximation.
- **Funding rates / borrow costs.** Not modeled. For long-running short
  positions or perp funding, you'd need to subtract from `cash` manually
  in your strategy's `on_bar_close`.
- **Per-asset commission.** A single `Commission` instance applies to
  all assets in a multi-asset backtest. If you need different rates
  per asset, write a custom `Commission` that branches on the order's
  `_fill_price` or wrap it in your strategy logic.
