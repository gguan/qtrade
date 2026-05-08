"""Contract specifications: bundle ``multiplier`` + ``margin_ratio`` per asset.

Use a built-in spec for common instruments (`STOCK_CASH`, `GC_COMEX`, `ES_CME`,
…) or define your own — `Contract` is just a small frozen dataclass:

.. code-block:: python

    from qtrade.contracts import Contract, STOCK_CASH, GC_COMEX

    # Custom spec for a Shanghai gold futures contract (1000g per lot)
    SHFE_AU = Contract(multiplier=1000, margin_ratio=0.08, name="SHFE Gold")

    bt = Backtest(data, MyStrategy, contracts={
        "AAPL":  STOCK_CASH,   # default if you omit the key entirely
        "GC=F":  GC_COMEX,
        "AU0":   SHFE_AU,
    })

Assets you don't list in ``contracts`` automatically resolve to
:data:`STOCK_CASH` (no leverage, multiplier 1.0) — so a pure-stock backtest
still works with no extra configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contract:
    """Specification for a tradable instrument.

    Args:
        multiplier: Dollar P&L per 1 unit of price movement per contract.
            ``1`` for a stock share, ``50`` for E-mini S&P, ``100`` for
            COMEX gold, ``1000`` for crude.
        margin_ratio: Initial margin required, expressed as a fraction of
            notional (``size × multiplier × price``). ``1.0`` for cash
            stock accounts (no leverage), ``0.5`` for Reg-T margin
            accounts, ``~0.05`` for typical futures.
        name: Optional human-readable label, surfaced in repr / docs.
    """

    multiplier: float = 1.0
    margin_ratio: float = 1.0
    name: str = ""

    def __repr__(self) -> str:
        if self.name:
            return f"Contract(multiplier={self.multiplier}, margin_ratio={self.margin_ratio}, name={self.name!r})"
        return f"Contract(multiplier={self.multiplier}, margin_ratio={self.margin_ratio})"


# ---------------------------------------------------------------------------
# Built-in specs for common instruments.
#
# ⚠️  These are CONVENTIONAL values, not authoritative ones. ``multiplier``
#     is fixed by the contract spec (and rarely changes — check the exchange
#     page if in doubt), but ``margin_ratio`` is set by your broker and the
#     exchange's SPAN parameters and floats over time (typically 4-12% for
#     index futures, 5-10% for metals, 8-15% for energy/ag during volatile
#     periods). Treat the values below as templates — verify against your
#     own account before trusting backtest leverage numbers.
#
# Override is just one line:
#
#     from dataclasses import replace
#     MY_GC = replace(GC_COMEX, margin_ratio=0.07)
#
# Or define from scratch:
#
#     MY_GC = Contract(multiplier=100, margin_ratio=0.07, name="My GC")
# ---------------------------------------------------------------------------

# Stocks
STOCK_CASH = Contract(name="Stock (cash account)")
"""Default: 1 share = 1 unit of price, no leverage.

This is what you get if you pass nothing to ``contracts=``.
"""

STOCK_REGT = Contract(margin_ratio=0.5, name="Stock (Reg-T margin)")
"""US Reg-T margin account: 2x leverage on equities."""

# CME equity index futures
ES_CME = Contract(multiplier=50, margin_ratio=0.05, name="E-mini S&P 500 (ES)")
NQ_CME = Contract(multiplier=20, margin_ratio=0.05, name="E-mini NASDAQ-100 (NQ)")
YM_CBOT = Contract(multiplier=5, margin_ratio=0.05, name="E-mini Dow (YM)")
RTY_CME = Contract(multiplier=50, margin_ratio=0.05, name="E-mini Russell 2000 (RTY)")
MES_CME = Contract(multiplier=5, margin_ratio=0.05, name="Micro E-mini S&P 500 (MES)")
MNQ_CME = Contract(multiplier=2, margin_ratio=0.05, name="Micro E-mini NASDAQ-100 (MNQ)")

# COMEX metals
GC_COMEX = Contract(multiplier=100, margin_ratio=0.05, name="COMEX Gold (GC)")
MGC_COMEX = Contract(multiplier=10, margin_ratio=0.05, name="COMEX Micro Gold (MGC)")
SI_COMEX = Contract(multiplier=5000, margin_ratio=0.05, name="COMEX Silver (SI)")
HG_COMEX = Contract(multiplier=25000, margin_ratio=0.05, name="COMEX Copper (HG)")

# NYMEX energy
CL_NYMEX = Contract(multiplier=1000, margin_ratio=0.10, name="NYMEX Crude Oil (CL)")
NG_NYMEX = Contract(multiplier=10000, margin_ratio=0.10, name="NYMEX Natural Gas (NG)")
RB_NYMEX = Contract(multiplier=42000, margin_ratio=0.10, name="NYMEX RBOB Gasoline (RB)")
HO_NYMEX = Contract(multiplier=42000, margin_ratio=0.10, name="NYMEX Heating Oil (HO)")

# CBOT agriculture
ZC_CBOT = Contract(multiplier=50, margin_ratio=0.07, name="CBOT Corn (ZC)")
ZS_CBOT = Contract(multiplier=50, margin_ratio=0.07, name="CBOT Soybeans (ZS)")
ZW_CBOT = Contract(multiplier=50, margin_ratio=0.07, name="CBOT Wheat (ZW)")


__all__ = [
    "CL_NYMEX",
    "ES_CME",
    "GC_COMEX",
    "HG_COMEX",
    "HO_NYMEX",
    "MES_CME",
    "MGC_COMEX",
    "MNQ_CME",
    "NG_NYMEX",
    "NQ_CME",
    "RB_NYMEX",
    "RTY_CME",
    "SI_COMEX",
    "STOCK_CASH",
    "STOCK_REGT",
    "YM_CBOT",
    "ZC_CBOT",
    "ZS_CBOT",
    "ZW_CBOT",
    "Contract",
]
