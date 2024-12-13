# tests/test_commission.py

import pytest
from qtrade.core.commission import (
    NoCommission,
    PercentageCommission,
    FixedCommission,
    SlippageCommission,
)


def test_no_commission():
    commission = NoCommission()
    assert commission.calculate_commission(order_size=100, fill_price=50.0) == 0.0
    assert commission.calculate_commission(order_size=-50, fill_price=100.0) == 0.0
    assert repr(commission) == "NoCommission()"


def test_percentage_commission():
    percentage = 0.001  # 0.1%
    commission = PercentageCommission(percentage=percentage)

    order_size = 100
    fill_price = 50.0
    expected_commission = abs(order_size * fill_price * percentage)
    assert commission.calculate_commission(order_size, fill_price) == expected_commission

    order_size = -200
    fill_price = 75.0
    expected_commission = abs(order_size * fill_price * percentage)
    assert commission.calculate_commission(order_size, fill_price) == expected_commission

    assert commission.percentage == percentage
    assert repr(commission) == f"PercentageCommission(percentage={percentage})"


def test_percentage_commission_negative_percentage():
    with pytest.raises(ValueError, match="Percentage cannot be negative."):
        PercentageCommission(percentage=-0.001)


def test_fixed_commission():
    fixed_fee = 10.0
    commission = FixedCommission(fixed_fee=fixed_fee)

    assert commission.calculate_commission(order_size=100, fill_price=50.0) == fixed_fee
    assert commission.calculate_commission(order_size=-50, fill_price=100.0) == fixed_fee

    assert commission.fixed_fee == fixed_fee
    assert repr(commission) == f"FixedCommission(fixed_fee={fixed_fee})"


def test_fixed_commission_negative_fee():
    with pytest.raises(ValueError, match="Fixed fee cannot be negative."):
        FixedCommission(fixed_fee=-10.0)


def test_slippage_commission():
    slippage_percentage = 0.002  # 0.2%
    commission = SlippageCommission(slippage_percentage=slippage_percentage)

    order_size = 100
    fill_price = 50.0
    expected_commission = abs(order_size * fill_price * slippage_percentage)
    assert commission.calculate_commission(order_size, fill_price) == expected_commission

    order_size = -200
    fill_price = 75.0
    expected_commission = abs(order_size * fill_price * slippage_percentage)
    assert commission.calculate_commission(order_size, fill_price) == expected_commission

    assert commission.slippage_percentage == slippage_percentage
    assert repr(commission) == f"SlippageCommission(slippage_percentage={slippage_percentage})"


def test_slippage_commission_negative_percentage():
    with pytest.raises(ValueError, match="Slippage percentage cannot be negative."):
        SlippageCommission(slippage_percentage=-0.002)