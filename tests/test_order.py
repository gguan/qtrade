# tests/test_order.py

import pytest
from qtrade.core.order import Order


def test_order_initialization():
    order = Order(size=10, limit=105.0, stop=95.0, sl=95.0, tp=105.0, tag="Order1")
    assert order.size == 10
    assert order.limit == 105.0
    assert order.stop == 95.0
    assert order.sl == 95.0
    assert order.tp == 105.0
    assert order.tag == "Order1"
    assert order.is_filled is False
    assert order.fill_price is None
    assert order.fill_date is None


def test_order_fill():
    order = Order(size=10, limit=None, stop=None, sl=95.0, tp=105.0, tag="Order1")
    fill_price = 102.0
    fill_date = '2024-01-01'

    order._fill(fill_price, fill_date)

    assert order.is_filled is True
    assert order.fill_price == fill_price
    assert order.fill_date == fill_date


def test_order_reject():
    order = Order(size=10, limit=None, stop=None, sl=95.0, tp=105.0, tag="Order1")
    reject_reason = "Insufficient margin"

    order._close(reason=reject_reason)

    assert order._close_reason == reject_reason


def test_order_fill_then_reject():
    order = Order(size=10, limit=None, stop=None, sl=95.0, tp=105.0, tag="Order1")
    order._fill(102.0, '2024-01-01')

    with pytest.raises(Exception, match="Order already filled."):
        order._fill(102.0, '2024-01-01')

    with pytest.raises(Exception, match="Order already filled."):
        order._close(reason="Cannot reject a filled order.")


def test_order_reject_then_fill():
    order = Order(size=10, limit=None, stop=None, sl=95.0, tp=105.0, tag="Order1")
    order._close(reason="Insufficient margin")

    with pytest.raises(Exception, match="Order already closed."):
        order._fill(102.0, '2024-01-01')