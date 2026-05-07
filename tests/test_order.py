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

    with pytest.raises(ValueError, match=r"Order already filled."):
        order._fill(102.0, '2024-01-01')

    with pytest.raises(ValueError, match=r"Order already filled."):
        order._close(reason="Cannot reject a filled order.")


def test_order_reject_then_fill():
    order = Order(size=10, limit=None, stop=None, sl=95.0, tp=105.0, tag="Order1")
    order._close(reason="Insufficient margin")

    with pytest.raises(ValueError, match=r"Order already closed."):
        order._fill(102.0, '2024-01-01')


def test_order_zero_size_raises():
    with pytest.raises(AssertionError, match=r"Order size cannot be zero."):
        Order(size=0)


def test_order_is_long_and_is_short():
    assert Order(size=10).is_long is True
    assert Order(size=10).is_short is False
    assert Order(size=-5).is_long is False
    assert Order(size=-5).is_short is True


def test_order_repr_includes_non_none_fields():
    order = Order(size=10, limit=105.0, sl=95.0, tag='Entry')
    text = repr(order)
    assert text.startswith('<Order')
    assert 'Size=10' in text
    assert 'Limit=105.0' in text
    assert 'Sl=95.0' in text
    assert 'Tag=Entry' in text
    assert 'Stop=' not in text  # None values omitted
    assert 'Tp=' not in text


def test_order_close_then_close_raises():
    order = Order(size=5)
    order._close(reason='Insufficient margin')
    with pytest.raises(ValueError, match=r"Order already closed."):
        order._close(reason='Trying again')