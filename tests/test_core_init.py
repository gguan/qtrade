
def test_core_module_imports():
    from qtrade.core import (
        Order,
        Trade,
        Position,
        Broker,
        Commission,
        PercentageCommission,
        FixedCommission,
        SlippageCommission
    )
    # Ensure that all modules are correctly imported
    assert Order is not None
    assert Trade is not None
    assert Position is not None
    assert Broker is not None
    assert Commission is not None
    assert PercentageCommission is not None
    assert FixedCommission is not None
    assert SlippageCommission is not None