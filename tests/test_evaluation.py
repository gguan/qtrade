"""Smoke test for qtrade.utils.evaluation. Skipped when stable-baselines3 is not installed."""

import pytest

pytest.importorskip("stable_baselines3")


def test_evaluation_module_imports():
    from qtrade.utils.evaluation import EvalWithInfoCallback, evaluate_policy_with_infos
    assert EvalWithInfoCallback is not None
    assert callable(evaluate_policy_with_infos)
