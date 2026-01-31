"""
Unit tests for refinement_metrics module (pure functions only).
"""
import pytest

from refinement_metrics import normalize_rwp


def test_normalize_rwp_zero_rwp():
    # RWP 0 -> score 1
    assert normalize_rwp(0) == pytest.approx(1.0, rel=1e-5)


def test_normalize_rwp_forty():
    # RWP 40 -> score 0
    assert normalize_rwp(40) == pytest.approx(0.0, rel=1e-5)


def test_normalize_rwp_linear():
    # (rwp - 40) / (-40) => 20 -> 0.5
    assert normalize_rwp(20) == pytest.approx(0.5, rel=1e-5)


def test_normalize_rwp_above_forty():
    # RWP > 40 can go negative
    assert normalize_rwp(80) == pytest.approx(-1.0, rel=1e-5)
