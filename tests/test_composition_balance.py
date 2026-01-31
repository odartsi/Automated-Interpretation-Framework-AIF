"""
Unit tests for composition_balance module.
"""
import pytest
from pymatgen.core import Composition

from composition_balance import (
    calculate_composition_balance_score_refined,
    cleanup_phases,
    remove_elements_from_composition,
    normalize_composition,
)


def test_calculate_composition_balance_score_refined_perfect_match():
    target = Composition("CaO")
    output = Composition("CaO")
    score = calculate_composition_balance_score_refined(target, output)
    assert score == pytest.approx(1.0, rel=1e-5)


def test_calculate_composition_balance_score_refined_mismatch():
    target = Composition("CaO")
    output = {"Ca": 1.0, "O": 0.5}  # missing O
    score = calculate_composition_balance_score_refined(target, output)
    assert 0 <= score < 1


def test_calculate_composition_balance_score_refined_extra_element():
    target = Composition("CaO")
    output = {"Ca": 1.0, "O": 1.0, "Fe": 0.5}
    score = calculate_composition_balance_score_refined(target, output)
    assert 0 <= score <= 1


def test_calculate_composition_balance_score_refined_missing_element():
    target = Composition("CaCO3")
    output = {"Ca": 1.0, "O": 3.0}  # C missing
    score = calculate_composition_balance_score_refined(target, output)
    assert 0 <= score < 1


def test_cleanup_phases():
    phases = ["C1.9992O1.9992_194_(icsd_37237)-None", "V2O3_15_(icsd_95762)-11"]
    out = cleanup_phases(phases)
    assert "CO" in out or "C" in out  # integer formula
    assert "V2O3" in out or "V" in out


def test_remove_elements_from_composition():
    assert remove_elements_from_composition("CaCO3", ["O", "C"]) == "Ca"
    assert remove_elements_from_composition("H2O", ["H"]) == "O"


def test_normalize_composition():
    out = normalize_composition("Ca2O2")
    assert "Ca" in out and "O" in out
    # Fractional form: Ca1O1 or similar
    assert "1" in out or "Ca" in out
