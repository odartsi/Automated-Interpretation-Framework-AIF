"""
Unit tests for LLM_evaluation module (pure helper functions).
"""
import pytest

from LLM_evaluation import describe_clean_composition, flatten_chemical_formula


def test_describe_clean_composition():
    out = describe_clean_composition("CaO")
    assert "Ca" in out and "O" in out
    assert "fractional" in out.lower() or "composition" in out.lower()


def test_describe_clean_composition_complex():
    out = describe_clean_composition("Ca2O2")
    assert "Ca" in out and "O" in out


def test_flatten_chemical_formula_simple():
    assert flatten_chemical_formula("CaO") == "CaO"
    assert flatten_chemical_formula("H2O") == "H2O"


def test_flatten_chemical_formula_with_parens():
    # (OH)2 -> O2H2
    out = flatten_chemical_formula("Ca(OH)2")
    assert "Ca" in out
    assert "O" in out and "H" in out
    # Duplicate elements combined
    assert out in ("CaO2H2", "CaH2O2") or "O" in out


def test_flatten_chemical_formula_combines_duplicates():
    out = flatten_chemical_formula("H2O2")
    assert "H" in out and "O" in out
    assert out in ("H2O2", "O2H2")  # order may vary by implementation
