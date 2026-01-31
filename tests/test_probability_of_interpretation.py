"""
Unit tests for probability_of_interpretation module (pure helpers).
_infer_group_from_search depends on VALID_KEYS from the module (e.g. ["TRI"]).
"""
import pytest

# Import module to get _infer_group_from_search and VALID_KEYS
import probability_of_interpretation as poi


def test_infer_group_from_search_prefix():
    assert poi._infer_group_from_search("TRI-15") == "TRI"
    assert poi._infer_group_from_search("tri-197") == "TRI"


def test_infer_group_from_search_contains():
    assert poi._infer_group_from_search("something TRI something") == "TRI"
    assert poi._infer_group_from_search("TRI") == "TRI"


def test_infer_group_from_search_no_match():
    assert poi._infer_group_from_search("xyz") is None
    assert poi._infer_group_from_search("PG_123") is None


def test_infer_group_from_search_strips_whitespace():
    assert poi._infer_group_from_search("  TRI-15  ") == "TRI"
