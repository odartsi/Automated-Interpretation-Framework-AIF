"""
Unit tests for utils module (pure functions that do not require dara/refinement).
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import after conftest has added src to path
from utils import (
    load_json,
    load_csv,
    safe_pick_target,
    extract_project_number_from_filename,
    get_output_dir,
    celsius_to_kelvin,
    scaled_sigmoid,
    normalize_rwp_for_sample,
    normalize_scores_for_sample,
    calculate_fit_quality,
    calculate_prior_probability,
    calculate_posterior_probability_of_interpretation,
    calculate_phase_probabilities,
    extract_icsd_from_key,
    strip_phase_identifier,
    remove_cifs_prefix,
    remove_cifs_suffix,
    type_of_furnace,
)


# ----- load_json / load_csv -----
def test_load_json_missing_raises():
    with pytest.raises(FileNotFoundError, match="Missing JSON"):
        load_json("/nonexistent/path.json")


def test_load_json_success(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}')
    assert load_json(p) == {"a": 1}


def test_load_csv_missing_raises():
    with pytest.raises(FileNotFoundError, match="Missing CSV"):
        load_csv("/nonexistent/path.csv")


def test_load_csv_success(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("x,y\n1,2\n3,4")
    df = load_csv(p)
    assert len(df) == 2
    assert list(df.columns) == ["x", "y"]


# ----- safe_pick_target -----
def test_safe_pick_target_returns_first_non_null():
    df = pd.DataFrame({"Target": ["LiVO2"], "Name": ["TRI_1"]})
    assert safe_pick_target(df) == "LiVO2"


def test_safe_pick_target_uses_alternative_columns():
    df = pd.DataFrame({"Target.1": ["CaVO2"]})
    assert safe_pick_target(df) == "CaVO2"


def test_safe_pick_target_returns_none_when_all_null():
    df = pd.DataFrame({"Target": [None, None], "Name": ["a", "b"]})
    assert safe_pick_target(df) is None


def test_safe_pick_target_returns_none_when_no_target_column():
    df = pd.DataFrame({"Name": ["a"]})
    assert safe_pick_target(df) is None


# ----- extract_project_number_from_filename -----
def test_extract_project_number_from_filename_pg_style():
    assert extract_project_number_from_filename("PG_0106_1_Ag2O_Bi2O3_200C_60min_uuid") == "PG_0106_1"
    assert extract_project_number_from_filename("PG_2547-1_Ag2O_something") == "PG_2547_1"


def test_extract_project_number_from_filename_fallback():
    assert extract_project_number_from_filename("TRI_197") == "TRI_197"
    assert extract_project_number_from_filename("abc_def_ghi") == "abc_def"


def test_extract_project_number_from_filename_single_part():
    assert extract_project_number_from_filename("only") == "only"


# ----- get_output_dir -----
def test_get_output_dir():
    out = get_output_dir("CaVO2", "TRI_197")
    assert "TRI" in out
    assert "CaVO2" in out
    assert "TRI_197" in out


# ----- celsius_to_kelvin -----
def test_celsius_to_kelvin():
    assert celsius_to_kelvin(0) == "273.15"
    assert celsius_to_kelvin(25) == "298.15"
    assert celsius_to_kelvin(-273.15) == "0.0"


# ----- scaled_sigmoid -----
def test_scaled_sigmoid_bounds():
    # At score=1 the sigmoid should map to 1 (by design)
    assert abs(scaled_sigmoid(1.0) - 1.0) < 1e-6
    # At center, value is around 0.5 (scaled)
    assert 0 < scaled_sigmoid(0.2) < 1


def test_scaled_sigmoid_monotonic():
    x = [0.0, 0.2, 0.5, 0.8, 1.0]
    y = [scaled_sigmoid(v) for v in x]
    assert y == sorted(y)


# ----- normalize_rwp_for_sample -----
def test_normalize_rwp_for_sample():
    interps = {
        "I_1": {"rwp": 20},
        "I_2": {"rwp": 40},
        "I_3": {"rwp": 60},
    }
    out = normalize_rwp_for_sample(interps, max_rwp=60)
    assert out["I_1"]["normalized_rwp"] == pytest.approx(2/3, rel=1e-5)  # (60-20)/60
    assert out["I_2"]["normalized_rwp"] == pytest.approx(1/3, rel=1e-5)
    assert out["I_3"]["normalized_rwp"] == pytest.approx(0.0, rel=1e-5)


def test_normalize_rwp_for_sample_none_rwp():
    interps = {"I_1": {"rwp": None}}
    out = normalize_rwp_for_sample(interps)
    assert out["I_1"]["normalized_rwp"] is None


def test_normalize_rwp_for_sample_raises_on_non_dict():
    with pytest.raises(ValueError, match="Expected a dict"):
        normalize_rwp_for_sample([])


# ----- normalize_scores_for_sample -----
def test_normalize_scores_for_sample():
    interps = {"I_1": {"score": 0.5}, "I_2": {"score": 1.0}}
    out = normalize_scores_for_sample(interps)
    assert 0 <= out["I_1"]["normalized_score"] <= 1
    assert abs(out["I_2"]["normalized_score"] - 1.0) < 1e-5


def test_normalize_scores_for_sample_raises_on_non_dict():
    with pytest.raises(ValueError, match="Expected a dict"):
        normalize_scores_for_sample([])


# ----- calculate_fit_quality -----
def test_calculate_fit_quality():
    interps = {
        "I_1": {"normalized_rwp": 0.8, "normalized_score": 0.6},
        "I_2": {"normalized_rwp": 0.5, "normalized_score": 0.5},
    }
    out = calculate_fit_quality(interps, w_rwp=1, w_score=1)
    assert out["I_1"]["fit_quality"] == pytest.approx(0.7, rel=1e-5)
    assert out["I_2"]["fit_quality"] == pytest.approx(0.5, rel=1e-5)


# ----- calculate_prior_probability -----
def test_calculate_prior_probability():
    interps = {
        "I_1": {"LLM_interpretation_likelihood": 0.5, "balance_score": 0.5},
        "I_2": {"LLM_interpretation_likelihood": 1.0, "balance_score": 1.0},
    }
    out = calculate_prior_probability(interps, w_llm=1, w_bscore=1)
    assert out["I_1"]["prior_probability"] == pytest.approx(0.5, rel=1e-5)
    assert out["I_2"]["prior_probability"] == pytest.approx(1.0, rel=1e-5)


# ----- calculate_posterior_probability_of_interpretation -----
def test_calculate_posterior_probability_of_interpretation():
    interps = {
        "I_1": {"prior_probability": 0.5, "fit_quality": 0.4, "trustworthy": True},
        "I_2": {"prior_probability": 0.5, "fit_quality": 0.6, "trustworthy": True},
    }
    out = calculate_posterior_probability_of_interpretation(interps)
    assert "posterior_probability" in out["I_1"]
    total = out["I_1"]["posterior_probability"] + out["I_2"]["posterior_probability"]
    assert abs(total - 1.0) < 1e-5


def test_calculate_posterior_probability_trust_penalty():
    interps = {
        "I_1": {"prior_probability": 0.5, "fit_quality": 0.5, "trustworthy": True},
        "I_2": {"prior_probability": 0.5, "fit_quality": 0.5, "trustworthy": False},
    }
    out = calculate_posterior_probability_of_interpretation(interps)
    assert out["I_1"]["posterior_probability"] > out["I_2"]["posterior_probability"]


def test_calculate_posterior_probability_raises_on_non_dict():
    with pytest.raises(ValueError, match="dictionary"):
        calculate_posterior_probability_of_interpretation("not a dict")


# ----- calculate_phase_probabilities -----
def test_calculate_phase_probabilities():
    interps = {
        "I_1": {"posterior_probability": 0.6, "phases": ["A", "B"]},
        "I_2": {"posterior_probability": 0.4, "phases": ["B", "C"]},
    }
    out = calculate_phase_probabilities(interps)
    assert out["A"] == pytest.approx(60.0, rel=1e-5)
    assert out["B"] == pytest.approx(100.0, rel=1e-5)
    assert out["C"] == pytest.approx(40.0, rel=1e-5)


def test_calculate_phase_probabilities_raises_on_non_dict():
    with pytest.raises(ValueError, match="dictionary"):
        calculate_phase_probabilities([])


# ----- extract_icsd_from_key -----
def test_extract_icsd_from_key():
    assert extract_icsd_from_key("ZrO2_14_(icsd_157403)-0") == "157403"
    assert extract_icsd_from_key("Zn1.96O2_186_(icsd_13952)-None") == "13952"
    assert extract_icsd_from_key("phase_icsd-999") == "999"
    assert extract_icsd_from_key("NoIcsdHere") is None


# ----- strip_phase_identifier -----
def test_strip_phase_identifier():
    assert strip_phase_identifier("V2O3_167_(icsd_1869)-0") == "V2O3_167"
    assert strip_phase_identifier("CaO_1") == "CaO_1"


# ----- remove_cifs_prefix / remove_cifs_suffix -----
def test_remove_cifs_prefix():
    assert remove_cifs_prefix(["cifs/A.cif", "cifs/B.cif"]) == ["A.cif", "B.cif"]


def test_remove_cifs_suffix():
    assert remove_cifs_suffix(["cifs/A.cif", "cifs/B.cif"]) == ["cifs/A", "cifs/B"]


# ----- type_of_furnace -----
def test_type_of_furnace():
    assert "Box furnace" in type_of_furnace("BF")
    assert "Argon" in type_of_furnace("TF-Ar")
    assert "Oxygen" in type_of_furnace("TF-O2")
    assert type_of_furnace("unknown") == "unknown"
