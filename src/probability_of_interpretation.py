from refinement_metrics import calculate_spectrum_likelihood_given_interpretation_wrapper
from LLM_evaluation import evaluate_interpretations_with_llm
from composition_balance import calculate_chemical_factors
import time 
import pandas as pd
import json
import os
import logging
from pathlib import Path
from typing import Optional
import re
from utils import (
    plot_interpretation_probabilities_with_statistical,
    calculate_posterior_probability_of_interpretation,
    calculate_prior_probability,
    plot_phase_and_interpretation_probabilities_newstyle,
    plot_phase_and_interpretation_unnormalized_probabilities_newstyle,
    calculate_fit_quality,
    normalize_scores_for_sample,
    flag_interpretation_trustworthiness,
    compute_trust_score,
    normalize_rwp_for_sample,
    plot_contribution_decomposition,
    plot_contribution_decomposition_dual,
    plot_contribution_decomposition_dual_normalized_right_v2,
    plot_contribution_pie_scaled,
    plot_contribution_decomposition_dual_normalized_right_v3,
    plot_contribution_decomposition_dual_normalized_right_v4,
)



# --- configure logging once (console + file) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("run_errors.log", mode="w")]
)

def safe_pick_target(filtered_df: pd.DataFrame) -> Optional[str]:
    """Return a usable target column if present, else None."""
    for col in ["Target", "Target.1", "target", "target_1"]:
        if col in filtered_df.columns and not filtered_df[col].isna().all():
            return filtered_df[col].iloc[0]
    return None
    
SAMPLE_PREFIX_RE = re.compile(r'^(PG_\d+(?:[-_]\d+))_')  
# captures 'PG_0106_1' or 'PG_0106-1' right before the next underscore

def extract_project_number_from_filename(base_name: str) -> str:
    """
    base_name: filename without extension
      e.g. 'PG_0106_1_Ag2O_Bi2O3_200C_60min_uuid'
           'PG_2547-1_Ag2O_...'
           'PG_0750_1_uuid'
    returns underscore-style project_number, e.g. 'PG_0106_1'
    """
    m = SAMPLE_PREFIX_RE.match(base_name)
    if m:
        # normalize to underscore style for CSV join
        return m.group(1).replace('-', '_')
    # fallback: if nothing matches, keep your old behavior
    parts = base_name.split('_')
    if len(parts) >= 2:
        return parts[0] + '_' + parts[1]
    return base_name

# -----------------------------
# Dataset registry 
# -----------------------------
DATASETS = {
    "TRI": {
        "csv": "../data/alab_synthesis_data/synthesis_TRI.csv",
        "combos": "../data/xrd_data/combinations.json",
        "interpretations":"../data/xrd_data/interpretations/interpretations_for_brier.json" # tri-80"../data/xrd_data/interpretations/interpretations_test.json"#tri-197: "../data/xrd_data/interpretations/interpretations_for_brier.json",#
    },
    "MINES": {
        "csv": "../data/alab_synthesis_data/synthesis_MINES.csv",
        "combos": "../data/alab_synthesis_data/composition_MINES.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_16_more_forevaluation.json",
    },
    "ARR": {
        "csv": "../data/alab_synthesis_data/synthesis_ARR.csv",
        "combos": "../data/alab_synthesis_data/composition_ARR.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_for_brier.json",
    },
    "GENOME": {
        "csv": "../data/alab_synthesis_data/synthesis_PG_genome.csv",
        "combos": "../data/alab_synthesis_data/composition_PG_genome.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_for_brier.json",
    },
    "OPXRD": {
        "csv": "../data/xrd_data/opXRD/synthesis_opxrd_full_oly.csv",
        "combos": "../data/xrd_data/opXRD/combinations_opxrd_oly.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_16_more_forevaluation.json",
    },
    # "GENOME_LAUREN": {
    #     "csv": "../data/xrd_data/opXRD/Genome_Lauren/genome_lauren_metadata.csv",
    #     "combos": "../data/xrd_data/opXRD/Genome_Lauren/combinations_genome_lauren.json",
    #     "interpretations": "../data/xrd_data/interpretations/interpretations_genome_lauren.json",
    # },
    "GENOME_LAUREN": {
        "csv": "../data/xrd_data/opXRD/Genome_Lauren/genome_lauren_metadata_false_only.csv",
        "combos": "../data/xrd_data/opXRD/Genome_Lauren/combinations_genome_lauren_false_only_2.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_genome_lauren_false.json",
    },
     "GENOME_LAUREN_NEW": {
        "csv": "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/opXRD/Genome_Lauren/lauren_ge900_true.csv",
        "combos": "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/opXRD/Genome_Lauren/lauren_ge900_true.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_genome_lauren_ge900_true.json",
    },
    
    "RRUFF": {
        "csv": "../data/xrd_data/RRUFF/metadata_rruff_replaced.csv",
        "combos": "../data/xrd_data/RRUFF/combinations_rruff_selection.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_rruff_new.json",
    },
    "CNRS":{
        "csv": "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/opXRD/opxrd/CNRS/summary.csv",
        "combos": "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/opXRD/opxrd/CNRS/pattern_index.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_opXRD_CNRS.json",
    }
}

VALID_KEYS = list(DATASETS.keys())

def _read_json(path):
    """Load JSON from a file path; raises FileNotFoundError if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r") as f:
        return json.load(f)

def _read_csv(path):
    """Load CSV from a file path; raises FileNotFoundError if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)

def _infer_group_from_search(term: str) -> str | None:
    """Infer dataset group from a search term (e.g. 'TRI-15' -> 'TRI'). Returns None if no match."""
    t = term.strip().upper()
    # Strong heuristic: prefix like "TRI-15"
    for key in VALID_KEYS:
        if t.startswith(f"{key}-"):
            return key
    # Fallback: contains keyword anywhere
    for key in VALID_KEYS:
        if key in t:
            return key
    return None

def _choose_group_interactively(prompt_text="Choose dataset group"):
    """Prompt the user to pick a dataset group; raises ValueError if the input is not in DATASETS."""
    print(f"{prompt_text} [{', '.join(VALID_KEYS)}]: ", end="")
    g = input().strip().upper()
    if g not in DATASETS:
        raise ValueError(f"Invalid group '{g}'. Must be one of: {', '.join(VALID_KEYS)}")
    return g

# -----------------------------
# Interactive flow
# -----------------------------
choice = input("Run (a)ll patterns or (s)elect specific ones? (a/s): ").strip().lower()
if choice not in {"a", "s"}:
    raise ValueError("Please answer 'a' or 's'.")

if choice == "a":
    group = _choose_group_interactively("Which dataset group to run")
    cfg = DATASETS[group]
    df = _read_csv(cfg["csv"])
    all_combinations = _read_json(cfg["combos"])
    combinations = all_combinations
    interpretations_file = cfg["interpretations"]

else:
    search_term = input("Enter part of the pattern_path to filter (e.g., 'TRI-15'): ").strip()
    # Try to infer the group from the search term; if ambiguous, ask.
    group = _infer_group_from_search(search_term)
    if group is None:
        group = _choose_group_interactively("Could not infer dataset group. Please specify")

    cfg = DATASETS[group]
    df = _read_csv(cfg["csv"])
    all_combinations = _read_json(cfg["combos"])
    interpretations_file = cfg["interpretations"]

    combinations = [c for c in all_combinations if search_term in c.get("pattern_path", "")]
    if not combinations:
        print(f"No matching combinations found for '{search_term}' in group '{group}'.")
        raise SystemExit(0)

print(f"[{group}] Using CSV: {cfg['csv']}")
print(f"[{group}] Using combos JSON: {cfg['combos']}")
print(f"[{group}] Using interpretations file: {interpretations_file}")
print(f"Running {len(combinations)} combination(s)...")


start_time = time.time()

# Load existing interpretations from file (used for display-only path)
if os.path.exists(interpretations_file):
    with open(interpretations_file, "r") as f:
        all_interpretations = json.load(f)
else:
    all_interpretations = {}

# ---------------------------------------------------------------------------
# Two paths per combo:
#   A) Display only: interpretations already in file -> load and plot, skip refinement.
#   B) Full refinement: run refinement + LLM + balance + prior/posterior, then plot and save.
# ---------------------------------------------------------------------------
for combo in combinations:
    pattern_path = combo['pattern_path']

    # Calculate the project number first to avoid unnecessary calculations
    base_name = os.path.basename(pattern_path)
    base_name = os.path.splitext(base_name)[0]  # always remove .xy or .xrdml

    try:
        project_number = extract_project_number_from_filename(base_name)
    except IndexError:
        project_number = "Invalid format"

    filtered_df = df[df["Name"].str.contains(rf'^{project_number}$', na=False)]

    if filtered_df.empty:
        swapped = project_number.replace("_", "-") if "_" in project_number else project_number.replace("-", "_") if "-" in project_number else project_number
        filtered_df = df[df["Name"].str.contains(rf'^{swapped}$', na=False)]

    if filtered_df.empty:
        print("No rows match the filter — skipping...")
        continue
    target = filtered_df["Target"].iloc[0]

    # ---- Path A: Display only (interpretations already in file) ----
    if project_number in all_interpretations:
        interpretations = all_interpretations[project_number]
        plot_contribution_decomposition_dual(interpretations, project_number, target)
        plot_contribution_decomposition_dual_normalized_right_v2(interpretations, project_number, target)
        plot_contribution_decomposition_dual_normalized_right_v3(interpretations, project_number, target)
        plot_contribution_decomposition_dual_normalized_right_v4(interpretations, project_number, target)
        plot_contribution_pie_scaled(interpretations, project_number, target)
        plot_phase_and_interpretation_unnormalized_probabilities_newstyle(interpretations, project_number, filtered_df, target)
        # interpretations = flag_interpretation_trustworthiness(interpretations)
        # interpretations = compute_trust_score(interpretations)
        # # plot_phase_and_interpretation_probabilities_newstyle(interpretations, project_number, filtered_df, target)
        # plot_phase_and_interpretation_unnormalized_probabilities_newstyle(interpretations, project_number, filtered_df, target)
        continue

    # ---- Path B: Full refinement (run refinement, pipeline, plots, save) ----
    print(f"Processing: {combo['pattern_path']} with chemical system: {combo['chemical_system']}")
    result, project_number, target = calculate_spectrum_likelihood_given_interpretation_wrapper(
        combo["pattern_path"], combo["chemical_system"], target, alpha=1)

    if result:
        interpretations = calculate_chemical_factors(filtered_df, result)
        interpretations = evaluate_interpretations_with_llm(filtered_df, result, project_number)
        interpretations = calculate_chemical_factors(filtered_df,interpretations)
        # Normalize scores
        interpretations = normalize_scores_for_sample(interpretations)
        interpretations = normalize_rwp_for_sample(interpretations)
        interpretations = calculate_prior_probability(interpretations, w_llm=0.5, w_bscore =0.7)
        interpretations = calculate_fit_quality(interpretations,w_rwp=1, w_score=0.5)
        interpretations = flag_interpretation_trustworthiness(interpretations)
        interpretations = compute_trust_score(interpretations)
        interpretations = calculate_posterior_probability_of_interpretation(interpretations)

        plot_interpretation_probabilities_with_statistical(interpretations,project_number, target)
        plot_phase_and_interpretation_probabilities_newstyle(interpretations, project_number, filtered_df, target)
        plot_phase_and_interpretation_unnormalized_probabilities_newstyle(interpretations, project_number, filtered_df, target)
        plot_contribution_decomposition(interpretations, project_number, target)

        print("-" * 50)
        print(f"For {project_number}")
        for key, value in interpretations.items():
            print(f"Interpretation: {key}")
            print(f"  Phases: {', '.join(value['phases'])}")
            print(f"  LLM Interpretation Likelihood: {value['LLM_interpretation_likelihood']*100}")
            print(f"  Composition Balance: {value['balance_score']*100}")
            print(f"  Fit quality: {value['fit_quality']}")
            print(f"  Normalized_rwp: {value['normalized_rwp']*100}")
            print(f"  Normalized_score: {value['normalized_score']*100}")
            print(f"  Prior Probability: {value['prior_probability']*100}")
            print(f"  Posterior Probability: {value['posterior_probability']*100}")
            print("-" * 50)

        # Save interpretations to the dictionary and file
        all_interpretations[project_number] = interpretations
        with open(interpretations_file, "w") as f:
            json.dump(all_interpretations, f, indent=4)

print("All interpretations have been processed and saved.")
end_time = time.time()

# Calculate total execution time
total_time = end_time - start_time
print("Total execution time:", round(total_time, 2), "seconds")
