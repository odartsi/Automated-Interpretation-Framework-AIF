from refinement_metrics import calculate_spectrum_likelihood_given_interpretation_wrapper
from LLM_evaluation import evaluate_interpretations_with_llm
from composition_balance import calculate_chemical_factors
import time
import json
import os
import logging
from utils import (
    calculate_posterior_probability_of_interpretation,
    calculate_prior_probability,
    calculate_fit_quality,
    normalize_scores_for_sample,
    compute_trust_score,
    normalize_rwp_for_sample,
    plot_metrics_contribution,
    plot_phase_and_interpretation_probabilities,
    flag_interpretation_trustworthiness,
    load_json,
    load_csv,
    extract_project_number_from_filename,
)


# --- configure logging once (console + file) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("run_errors.log", mode="w")]
)

# -----------------------------
# Dataset registry
# -----------------------------
DATASETS = {
    "TRI": {
        "csv": "../data/xrd_data/synthesis_data.csv",
        "combos": "../data/xrd_data/difractogram_paths.json",
        "interpretations":"../data/xrd_data/interpretations/interpretations.json"
        }
  
}

VALID_KEYS = list(DATASETS.keys())

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
    df = load_csv(cfg["csv"])
    all_combinations = load_json(cfg["combos"])
    combinations = all_combinations
    interpretations_file = cfg["interpretations"]

else:
    search_term = input("Enter part of the pattern_path to filter (e.g., 'TRI-15'): ").strip()
    # Try to infer the group from the search term; if ambiguous, ask.
    group = _infer_group_from_search(search_term)
    if group is None:
        group = _choose_group_interactively("Could not infer dataset group. Please specify")

    cfg = DATASETS[group]
    df = load_csv(cfg["csv"])
    all_combinations = load_json(cfg["combos"])
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
    # Look up using project_number or underscore-normalized form (file may have "TRI_197" vs pattern "TRI-197")
    display_key = project_number
    if display_key not in all_interpretations:
        alt_key = project_number.replace("-", "_")
        if alt_key in all_interpretations:
            display_key = alt_key
    if display_key not in all_interpretations:
        alt_key = project_number.replace("_", "-")
        if alt_key in all_interpretations:
            display_key = alt_key

    if display_key in all_interpretations:
        interpretations = all_interpretations[display_key]
        interpretations = compute_trust_score(interpretations)
        interpretations = flag_interpretation_trustworthiness(interpretations)
        plot_phase_and_interpretation_probabilities(interpretations, display_key, filtered_df, target)
        plot_metrics_contribution(interpretations, display_key, target)

        print("-" * 50)
        print(f"For {project_number}")
        for key, value in interpretations.items():
            print(f"Interpretation: {key}")
            print(f"  Phases: {', '.join(value['phases'])}")
            print(f"  LLM Interpretation Likelihood: {value['LLM_interpretation_likelihood']*100}")
            print(f"  Composition Balance: {value['balance_score']*100}")
            print(f"  Normalized_rwp: {value['normalized_rwp']*100}")
            print(f"  Normalized_score: {value['normalized_score']*100}")
            print(f"  Prior Probability: {value['prior_probability']*100}")
            print(f"  Posterior Probability: {value['posterior_probability']*100}")
            print(f"  Trust Score: {value['trust_score']}")
            print(f"  Trustworthiness: {value['trustworthy']}")
            print("-" * 50)
        continue

    # ---- Path B: Full refinement (run refinement, pipeline, plots, save) ----
    print(f"Processing: {combo['pattern_path']} with chemical system: {combo['chemical_system']}")
    result, project_number, target = calculate_spectrum_likelihood_given_interpretation_wrapper(
        combo["pattern_path"], combo["chemical_system"], target, alpha=1)

    if result:
        interpretations = calculate_chemical_factors(filtered_df, result)
        interpretations = evaluate_interpretations_with_llm(filtered_df, result, project_number)
        interpretations = calculate_chemical_factors(filtered_df,interpretations)
        interpretations = normalize_scores_for_sample(interpretations)
        interpretations = normalize_rwp_for_sample(interpretations)
        interpretations = calculate_prior_probability(interpretations, w_llm=0.5, w_bscore =0.7)
        interpretations = calculate_fit_quality(interpretations,w_rwp=1, w_score=0.5)
        interpretations = compute_trust_score(interpretations)
        interpretations = flag_interpretation_trustworthiness(interpretations)
        interpretations = calculate_posterior_probability_of_interpretation(interpretations)

        plot_phase_and_interpretation_probabilities(interpretations, project_number, filtered_df, target)
        plot_metrics_contribution(interpretations, project_number, target)

        print("-" * 50)
        print(f"For {project_number}")
        for key, value in interpretations.items():
            print(f"Interpretation: {key}")
            print(f"  Phases: {', '.join(value['phases'])}")
            print(f"  LLM Interpretation Likelihood: {value['LLM_interpretation_likelihood']*100}")
            print(f"  Composition Balance: {value['balance_score']*100}")
            print(f"  Normalized_rwp: {value['normalized_rwp']*100}")
            print(f"  Normalized_score: {value['normalized_score']*100}")
            print(f"  Prior Probability: {value['prior_probability']*100}")
            print(f"  Posterior Probability: {value['posterior_probability']*100}")
            print(f"  Trust Score: {value['trust_score']}")
            print(f"  Trustworthiness: {value['trustworthy']}")
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
