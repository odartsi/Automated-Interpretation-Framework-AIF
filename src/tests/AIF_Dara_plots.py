import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_shared_samples(json_data, reference_metric, project=None):
    """Get samples where AIF ≠ DARA and both have reference_metric defined."""
    shared = []
    for sample, interpretations in json_data.items():
        if project and not sample.startswith(project):
            continue
        
        dara_id, dara = min(
            interpretations.items(),
            key=lambda x: x[1].get("rwp", 0)
        )
        if not dara:
            continue

        best_id, best_interp = max(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0)
        )
        if best_id == dara_id:
            print("I continue for sample : ", sample , "dara = ",dara_id, "and aif = ", best_id)
            continue

        d_val = dara.get(reference_metric)
        a_val = best_interp.get(reference_metric)
        if d_val is not None and a_val is not None:
            shared.append((sample, best_id, dara_id))
    return shared


def plot_metric(json_data, metric_key, shared_samples, higher_is_better=True, save_dir="plots", project=""):
    os.makedirs(save_dir, exist_ok=True)

    samples = []
    dara_vals = []
    aif_vals = []
    aif_ids = []

    for sample, aif_id, dara_id in shared_samples:
        print("I made it for : ",sample, "aid :", aif_id, " dara_id :", dara_id)
        interpretations = json_data[sample]
        dara = interpretations.get(dara_id)
        aif = interpretations.get(aif_id)

        d_val = dara.get(metric_key)
        a_val = aif.get(metric_key)

        if d_val is not None and a_val is not None:
            samples.append(sample)
            dara_vals.append(d_val)
            aif_vals.append(a_val)
            aif_ids.append(aif_id)

    if not samples:
        print(f"No valid samples for {metric_key}.")
        return

    x = np.arange(len(samples))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color logic
    if higher_is_better:
        dara_colors = ['blue' if d == a else 'green' if d > a else 'red' for d, a in zip(dara_vals, aif_vals)]
        aif_colors = ['blue' if d == a else 'green' if a > d else 'red' for d, a in zip(dara_vals, aif_vals)]
    else:
        dara_colors = ['blue' if d == a else 'green' if d < a else 'red' for d, a in zip(dara_vals, aif_vals)]
        aif_colors = ['blue' if d == a else 'green' if a < d else 'red' for d, a in zip(dara_vals, aif_vals)]

    bars1 = ax.bar(x - width / 2, dara_vals, width, color=dara_colors)
    bars2 = ax.bar(x + width / 2, aif_vals, width, color=aif_colors)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height(), f"{dara_vals[i]:.2f}", ha='center', va='bottom', fontsize=8)
        ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height(), f"{aif_vals[i]:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(metric_key.replace("_", " ").title())
    title_suffix = f" - {project}" if project else ""
    ax.set_title(f"{metric_key.replace('_', ' ').title()}: AIF vs DARA{title_suffix}\nGreen = Better, Red = Worse, Blue = Same")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    filename_suffix = f"_{project}" if project else ""
    output_path = os.path.join(save_dir, f"{metric_key}_comparison{filename_suffix}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison metrics for DARA vs AIF.")
    parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
    parser.add_argument("--project", help="Project prefix to filter samples (e.g., TRI, ARR).", default=None)
    args = parser.parse_args()

    json_data = load_json(args.json_file)
    shared = get_shared_samples(json_data,"rwp" , project=args.project) # change the reference metric fromo "search_result_rwp"

    metrics = [
        ("search_result_rwp", False),
        ("rwp", False),
        ("score", True),
        ("LLM_interpretation_likelihood", True),
        ("balance_score", True)
    ]

    for metric_key, higher_is_better in metrics:
        plot_metric(json_data, metric_key, shared, higher_is_better, project=args.project)

# python tests/AIF_Dara_plots.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json --project ARR