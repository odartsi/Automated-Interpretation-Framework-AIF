import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_summary_all_differing_samples(json_data, metrics_to_check, save_dir="plots", project=None):
    os.makedirs(save_dir, exist_ok=True)

    total_samples = 0
    counts = {metric: {"AIF": 0, "DARA": 0, "Equal": 0} for metric in metrics_to_check}

    for sample, interpretations in json_data.items():
        if project and not sample.startswith(project):
            continue

        dara_id, dara = min(
            interpretations.items(),
            key=lambda x: x[1].get("rwp", 0)
        )

        # dara = interpretations.get("I_1")
        if not dara:
            continue

        best_id, best = max(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0)
        )
        if best_id == dara_id:
            continue

        total_samples += 1
        for metric, higher_is_better in metrics_to_check.items():
            a_val = best.get(metric)
            d_val = dara.get(metric)
            if a_val is not None and d_val is not None:
                if a_val == d_val:
                    counts[metric]["Equal"] += 1
                elif (a_val > d_val and higher_is_better) or (a_val < d_val and not higher_is_better):
                    counts[metric]["AIF"] += 1
                else:
                    counts[metric]["DARA"] += 1

    # Print and plot per metric
    for metric, result in counts.items():
        total_metric = result["AIF"] + result["DARA"] + result["Equal"]
        print(f"\n📊 {metric} (based on {total_metric} differing samples):")
        for label in ["AIF", "DARA", "Equal"]:
            pct = 100 * result[label] / total_metric if total_metric else 0
            print(f"  {label}: {result[label]} ({pct:.1f}%)")

        # Bar plot
        labels = ["AIF", "DARA", "Equal"]
        values = [result[k] * 100 / total_metric if total_metric else 0 for k in labels]

        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, values, color=["green", "red", "gray"])
        plt.ylim(0, 100)
        plt.ylabel("Percentage of Samples")
        title_suffix = f" - {project}" if project else ""
        plt.title(f"AIF vs DARA on '{metric}'{title_suffix}")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        filename_suffix = f"_{project}" if project else ""
        outpath = os.path.join(save_dir, f"aif_vs_dara_{metric}_percent_comparison{filename_suffix}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"✅ Saved plot: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize AIF vs DARA metrics comparison.")
    parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
    parser.add_argument("--project", help="Project prefix to filter samples (e.g., TRI, ARR).", default=None)
    args = parser.parse_args()

    json_data = load_json(args.json_file)
    compare_summary_all_differing_samples(
        json_data,
        metrics_to_check={
            "rwp": False,
            "score": True,
            "LLM_interpretation_likelihood": True,
            "balance_score": True
        },
        project=args.project
    )

# python tests/comparison_statistics.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json --project ARR