import json
from pathlib import Path
from collections import defaultdict, Counter
import argparse

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_values(aif_val, dara_val, is_lower_better=False):
    if aif_val is None or dara_val is None:
        return None
    if is_lower_better:
        return "AIF" if aif_val < dara_val else "DARA" if dara_val < aif_val else "Equal"
    return "AIF" if aif_val > dara_val else "DARA" if dara_val > aif_val else "Equal"


def compare_metrics(json_data):
    better_counts = {
        "AIF": defaultdict(list),
        "DARA": defaultdict(list)
    }
    combo_counter = Counter()
    combo_samples = defaultdict(list)

    for sample, interps in json_data.items():
        
        dara_id, dara = min(
            interps.items(),
            key=lambda x: x[1].get("rwp", 0)
        )
        if not dara:
            continue

        aif_id, aif = max(interps.items(), key=lambda x: x[1].get("posterior_probability", -1))

        if aif_id == dara_id:
            continue  # skip if AIF and DARA agree

        comparison = {
            "RWP": compare_values(aif.get("rwp"), dara.get("rwp"), is_lower_better=True),
            "Score": compare_values(aif.get("score"), dara.get("score")),
            "LLM": compare_values(aif.get("LLM_interpretation_likelihood"), dara.get("LLM_interpretation_likelihood")),
            "Balance score": compare_values(aif.get("balance_score"), dara.get("balance_score")),
        }

        for metric, result in comparison.items():
            if result in ["AIF", "DARA"]:
                better_counts[result][metric].append(sample)

        aif_better_combo = tuple(sorted(k for k, v in comparison.items() if v == "AIF"))
        dara_better_combo = tuple(sorted(k for k, v in comparison.items() if v == "DARA"))

        if aif_better_combo:
            combo_counter[("AIF", aif_better_combo)] += 1
            combo_samples[("AIF", aif_better_combo)].append(sample)
        if dara_better_combo:
            combo_counter[("DARA", dara_better_combo)] += 1
            combo_samples[("DARA", dara_better_combo)].append(sample)

    return better_counts, combo_counter, combo_samples


def summarize_results(better_counts, combo_counter, combo_samples):
    print("⭐ AIF better (individual metrics):")
    for metric, samples in better_counts["AIF"].items():
        print(f"  - {metric}: {len(samples)} samples → {samples}")

    print("\n🔻 DARA better (individual metrics):")
    for metric, samples in better_counts["DARA"].items():
        print(f"  - {metric}: {len(samples)} samples → {samples}")

    # print("\n📊 Exact combinations where AIF or DARA is better:")
    # for (winner, combo), count in combo_counter.items():
    #     combo_str = ", ".join(combo)
    #     samples = combo_samples[(winner, combo)]
    #     print(f"  - {winner} better in [{combo_str}]: {count} sample(s) → {samples}")
    print("\n📊 Detailed combinations where AIF is better:")
    for (winner, combo), count in combo_counter.items():
        if winner != "AIF":
            continue
        combo_str = ", ".join(combo)
        samples = combo_samples[(winner, combo)]
        print(f"  - 📌 AIF better in [{combo_str}]: {count} sample(s)")
        print(f"    → {', '.join(samples)}")

    print("\n📊 Detailed combinations where DARA is better:")
    for (winner, combo), count in combo_counter.items():
        if winner != "DARA":
            continue
        combo_str = ", ".join(combo)
        samples = combo_samples[(winner, combo)]
        print(f"  - 🧱 DARA better in [{combo_str}]: {count} sample(s)")
        print(f"    → {', '.join(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison best selection AIF.")
    parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
    args = parser.parse_args()
    json_data = load_json(args.json_file)
    if not args.json_file:
        json_path = Path("../data/xrd_data/interpretations/interpretations_second_evaluation.json")  # ⬅️ Replace with your path
        json_data = load_json(json_path)
    better_counts, combo_counter, combo_samples = compare_metrics(json_data)
    summarize_results(better_counts, combo_counter, combo_samples)

#python tests/check_if_AIF_selected_the_best.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json