import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_rwp_comparison(data):
    stats = {
        "total_samples": 0,
        "total_interpretations": 0,
        "compared": 0,
        "rwp_better": 0,
        "rwp_worse": 0,
        "rwp_equal": 0,
        "no_search_rwp": 0,
        "sample_details": defaultdict(list)
    }

    for sample, interps in data.items():
        stats["total_samples"] += 1

        for interp_id, interp_data in interps.items():
            stats["total_interpretations"] += 1
            rwp = interp_data.get("rwp")
            search_rwp = interp_data.get("search_result_rwp")

            if rwp is None or search_rwp is None:
                stats["no_search_rwp"] += 1
                continue

            stats["compared"] += 1
            if rwp < search_rwp:
                stats["rwp_better"] += 1
                result = "better"
            elif rwp > search_rwp:
                stats["rwp_worse"] += 1
                result = "worse"
            else:
                stats["rwp_equal"] += 1
                result = "equal"

            if (rwp-search_rwp ) > 2.0:
                print("⚠️ Large jump in Rwp! Investigate.")
            stats["sample_details"][sample].append((interp_id, rwp, search_rwp, result))

    return stats


# def print_summary(stats):
#     print("\n📊 RWP Comparison Summary:")
#     print(f"  - Total samples: {stats['total_samples']}")
#     print(f"  - Total interpretations: {stats['total_interpretations']}")
#     print(f"  - Comparisons made: {stats['compared']}")
#     print(f"  - AIF RWP better than search_result_rwp: {stats['rwp_better']}")
#     print(f"  - AIF RWP worse than search_result_rwp: {stats['rwp_worse']}")
#     print(f"  - AIF RWP equal to search_result_rwp: {stats['rwp_equal']}")
#     print(f"  - Interpretations with missing search_result_rwp: {stats['no_search_rwp']}")

#     print("\n🔍 Sample-wise Details (only where RWP is worse than search_result_rwp):")
#     for sample, results in stats["sample_details"].items():
#         for interp_id, rwp, search_rwp, result in results:
#             if result == "worse":
#                 print(f"  - {sample} → {interp_id}: RWP={rwp} vs Search={search_rwp} → WORSE")
def print_summary(stats):
    
    compared = stats['compared']
    print("\n📊 RWP Comparison Summary:")
    print(f"  - Total samples: {stats['total_samples']}")
    print(f"  - Total interpretations: {stats['total_interpretations']}")
    print(f"  - Comparisons made (rwp vs search_result_rwp): {compared} (out of {stats['total_interpretations']})")

    def pct(part):
        return f"{(part / compared * 100):.1f}%" if compared else "N/A"

    print(f"  - ✅ RWP better than search_result_rwp: {stats['rwp_better']} ({pct(stats['rwp_better'])})")
    print(f"  - ❌ RWP worse than search_result_rwp: {stats['rwp_worse']} ({pct(stats['rwp_worse'])})")
    print(f"  - ➖ RWP equal to search_result_rwp: {stats['rwp_equal']} ({pct(stats['rwp_equal'])})")
    print(f"  - ⚠️ Interpretations with missing search_result_rwp: {stats['no_search_rwp']}")

    print("\n🔍 Sample-wise Details (only where RWP is worse than search_result_rwp):")
    for sample, results in stats["sample_details"].items():
        for interp_id, rwp, search_rwp, result in results:
            if result == "worse":
                print(f"  - {sample} → {interp_id}: RWP={rwp} vs Search={search_rwp} → WORSE")

if __name__ == "__main__":
    json_path = Path("../data/xrd_data/interpretations/interpretations_second_evaluation.json")  
    # json_path = Path("../data/xrd_data/interpretations/interpretations.json") 
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
    else:
        data = load_json(json_path)
        stats = evaluate_rwp_comparison(data)
        print_summary(stats)

        # 🔢 Collect all RWP and score values
        all_rwp = []
        all_scores = []
        all_rwp_norm = []
        all_norm_scores = []
        all_balance_scores =[]

        for sample, interps in data.items():
            for interp_id, interp_data in interps.items():
                rwp = interp_data.get("rwp")
                score = interp_data.get("score")
                normalized_rwp_original = interp_data.get("normalized_rwp")
                norm_score = interp_data.get("normalized_score")
                balance_score = interp_data.get("balance_score")

                if balance_score is not None:
                    all_balance_scores.append(balance_score)

                if rwp is not None:
                    all_rwp.append(rwp)
                if score is not None:
                    all_scores.append(score)
                if normalized_rwp_original is not None:
                    all_rwp_norm.append(normalized_rwp_original)
                if norm_score is not None:
                    all_norm_scores.append(norm_score)
        print("max score = ", max(all_scores))
        print("min score = ", min(all_scores))

        # plt.figure(figsize=(8, 5))
        # sns.histplot(all_balance_scores, bins=30, kde=True)
        # plt.title("Distribution of balance score values across all interpretations")
        # plt.xlabel("Balance score")
        # plt.ylabel("Count")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # # 📊 Plot RWP distribution
        # plt.figure(figsize=(8, 5))
        # sns.histplot(all_rwp, bins=30, kde=True)
        # plt.title("Distribution of RWP values across all interpretations")
        # plt.xlabel("RWP")
        # plt.ylabel("Count")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # 📈 Plot score values as scatter
        # plt.figure(figsize=(8, 5))
        # sns.histplot(all_scores, bins=30, kde=True)
        # plt.title("Distribution of Interpretation Scores")
        # plt.xlabel("Score")
        # plt.ylabel("Count")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        def normalize_rwp_reciprocal(rwp, scale=10):
            return 1 / (1 + rwp / scale)

        normalized_rwp = [normalize_rwp_reciprocal(r) for r in all_rwp]

        # plt.figure(figsize=(8, 5))  
        # plt.scatter(all_rwp, normalized_rwp, alpha=0.6, color="pink")
        # plt.title("Raw RWP vs Normalized RWP Score")
        # plt.xlabel("Raw RWP")
        # plt.ylabel("Normalized Score reciptrocal")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        import numpy as np
        import matplotlib.pyplot as plt

        # 1. Reciprocal normalization
        def normalize_rwp_reciprocal(rwp, scale=10):
            return 1 / (1 + rwp / scale)

        # 2. Exponential decay normalization
        def normalize_rwp_exponential(rwp, scale=20):
            return np.exp(-rwp / scale)

        # 3. Log-inverse normalization
        def normalize_rwp_log_inverse(rwp, max_rwp=50):
            return max(0, 1 - np.log(rwp) / np.log(max_rwp))

        def normalize_rwp_sigmoid_flat(rwp, k=15, center=9):
            return 1 / (1 + np.exp(k * (rwp - center)))
        
        def normalize_rwp_softclip_linear(rwp, max_rwp=50):
            """Linear decay with clipping: score = 1 - (rwp / max_rwp), clipped to [0, 1]."""
            return np.clip((max_rwp - rwp) / max_rwp, 0, 1)
        normalized_rwp = [normalize_rwp_softclip_linear(r) for r in all_rwp]
        plt.figure(figsize=(8, 5))
        plt.scatter(all_rwp, normalized_rwp, alpha=0.6, color="cyan")
        plt.title("Raw RWP vs Normalized RWP Score (Softclip Linear)")
        plt.xlabel("Raw RWP")
        plt.ylabel("Normalized Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # Define softclip normalization
        def normalize_rwp_softclip_linear(rwp, max_rwp=50):
            return np.clip((max_rwp - rwp) / max_rwp, 0, 1)

        # Create RWP values and set max_rwp thresholds
        rwp_values = np.linspace(0, 60, 150)
        max_rwp_values = [20, 30, 40, 50, 60]

        # Normalize for each max_rwp and store in a dictionary
        normalized_dict = {}
        for max_rwp in max_rwp_values:
            normalized_dict[f"max_rwp_{max_rwp}"] = [normalize_rwp_softclip_linear(r, max_rwp=max_rwp) for r in rwp_values]

        # Create DataFrame for tabular inspection
        comparison_df = pd.DataFrame({"RWP": rwp_values})
        for max_rwp in max_rwp_values:
            comparison_df[f"max_rwp_{max_rwp}"] = normalized_dict[f"max_rwp_{max_rwp}"]

        # Print the first 20 rows
        print(comparison_df.to_string(index=False))

        # Plot the curves
        plt.figure(figsize=(10, 6))
        for max_rwp in max_rwp_values:
            plt.plot(rwp_values, normalized_dict[f"max_rwp_{max_rwp}"], label=f"max_rwp = {max_rwp}")
        plt.title("Softclip Linear Normalization of RWP")
        plt.xlabel("Raw RWP")
        plt.ylabel("Normalized Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # # Generate normalized lists 
        # normalized_reciprocal = [normalize_rwp_reciprocal(r) for r in all_rwp]
        # normalized_exp = [normalize_rwp_exponential(r) for r in all_rwp]
        # normalized_log = [normalize_rwp_log_inverse(r) for r in all_rwp]
        # normalized_sigmoid_flat = [normalize_rwp_sigmoid_flat(r) for r in all_rwp]

        # plt.figure(figsize=(8, 5))
        # plt.scatter(all_rwp, normalized_exp, label="exp(-rwp/20)", color="purple", alpha=0.5)
        # plt.scatter(all_rwp, normalized_reciprocal, label="1 / (1 + rwp/10)", color="green", alpha=0.5)
        # plt.scatter(all_rwp, normalized_log, label="1 - log(rwp)/log(50)", color="blue", alpha=0.5)
        # plt.scatter(all_rwp, normalized_sigmoid_flat, label="sigmoid(center=9)", color="orange", alpha=0.5)

        # plt.title("Raw RWP vs Normalized Score — Improved Comparison")
        # plt.xlabel("Raw RWP")
        # plt.ylabel("Normalized Score")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # # scale = 20
        # # normalized_rwp = [np.exp(-r / scale) for r in all_rwp]
        # # plt.figure(figsize=(8, 5))
        # # plt.scatter(all_rwp, all_rwp_norm, alpha=0.6, color="green")
        # # plt.title("Raw RWP vs Normalized RWP Score")
        # # plt.xlabel("Raw RWP")
        # # plt.ylabel("Original Normalized Score)")
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()

        # # # 🔹 2. Scatter plot: Raw RWP vs Normalized Score
        # # plt.figure(figsize=(8, 5))
        # # plt.scatter(all_rwp, normalized_rwp, alpha=0.6, color="purple")
        # # plt.title("Raw RWP vs Normalized RWP Score")
        # # plt.xlabel("Raw RWP")
        # # plt.ylabel("Normalized Score (exp(-rwp/20))")
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()

        # # k = 7
        # # center = 0.2
        # # normalized_scores = [1 / (1 + np.exp(-k * (s - center))) for s in all_scores]
        # # print("for k=7",max(all_scores), max(normalized_scores))
        # # plt.figure(figsize=(8, 5))
        # # plt.scatter(all_scores, normalized_scores, alpha=0.6, color="purple")
        # # plt.title("Sigmoid Normalization of Score")
        # # plt.xlabel("Original Score")
        # # plt.ylabel("Normalized Score")
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()
        # # def scaled_sigmoid(s, k=7, center=0.2):
        # #     raw = 1 / (1 + np.exp(-k * (s - center)))
        # #     max_val = 1 / (1 + np.exp(-k * (1 - center)))  # Ensures score=1 maps to normalized=1
        # #     return raw / max_val
        # # normalized_scores = [scaled_sigmoid(s, k=k, center=center) for s in all_scores]
        # # print("Scaled sigmoid max (should be 1):", scaled_sigmoid(max(all_scores), k=k, center=center))
        # # plt.figure(figsize=(8, 5))
        # # plt.scatter(all_scores, normalized_scores, alpha=0.6, color="red")
        # # plt.title("Sigmoid Normalization of Score")
        # # plt.xlabel("Original Score")
        # # plt.ylabel("Normalized Score")
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()

        # # plt.figure(figsize=(8, 5))
        # # plt.scatter(all_scores, all_norm_scores, alpha=0.6, color="green")
        # # plt.title("Original Normalization of Score")
        # # plt.xlabel("Original Score")
        # # plt.ylabel("Normalized Score")
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()


        # # ks = [3, 7, 15]
        # # scores = np.linspace(0, 1, 200)
        # # plt.figure(figsize=(8, 5))
        # # for k_val in ks:
        # #     plt.plot(scores, [scaled_sigmoid(s, k=k_val, center=0.2) for s in scores], label=f"k={k_val}")
        # # plt.title("Scaled Sigmoid Curves for Different k")
        # # plt.xlabel("Raw Score")
        # # plt.ylabel("Normalized Score")
        # # plt.legend()
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()