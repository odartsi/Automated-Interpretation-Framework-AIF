import json
import random
import argparse
from pathlib import Path

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_differing_samples(json_data):
    differing = {}

    for sample, interps in json_data.items():
        if not isinstance(interps, dict):
            continue

        try:
            aif_id = max(interps.items(), key=lambda x: x[1].get("posterior_probability", 0))[0]
            dara_id = min(interps.items(), key=lambda x: x[1].get("rwp", 1e9))[0]

            if aif_id != dara_id:
                differing[sample] = interps
        except Exception:
            continue

    return differing

def split_and_save(samples_dict, train_ratio=0.8, seed=42, out_prefix=""):
    samples = list(samples_dict.keys())
    random.seed(seed)
    random.shuffle(samples)

    split_point = int(len(samples) * train_ratio)
    train_samples = {k: samples_dict[k] for k in samples[:split_point]}
    test_samples = {k: samples_dict[k] for k in samples[split_point:]}

    with open(f"{out_prefix}train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(f"{out_prefix}test.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    print(f"Saved {len(train_samples)} training and {len(test_samples)} test samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and split differing samples into train/test.")
    parser.add_argument("json_file", help="Path to interpretations JSON file.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data (default: 0.8)")
    parser.add_argument("--out_prefix", default="", help="Optional prefix for output files (e.g., 'tri_')")
    args = parser.parse_args()

    path = Path(args.json_file)
    if not path.exists():
        print(f"❌ File not found: {path}")
        exit(1)

    json_data = load_json(path)
    diff_samples = extract_differing_samples(json_data)
    split_and_save(diff_samples, train_ratio=args.train_ratio, out_prefix=args.out_prefix)