import json
from pathlib import Path

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def check_duplicate_interpretations(json_data):
    duplicates_found = False

    for sample_name, interpretations in json_data.items():
        seen_phase_sets = {}
        for interp_id, interp_data in interpretations.items():
            phase_set = frozenset([p.strip() for p in interp_data.get("phases", [])])
            for other_id, other_set in seen_phase_sets.items():
                if phase_set == other_set:
                    print(f"Duplicate found in sample '{sample_name}': {interp_id} and {other_id} have identical phases.")
                    duplicates_found = True
                    break
            seen_phase_sets[interp_id] = phase_set

    if not duplicates_found:
        print("✅ No duplicate interpretations found.")

# Example usage
if __name__ == "__main__":
    # json_path = Path("../interpretations.json")  
    json_path = Path("interpretations_LLM_newapproach_and_dara_score.json")  
    if not json_path.exists():
        print(f"File not found: {json_path}")
    else:
        json_data = load_json(json_path)
        check_duplicate_interpretations(json_data)