import json

def get_aif_selections(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    aif_selection = {}
    for sample, interps in data.items():
        if not interps:
            continue
        # Select interpretation with the highest posterior probability
        best_id, _ = max(interps.items(), key=lambda x: x[1].get("posterior_probability", 0))
        aif_selection[sample] = best_id
    return aif_selection

def compare_aif_choices(file1, file2):
    aif1 = get_aif_selections(file1)
    aif2 = get_aif_selections(file2)

    all_samples = set(aif1) | set(aif2)
    differences = []

    for sample in sorted(all_samples):
        id1 = aif1.get(sample)
        id2 = aif2.get(sample)
        if id1 != id2:
            differences.append((sample, id1, id2))

    print(f"📊 Found {len(differences)} samples with different AIF selections:")
    for sample, id1, id2 in differences:
        print(f" - {sample}: {file1} → {id1}, {file2} → {id2}")

if __name__ == "__main__":
    file1 = "train_new.json"
    file2 = "train_new6.json"
    compare_aif_choices(file1, file2)