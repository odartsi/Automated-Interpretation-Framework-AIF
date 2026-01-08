import pandas as pd
import json
import ast
from pymatgen.core.composition import Composition

def extract_elements(precursors_str):
    try:
        precursors = ast.literal_eval(precursors_str)
    except Exception as e:
        print(f"Could not parse precursors: {precursors_str} -> {e}")
        return []

    elements = set()
    for formula in precursors:
        try:
            comp = Composition(formula)
            for el in comp.elements:
                elements.add(el.symbol)
        except Exception as e:
            print(f"Could not parse formula '{formula}': {e}")
    return sorted(elements)

def main():
    project = input("Enter project name (e.g., MINES): ").strip()
    # csv_file = f"synthesis_{project}.csv"
    csv_file = f"../data/alab_synthesis_data/synthesis_{project}.csv"
    output_json = f"composition_{project}.json"
    pattern_prefix = "../../XRD_Likelihood_ML/XRD_all"

    df = pd.read_csv(csv_file)

    result = []
    for _, row in df.iterrows():
        # name = row["Name"]
        name = row["Name"].replace("-", "_")
        precursors_str = row["Precursors"]

        elements = extract_elements(precursors_str)
        chem_sys = "-".join(elements)

        result.append({
            "pattern_path": f"{pattern_prefix}/{name}.xy",
            "chemical_system": chem_sys
        })

    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved {len(result)} entries to {output_json}")

if __name__ == "__main__":
    main()