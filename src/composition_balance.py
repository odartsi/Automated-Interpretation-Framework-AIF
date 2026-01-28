from pymatgen.core.composition import Composition

def calculate_composition_balance_score_refined(target_composition, output_composition):
    """
    Compute composition balance score between target and output compositions.

    Penalizes:
        - Missing elements in the output
        - Deviations in shared elements (quadratically)
        - Extra elements in the output (max penalty * their fraction)

    Normalizes only over target composition to avoid dilution from extra elements.

    Args:
        target_composition (Composition): Normalized target composition.
        output_composition (dict): Normalized output composition as a dictionary.

    Returns:
        float: Composition balance score (0 to 1, where 1 is a perfect match).
    """
    target_elements = target_composition.elements
    output_elements = output_composition.keys()

    total_penalty = 0.0
    target_total_weight = 0.0  # Denominator only includes target elements

    # 1. Penalize missing/mismatched target elements
    for element in target_elements:
        target_amount = target_composition[element]
        output_amount = output_composition.get(element, 0.0)

        difference = abs(target_amount - output_amount)
        if output_amount == 0.0:
            penalty = 1.0  # max penalty for completely missing
        else:
            penalty = difference ** 2  # quadratic penalty

        total_penalty += penalty * target_amount
        target_total_weight += target_amount

    # 2. Penalize unexpected (extra) elements in output
    for element in output_elements:
        if element not in target_elements:
            output_amount = output_composition[element]
            penalty = 1.0  # max penalty for extra elements
            total_penalty += penalty * output_amount
            # DO NOT add to target_total_weight

    # Normalize and clip to [0, 1]
    normalized_penalty = total_penalty / target_total_weight if target_total_weight > 0 else 1.0
    balance_score = 1 - normalized_penalty
    return max(0, min(1, balance_score))

def cleanup_phases(phases):
    """
    Convert phase compositions into integer formulas, removing space group numbers and metadata.
    
    Args:
        phases (list): List of phase strings with metadata (e.g., "C1.9992O1.9992_194_(icsd_37237)-None").
    
    Returns:
        list: List of cleaned phase formulas (e.g., ['CO']).
    
    Examples:
        ['C1.9992O1.9992_194_(icsd_37237)-None'] → ['CO']
        ['V2O3_15_(icsd_95762)-11'] → ['V2O3']
    """
    new_phases = []
    for phase in phases:
        try:
            # Extract only the chemical formula (before the first underscore)
            chemical_formula = phase.split('_')[0]

            # Convert to integer formula
            comp = Composition(chemical_formula)
            integer_formula = comp.get_integer_formula_and_factor(max_denominator=10)[0]

            # Completely remove underscores and metadata
            cleaned_phase = integer_formula.strip()

            new_phases.append(cleaned_phase)
        except Exception as e:
            print(f" Error processing phase {phase}: {e}")
            new_phases.append(phase)  # Keep original if error occurs

    return new_phases


def remove_elements_from_composition(target, elements_to_remove):
    """
    Remove specified elements from a chemical composition.
    
    Args:
        target (str): Chemical formula string.
        elements_to_remove (list): List of element symbols to remove (e.g., ['O', 'C', 'N', 'H']).
    
    Returns:
        str: Chemical formula with specified elements removed.
    
    Example:
        remove_elements_from_composition("CaCO3", ["O", "C"]) → "Ca"
    """
    composition = Composition(target)
    filtered_composition = {
        el: amt for el, amt in composition.get_el_amt_dict().items() if el not in elements_to_remove
    }
    new_composition = Composition(filtered_composition)
    return new_composition.formula.replace("1", "").replace(" ", "")


def normalize_composition(target):
    """
    Normalize a chemical composition to fractional form.
    
    Args:
        target (str): Chemical formula string.
    
    Returns:
        str: Normalized fractional composition formula.
    
    Example:
        normalize_composition("Ca2O2") → "Ca1O1"
    """
    new_composition = Composition(target)
    normalized_composition = new_composition.fractional_composition
    normalized_formula = normalized_composition.formula.replace(" ", "")
    return normalized_formula

def calculate_chemical_factors(filtered_df, interpretations):
    """
    Calculate composition balance scores for each interpretation.
    
    Compares the target composition from synthesis data with the combined composition
    of phases in each interpretation, accounting for element removal (O, C, N, H).
    
    Args:
        filtered_df (pd.DataFrame): DataFrame containing synthesis data with 'Target' column.
        interpretations (dict or list): Dictionary where each key is an interpretation name
                                       and each value contains phase info. If a list with
                                       one element, extracts the dictionary.

    Returns:
        dict: Updated interpretations dictionary with 'balance_score' added to each interpretation.
    """
    synthesis_row = filtered_df.iloc[0]
    target_raw = synthesis_row['Target']

    # Remove unwanted elements (like O, C, N, H) from the target composition
    elements_to_remove = ["O", "C", "N", "H"]

    target_composition = Composition(target_raw)
    filtered_target = {
        el: amt for el, amt in target_composition.get_el_amt_dict().items() if el not in elements_to_remove
    }
    normalized_target = Composition(filtered_target).fractional_composition

    # Extract dictionary if interpretations is a list
    if isinstance(interpretations, list) and len(interpretations) == 1:
        interpretations = interpretations[0]

    for interpretation_name, interpretation in interpretations.items():
        # Extract & clean product phases
        raw_products = interpretation["phases"]
        cleaned_products = cleanup_phases(raw_products)

        # Extract weight fractions
        weight_fractions = interpretation.get("weight_fraction", [])

        # Normalize product compositions
        normalized_compositions = []
        for phase, weight in zip(cleaned_products, weight_fractions):
            try:
                composition = Composition(phase)
                filtered_composition = {
                    el: amt for el, amt in composition.get_el_amt_dict().items() if el not in elements_to_remove
                }
                normalized_composition = Composition(filtered_composition).fractional_composition
                weighted_composition = {el: amt * (weight / 100) for el, amt in normalized_composition.items()}
                normalized_compositions.append(weighted_composition)
            except Exception as e:
                print(f"Error processing phase {phase}: {e}")

        # Combine all normalized compositions
        combined_composition = {}
        for composition in normalized_compositions:
            for el, amt in composition.items():
                combined_composition[el] = combined_composition.get(el, 0) + amt

        # Normalize to ensure total sum is 1
        total = sum(combined_composition.values())
        if total == 0:
            final_output_composition = {}
        else:
            final_output_composition = {el: round(amt / total, 2) for el, amt in combined_composition.items()}

        # Compute composition balance score
        balance_score = calculate_composition_balance_score_refined(normalized_target, final_output_composition)
        interpretation["balance_score"] = balance_score

    return interpretations