import re
# from pymatgen.core import Composition
from pymatgen.core.composition import Composition

def calculate_normalized_composition(formula: str):
    """
    Calculate the normalized composition for a given formula.
    """
    composition = Composition(formula)
    normalized = composition.get_el_amt_dict()
    total = sum(normalized.values())
    normalized = {element: amount / total for element, amount in normalized.items()}
    return normalized

def clean_phase_name(phase):
    """
    Extract valid chemical formulas from a phase name.
    
    Parameters:
        phase (str): Phase name, potentially with metadata (e.g., "V2O3_167_(icsd_1869)-0").
    
    Returns:
        str: Cleaned chemical formula.
    """
    # Remove ICSD metadata and file extensions
    cleaned_phase = phase.split("_(icsd")[0]  # Remove metadata like _(icsd_1869)
    cleaned_phase = cleaned_phase.replace(".cif", "")  # Remove .cif extension
    cleaned_phase = re.sub(r"_\d+$", "", cleaned_phase)  # Remove trailing numeric metadata (_167, etc.)
    return cleaned_phase.strip()

def adjust_for_gas_loss(precursors, temperature, furnace_type):
    """
    Adjust the composition of precursors to account for expected losses.
    """
    adjusted_precursors = []
    for precursor in precursors:
        comp = Composition(precursor)
        elem_amt_dict = comp.get_el_amt_dict()

        # Check for decomposition of carbonates
        if 'C' in elem_amt_dict and 'O' in elem_amt_dict and int(temperature) > 600 and furnace_type == 'Air':
            if 'CO3' in precursor:
                moles_CO2 = elem_amt_dict['C']
                elem_amt_dict['C'] -= moles_CO2
                elem_amt_dict['O'] -= moles_CO2 * 3

        # Additional checks for other volatile groups
        if 'OH' in precursor and int(temperature) > 100:  # Example temperature for dehydration
            moles_H2O = elem_amt_dict['H']
            elem_amt_dict['H'] -= moles_H2O
            elem_amt_dict['O'] -= moles_H2O
        # Create a new Composition from the modified dictionary
        new_comp = Composition(elem_amt_dict)
        adjusted_precursors.append(new_comp)
    return adjusted_precursors

def compare_compositions(precursors, products, product_fractions, temperature, duration, furnace_type):
    """
    Compare the adjusted compositions of precursors with the products.
    """
    # Adjust precursors for gas loss
    adjusted_precursors = adjust_for_gas_loss(precursors, temperature, furnace_type)

    # Calculate combined composition of adjusted precursors
    combined_precursors = Composition({})
    for comp in adjusted_precursors:
        combined_precursors += comp
    normalized_before = calculate_normalized_composition(str(combined_precursors.formula))
    
    # Combine products using their fractions
    normalized_after = combine_compositions_with_fractions(products, product_fractions)
    
    return normalized_before, normalized_after

def combine_compositions_with_fractions(compositions, fractions):
    """
    Combine multiple compositions with their respective fractions.
    """
    combined_composition = Composition({})
    for comp, frac in zip(compositions, fractions):
        combined_composition += Composition(comp) * frac
    return calculate_normalized_composition(str(combined_composition.formula))

def calculate_balance_score(normalized_before, normalized_after):
    """
    Calculate a balance score between 0 and 1 based on the difference between normalized compositions.
    """
    elements = set(normalized_before.keys()).union(set(normalized_after.keys()))
    total_difference = 0
    for element in elements:
        before_value = normalized_before.get(element, 0)
        after_value = normalized_after.get(element, 0)
        total_difference += abs(before_value - after_value)
    
    # Scale score between 0 and 1
    score = 1 - (total_difference / 2) #TODO divide with the total number of elements
    return max(0, min(1, score))  # Ensure score is within bounds

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

def calculate_composition_balance_score_refined_(target_composition, output_composition):
    """
    Compute composition balance score between target and output compositions.

    Args:
        target_composition (Composition): Normalized target composition.
        output_composition (dict): Normalized output composition as a dictionary.

    Returns:
        float: Composition balance score (0 to 1, where 1 is a perfect match).
    """
    target_elements = target_composition.elements
    output_elements = output_composition.keys()

    # Initialize penalties
    total_penalty = 0.0
    total_weight = 0.0

    # Compute penalties for missing/mismatched elements
    for element in target_elements:
        target_amount = target_composition[element]
        output_amount = output_composition[element] if element in output_elements else 0.0

        difference = abs(target_amount - output_amount)

        # Scalable penalty: larger differences are penalized more
        if output_amount == 0.0:
            penalty = 1.0  # Maximum penalty for missing elements
        else:
            penalty = difference ** 2  # Quadratic penalty for differences

        total_penalty += penalty * target_amount
        total_weight += target_amount

    # Normalize penalties
    normalized_penalty = total_penalty / total_weight if total_weight > 0 else 1.0
    balance_score = 1 - normalized_penalty  # Convert penalty to a score (higher is better)

    return max(0, min(1, balance_score))  # Ensure score is within [0,1] 

def cleanup_phases(phases):
    """
    Convert phase compositions into integer formulas and preserve space group numbers.
    
    Examples:
        ['C1.9992O1.9992_194_(icsd_37237)-None'] → ['CO_194']
        ['V2O3_15_(icsd_95762)-11'] → ['V2O3_15']
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
    # Parse the composition
    composition = Composition(target)

    # Create a new composition dictionary without the unwanted elements
    filtered_composition = {
        el: amt for el, amt in composition.get_el_amt_dict().items() if el not in elements_to_remove
    }

    # Create a new Composition object from the filtered dictionary
    new_composition = Composition(filtered_composition)

    # Return the updated formula as a compact string (remove spaces)
    return new_composition.formula.replace("1", "").replace(" ", "")


def normalize_composition(target):

    # Create a new Composition object from the filtered dictionary
    new_composition = Composition(target)
    # Normalize the composition to get fractional values
    normalized_composition = new_composition.fractional_composition

    # Format the normalized composition as a string
    normalized_formula = normalized_composition.formula.replace(" ", "")
    
    return normalized_formula

def calculate_chemical_factors(filtered_df, interpretations):
    """
    Calculate composition balance scores for each interpretation.
    
    Parameters:
        synthesis_csv (str): Path to the synthesis data CSV file.
        interpretations (dict): Dictionary where each key is an interpretation name,
                                and each value contains phase info.

    Returns:
        dict: Composition balance scores for each interpretation.
    """

    # # Load synthesis data
    # synthesis_df = pd.read_csv(synthesis_csv)
    print("In chemical factor syntheis_df ")
    print(filtered_df)
    # Extract and process the target composition
    synthesis_row = filtered_df.iloc[0]
    
    # target_raw = synthesis_row['Target.1']
    target_raw = synthesis_row['Target']
    print("\nTarget Composition:", target_raw)

    # Remove unwanted elements (like O, C, N, H) from the target composition
    elements_to_remove = ["O", "C", "N", "H"]
    cleaned_target = remove_elements_from_composition(target_raw, elements_to_remove)
    normalized_target = normalize_composition(cleaned_target)
    print("Normalized target: ", normalized_target)

    target_composition = Composition(target_raw)
    filtered_target = {
        el: amt for el, amt in target_composition.get_el_amt_dict().items() if el not in elements_to_remove
    }
    normalized_target = Composition(filtered_target).fractional_composition
    print("Normalized Target Composition:", normalized_target)

    # Dictionary to store balance scores
    balance_scores = {}
    if isinstance(interpretations, list) and len(interpretations) == 1:
        interpretations = interpretations[0]  # Extract the dictionary

    for interpretation_name, interpretation in interpretations.items():
        print("\nProcessing Interpretation:", interpretation_name)

        # Extract & clean product phases
        raw_products = interpretation["phases"]
        cleaned_products = cleanup_phases(raw_products)  # Convert to integer formulas
        print("Raw Output Phases:", raw_products)
        print("Cleaned Output Phases:", cleaned_products)

        # Extract weight fractions
        weight_fractions = interpretation.get("weight_fraction", [])
        # if not isinstance(weight_fractions, list) or len(weight_fractions) != len(cleaned_products):
        #     weight_fractions = [1.0 / len(cleaned_products)] * len(cleaned_products)  # Equal distribution if missing
        print("Weight Fractions:", weight_fractions)

        # Normalize product compositions
        normalized_compositions = []
        for phase, weight in zip(cleaned_products, weight_fractions):
            try:
                composition = Composition(phase)

                # Remove unwanted elements
                filtered_composition = {
                    el: amt for el, amt in composition.get_el_amt_dict().items() if el not in elements_to_remove
                }

                # Normalize and weight the composition
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
        print("debug ","combined compositions ", combined_composition)
        total = sum(combined_composition.values())
        print("debug" , "the total is = ", total)
        if total == 0:
            print("⚠️ Warning: combined_composition is zero — skipping normalization.")
            final_output_composition = {}
        else:
                final_output_composition = {el: round(amt / total, 2) for el, amt in combined_composition.items()}
        print("Normalized Output Composition:", final_output_composition)
        final_composition_str = Composition(final_output_composition).to_pretty_string()
        print("The other normalised output composition: ",final_composition_str )

        # Compute composition balance score
        balance_score = calculate_composition_balance_score_refined(normalized_target, final_output_composition)
        balance_scores[interpretation_name] = balance_score

        # Update the interpretation with the balance score
        interpretation["balance_score"] = balance_score
        print("Balance Score:", balance_score)

    return interpretations