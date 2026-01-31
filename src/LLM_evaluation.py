import openai
import re
import ast
import os
import numpy as np
import textwrap
from pymatgen.core import Composition
from math import gcd
from functools import reduce
from collections import OrderedDict
from utils import cleanup_phases, type_of_furnace, celsius_to_kelvin
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
PROMPT_TEMPLATE_PATH = PROMPT_DIR / "llm_prompt_template.txt"

# # NOTE: Uses CBORG gateway; API_KEY must be set in env.
# openai.api_key = os.getenv("API_KEY")
# if not openai.api_key:
#     raise ValueError("API key is missing!")
# openai.api_base = "https://api.cborg.lbl.gov"  # CBORG base URL
# model = "openai/gpt-4o"  # deployment id used by CBORG

DEFAULT_API_BASE = "https://api.cborg.lbl.gov"
DEFAULT_MODEL = "openai/gpt-4o"

def configure_openai_from_env():
    """
    Configure OpenAI client from environment variables.
    Required for actual API calls:
      - API_KEY
    Optional:
      - API_BASE (defaults to CBORG)
      - MODEL (defaults to CBORG deployment id)
    """
    openai.api_key = os.getenv("API_KEY")
    openai.api_base = os.getenv("API_BASE", DEFAULT_API_BASE)

    # Return model name to use
    return os.getenv("MODEL", DEFAULT_MODEL)

def load_prompt_template(template_path):
    """Load a prompt template from disk."""
    with open(template_path, "r") as f:
        return f.read()


def describe_clean_composition(formula_str, digits=4, max_denominator=30):
    """
    Produce a human-readable description of the fractional composition and a small-denominator
    integer approximation (for readability in prompts).
    """
    comp = Composition(formula_str)

    # Get rounded fractional composition
    frac_dict = {el: round(amt, digits) for el, amt in comp.fractional_composition.get_el_amt_dict().items()}

    # Reconstruct and quantize
    frac_comp = Composition(frac_dict)
    quantized_formula_str = frac_comp.get_integer_formula_and_factor(max_denominator=max_denominator)[0]
    quantized_dict = Composition(quantized_formula_str).get_el_amt_dict()

    # Format with clean chemical formula style
    ordered_elements = list(frac_dict.keys())
    quantized_str = ''.join(
        f"{el}" if int(quantized_dict[el]) == 1 else f"{el}{int(quantized_dict[el])}"
        for el in ordered_elements if el in quantized_dict
    )

    return f"{formula_str}, fractional_composition = {frac_dict}, approximately equal to {quantized_str}"

def flatten_chemical_formula(formula):
    """
    Flatten a chemical formula by expanding parentheses and combining duplicate elements.
    - Removes explicit 1s in the output.
    - Preserves first-seen element order.
    """
    def expand_parentheses(f):
        pattern = r'\(([^()]+)\)(\d*)'
        while re.search(pattern, f):
            f = re.sub(pattern, lambda m: expand_group(m.group(1), m.group(2)), f)
        return f

    def expand_group(group, multiplier):
        multiplier = int(multiplier) if multiplier else 1
        elems = re.findall(r'([A-Z][a-z]*)(\d*)', group)
        return ''.join(f"{el}{int(cnt or 1)*multiplier}" for el, cnt in elems)

    def parse_and_collapse(flat_formula):
        elems = re.findall(r'([A-Z][a-z]*)(\d*)', flat_formula)
        counts = OrderedDict()
        for el, cnt in elems:
            cnt = int(cnt or 1)
            counts[el] = counts.get(el, 0) + cnt
        return counts

    no_parens = expand_parentheses(formula)
    collapsed = parse_and_collapse(no_parens)
    return ''.join(f"{el}{cnt if cnt > 1 else ''}" for el, cnt in collapsed.items())


def get_phase_likelihood_via_prompt_all_interpretations(synthesis_data, all_phases, composition_balance_scores):
    """
    Evaluates likelihoods for multiple interpretations simultaneously.

    Parameters:
        synthesis_data (str): A string describing the synthesis process and parameters.
        all_phases (dict): Dictionary of interpretation name to list of phase names.

    Returns:
        dict: Dictionary mapping interpretation names to:
            - Likelihoods (dict)
            - Explanations (dict)
            - Interpretation_Likelihood (float)
            - Interpretation_Explanation (str)
    """
   

    prompt = textwrap.dedent(f"""
    Given the following synthesis data:
    {synthesis_data}

    Below are multiple proposed phase interpretations. For each interpretation, determine the likelihood that the listed solid phases have formed under the given synthesis conditions.

    Take into account:
    - Whether the oxidation state is thermodynamically plausible (based on precursors, temperature, and synthesis atmosphere).
    - Whether the specific polymorph (space group) is known to be stable at the synthesis temperature and pressure. If multiple polymorphs exist for the same composition, prefer the polymorph known to be stable under the synthesis conditions.
    - Whether the overall elemental composition of the phases, weighted by their fractions, matches the expected target composition. Interpretations with large elemental imbalances (e.g., excess or missing cations) should be penalized. Use the provided composition balance score as an indicator of this match.

    """)

    # Add interpretation info
    prompt += "\nInterpretations:\n"
    for name, phases in all_phases.items():
        prompt += f"- {name}: {', '.join(phases)}\n"

    # Add composition balance scores
    prompt += "\nComposition balance scores:\n"
    for name, score in composition_balance_scores.items():
        prompt += f"- {name}: {round(score, 3)}\n"
    
    prompt += load_prompt_template(PROMPT_TEMPLATE_PATH)
    model_to_use = configure_openai_from_env()
    if not openai.api_key:
        raise ValueError(
            "API_KEY is missing. Set it in your environment or in the notebook before running LLM evaluation."
        )
    try:
        response = openai.ChatCompletion.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are an expert in material synthesis and phase prediction. Use thermodynamics, kinetics, and polymorph knowledge to evaluate stability and likelihood of observed phases."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            seed=42,
            stream=False,
            # max_tokens=5000
        )
        content = response["choices"][0]["message"]["content"].strip()

        # Step 3: Handle Markdown wrapping
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip("`").strip()

        # Step 4: Parse each interpretation block
        interpretations = {}
        interp_blocks = re.finditer(r'"(I_\d+)"\s*:\s*\{(.*?)(?=^\s*"I_\d+"\s*:|\Z)', content, re.DOTALL | re.MULTILINE)

        for match in interp_blocks:
            interp_name, block = match.group(1), match.group(2)

            # Extract fields from block
            likelihoods_match = re.search(r'"Likelihoods"\s*:\s*\{(.*?)\}', block, re.DOTALL)
            explanations_match = re.search(r'"Explanations"\s*:\s*\{(.*?)\}', block, re.DOTALL)
            interp_lik_match = re.search(r'"Interpretation_Likelihood"\s*:\s*([\d.]+)', block)
            interp_expl_match = re.search(r'"Interpretation_Explanation"\s*:\s*"([^"]*?)"', block, re.DOTALL)

            try:
                interpretations[interp_name] = {
                    "Likelihoods": ast.literal_eval("{" + likelihoods_match.group(1) + "}") if likelihoods_match else {},
                    "Explanations": ast.literal_eval("{" + explanations_match.group(1) + "}") if explanations_match else {},
                    "Interpretation_Likelihood": float(interp_lik_match.group(1)) if interp_lik_match else None,
                    "Interpretation_Explanation": interp_expl_match.group(1).strip() if interp_expl_match else "",
                }
            except Exception as e:
                print(f"⚠️ Failed to parse {interp_name}: {e}")
                interpretations[interp_name] = {
                    "Likelihoods": {},
                    "Explanations": {},
                    "Interpretation_Likelihood": None,
                    "Interpretation_Explanation": "",
                }

        return interpretations

    except Exception as e:
        print("🚨 API call failed:", str(e))
        return {}



def evaluate_interpretations_with_llm(filtered_df, interpretations,project_number, model="openai/gpt-4o"):
    """
    Evaluates interpretations using LLM-based likelihoods and explanations.

    Parameters:
        synthesis_csv (str): Path to the synthesis data CSV.
        interpretations (dict): Dictionary containing interpretation data.
        model (str): The model to use for LLM evaluation.

    Returns:
        dict: Dictionary of LLM likelihoods and explanations per interpretation.
    """
   
    llm_evaluation_results = {}

    if filtered_df.empty:
            print(f"No matching rows found in synthesis data for project: {project_number}")
            llm_evaluation_results[project_number] = {
                "interpretation_likelihoods": {},
                "interpretation_explanations": {},
                "error": f"No synthesis data found for project {project_number}",
            }

    # Use the first matching row
    synthesis_row = filtered_df.iloc[0].copy()
    synthesis_row['Furnace'] = type_of_furnace(synthesis_row['Furnace'])
    precursors_raw = synthesis_row['Precursors']
    precursors_list = ast.literal_eval(precursors_raw)
    flattened_precursors = [flatten_chemical_formula(p) for p in precursors_list]
  
    synthesis_data = f"""
    Solid state synthesis: gram-quantity precursors are mixed and heated in a furnace.
    Target: {synthesis_row['Target']}
    Precursors: {", ".join(flattened_precursors)}
    Temperature: {celsius_to_kelvin(synthesis_row['Temperature (C)'])}K ({synthesis_row['Temperature (C)']}°C)
    Dwell Duration: {synthesis_row['Dwell Duration (h)']} hours
    Furnace: {synthesis_row['Furnace']}
    """

    all_phases = {}
    for interpretation_name, interpretation in interpretations.items():
        raw_phases = cleanup_phases(interpretation["phases"])
        weight_fractions = interpretation["weight_fraction"]

        entries = []
        for phase_str, weight in zip(raw_phases, weight_fractions):
            main_part = phase_str.split("_(")[0]  
            if "_" in main_part:
                formula, sg = main_part.split("_")
                try:
                    detailed_info = describe_clean_composition(formula)
                    entry = (f"{formula} (space group {sg}, weight fraction {round(weight, 2)}%, "
                            f"{detailed_info.split(', ', 1)[1]})")
                except Exception as e:
                    entry = (f"{formula} (space group {sg}, weight fraction {round(weight, 2)}%, "
                            f"normalization failed: {e})")
            else:
                try:
                    detailed_info = describe_clean_composition(main_part)
                    entry = (f"{main_part} (weight fraction {round(weight, 2)}%, "
                            f"{detailed_info.split(', ', 1)[1]})")
                except Exception as e:
                    entry = (f"{main_part} (weight fraction {round(weight, 2)}%, "
                            f"normalization failed: {e})")
            entries.append(entry)
        print("The entries are: ", entries)
        all_phases[interpretation_name] = entries   


    composition_balance_scores = {}
    for interpretation_name, interpretation in interpretations.items():
        composition_balance = interpretation["balance_score"]
        composition_balance_scores[interpretation_name] = composition_balance
  
    results = get_phase_likelihood_via_prompt_all_interpretations(synthesis_data, all_phases, composition_balance_scores)
    print(" New approach ")
    for interp_name, data in results.items():
        print(f"\n🔹 {interp_name}")

        # Handle interpretation likelihood safely
        il = data.get('Interpretation_Likelihood', 'N/A')
        if isinstance(il, (float, int)):
            print(f"  Interpretation Likelihood: {il:.2f}")
        else:
            print(f"  Interpretation Likelihood: {il}")

        print(f"  Explanation: {data.get('Interpretation_Explanation', 'N/A')}")

        print("\n  Phase Likelihoods:")
        for phase, score in data.get("Likelihoods", {}).items():
            if isinstance(score, (float, int)):
                print(f"    - {phase}: {score:.2f}")
            else:
                print(f"    - {phase}: {score}")

        print("\n  Phase Explanations:")
        for phase, explanation in data.get("Explanations", {}).items():
            print(f"    - {phase}: {explanation}")

        print("-" * 80)

    print("+"*100)
    for interp_name, data in results.items():
        if interp_name in interpretations:
            interpretations[interp_name]["LLM_phases_likelihood"] = data.get("Likelihoods", {})
            interpretations[interp_name]["LLM_phases_explanation"] = data.get("Explanations", {})
            interpretations[interp_name]["LLM_interpretation_likelihood"] = data.get("Interpretation_Likelihood", 0)
            interpretations[interp_name]["LLM_interpretation_explanation"] = data.get("Interpretation_Explanation", "")
    
    
    return [interpretations]
    
