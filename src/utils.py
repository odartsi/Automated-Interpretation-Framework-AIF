import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import ast
import pandas as pd
from pymatgen.core import Composition
import re
import shutil
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go
from dara.result import RefinementResult
import matplotlib.patches as mpatches
from scipy.ndimage import minimum_filter1d


def load_json(path):
    """Load JSON from a file path; raises FileNotFoundError if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r") as f:
        return json.load(f)


def load_csv(path):
    """Load CSV from a file path; raises FileNotFoundError if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def safe_pick_target(filtered_df: pd.DataFrame) -> Optional[str]:
    """Return a usable target column if present, else None."""
    for col in ["Target", "Target.1", "target", "target_1"]:
        if col in filtered_df.columns and not filtered_df[col].isna().all():
            return filtered_df[col].iloc[0]
    return None


SAMPLE_PREFIX_RE = re.compile(r"^(PG_\d+(?:[-_]\d+))_")


def extract_project_number_from_filename(base_name: str) -> str:
    """
    Extract project_number from a filename (without extension).
    e.g. 'PG_0106_1_Ag2O_Bi2O3_200C_60min_uuid' -> 'PG_0106_1'
    Returns underscore-style project_number for CSV join.
    """
    m = SAMPLE_PREFIX_RE.match(base_name)
    if m:
        return m.group(1).replace("-", "_")
    parts = base_name.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return base_name


def get_output_dir( target,project_number,):
    project = project_number.split("_")[0]
    return f"../data/xrd_data/xrd_analysis/{project}/{target}/{project_number}"

def visualize(
    result: RefinementResult,
    diff_offset: bool = False,
    missing_peaks: list[list[float]] | np.ndarray | None = None,
    extra_peaks: list[list[float]] | np.ndarray | None = None,
):
    """Visualize the result from the refinement. It uses plotly as the backend engine."""
    colormap = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    
    plot_data = result.plot_data

    # Create a Plotly figure with size 800x600
    fig = go.Figure()

    # Adding scatter plot for observed data
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y_obs,
            mode="markers",
            marker=dict(color="blue", size=3, symbol="cross-thin-open"),
            name="Observed",
        )
    )

    # Adding line plot for calculated data
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y_calc,
            mode="lines",
            line=dict(color="green", width=2),
            name="Calculated",
        )
    )

    # Adding line plot for background
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y_bkg,
            mode="lines",
            line=dict(color="#FF7F7F", width=2),
            name="Background",
            opacity=0.5,
        )
    )

    diff = np.array(plot_data.y_obs) - np.array(plot_data.y_calc)
    diff_offset_val = 1.1 * max(diff) if diff_offset else 0  # 10 percent below

    # Adding line plot for difference
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=diff - diff_offset_val,
            mode="lines",
            line=dict(color="#808080", width=1),
            name="Difference",
            hoverinfo="skip",  # "skip" to not show hover info for this trace
            opacity=0.7,
        )
    )

    # if there is no phase weight, it will return an empty dictionary (not shown in the legend)
    try:
        weight_fractions = result.get_phase_weights()
    except TypeError:
        weight_fractions = {}

    peak_data = result.peak_data
    max_y = max(np.array(result.plot_data.y_obs) + np.array(result.plot_data.y_bkg))
    min_y_diff = min(
        np.array(result.plot_data.y_obs) - np.array(result.plot_data.y_calc)
    )
    # Adding dashed lines for phases
    for i, (phase_name, phase) in enumerate(plot_data.structs.items()):
        # add area under the curve between the curve and the plot_data["y_bkg"]
        if i >= len(colormap) - 1:
            i = i % (len(colormap) - 1)

        name = (
            f"{phase_name} ({weight_fractions[phase_name] * 100:.2f} %)"
            if len(weight_fractions) > 1
            else phase_name
        )
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=plot_data.y_bkg,
                mode="lines",
                line=dict(color=colormap[i], width=0),
                fill=None,
                showlegend=False,
                hoverinfo="none",
                legendgroup=phase_name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=np.array(phase) + np.array(plot_data.y_bkg),
                mode="lines",
                line=dict(color=colormap[i], width=1.5),
                fill="tonexty",
                name=name,
                visible="legendonly",
                legendgroup=phase_name,
            )
        )
        refl = peak_data[peak_data["phase"] == phase_name]["2theta"]
        intensity = peak_data[peak_data["phase"] == phase_name]["intensity"]
        fig.add_trace(
            go.Scatter(
                x=refl,
                y=np.ones(len(refl)) * (i + 1) * -max_y * 0.1 + min_y_diff,
                mode="markers",
                marker={
                    "symbol": 142,
                    "size": 5,
                    "color": colormap[i],
                },
                name=name,
                legendgroup=phase_name,
                showlegend=False,
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in zip(refl, intensity)],
                hovertemplate="%{text}",
            )
        )
    missing_peaks_table = []
    extra_peaks_table = []
    

    # Define threshold percentages
    thresholds_l = [0, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]

    # Dictionary to store substantial peaks at different thresholds
    missing_peaks_list = []
    extra_peaks_list =[]

    for thresh in thresholds_l:
        threshold_value =  max(plot_data.y_obs)* thresh
        missing_peaks = np.array(missing_peaks).reshape(-1, 2)
        missing_peaks = np.unique(missing_peaks, axis=0) 
        extra_peaks = np.array(extra_peaks).reshape(-1, 2)
        extra_peaks = np.unique(extra_peaks, axis=0) # remove duplicates
        # Filter substantial missing and extra peaks
        substantial_missing = missing_peaks[missing_peaks[:, 1] > threshold_value]
        substantial_extra = extra_peaks[extra_peaks[:, 1] > threshold_value]
        missing_peaks_list.append(len(substantial_missing))
        extra_peaks_list.append(len(substantial_extra))

   
    if missing_peaks is not None:
        missing_peaks = np.array(missing_peaks).reshape(-1, 2)
        missing_peaks = np.unique(missing_peaks, axis=0)  # remove duplicates
        threshold = max(plot_data.y_obs)*0.04 #max(missing_peaks[:, 1])*0.1 #10% of the peak #100  # Define your threshold for substantial intensity
        substantial_peaks = missing_peaks[missing_peaks[:, 1] > threshold]

        num_missing = len(missing_peaks)
        num_substantial_missing = len(substantial_peaks)
        missing_peaks_table = [
            {"2θ": f"{x:.2f}", "Intensity": f"{y:.2f}"}
            for x, y in missing_peaks
        ]
        
        two_theta = missing_peaks[:, 0]
        intensity = missing_peaks[:, 1]

        # Example: Print the extracted columns
        fig.add_trace(
            go.Scatter(
                x=missing_peaks[:, 0],#substantial_peaks
                y=np.zeros_like(missing_peaks[:, 0]),#substantial_peaks
                mode="markers",
                marker=dict(color="#f9726a", symbol=53, size=10, opacity=0.8),
                name="Missing peaks",
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in missing_peaks],#substantial_peaks
                hovertemplate="%{text}",
            )
        )

    if extra_peaks is not None:
        extra_peaks = np.array(extra_peaks).reshape(-1, 2)
        extra_peaks = np.unique(extra_peaks, axis=0) # remove duplicates
        substantial_extra_peaks = extra_peaks[extra_peaks[:, 1] > threshold]
        extra_peaks_table = [
            {"2θ": f"{x:.2f}", "Intensity": f"{y:.2f}"}
            for x, y in extra_peaks
        ]

        # Count the extra peaks
        num_extra = len(extra_peaks)
        num_substantial_extra = len(substantial_extra_peaks)
        fig.add_trace(
            go.Scatter(
                x=extra_peaks[:, 0],#substantial_extra_peaks
                y=np.zeros_like(extra_peaks[:, 0]),#substantial_extra_peaks
                mode="markers",
                marker=dict(color="#335da0", symbol=53, size=10, opacity=0.8),
                name="Extra peaks",
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in extra_peaks],#substantial_extra_peaks
                hovertemplate="%{text}",
            )
        )
    title = f"{result.lst_data.pattern_name} (Rwp={result.lst_data.rwp:.2f}%)"

    fig.update_layout(
    autosize=True,
    xaxis=dict(
        range=[min(plot_data.x), max(plot_data.x)],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        title_font=dict(size=18),
    ),
    title=title,
    xaxis_title="2θ [°]",
    yaxis_title="Intensity",
    legend_title="",
    font=dict(family="Arial, sans-serif", color="RebeccaPurple"),
    plot_bgcolor="white",
    yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, title_font=dict(size=18)),  
    legend_tracegroupgap=1,
    legend=dict(
        font=dict(size=16),  # Increased legend text size
    ), 
    )
    # Annotation below the x-axis
    annotation_text = (
        f"There are {num_missing} missing and {num_extra} extra peaks.<br>"
        f"{num_substantial_missing} of the missing peaks are substantial (intensity > {round(threshold,2)}),<br>"
        f"and {num_substantial_extra} of the extra peaks are substantial (intensity > {round(threshold,2)})."
    )
    
    fig.update_layout(
        annotations=[
            dict(
                text=annotation_text,
                x=1, 
                y=1,  
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
                bgcolor="lavender",
                bordercolor="white",
                borderwidth=1,
            )
        ],
        xaxis=dict(
            title="2θ [°]",
            title_standoff=10, 
        ),
        yaxis=dict(
        domain=[0.3, 1], 
        ),
        margin=dict(
            l=50, 
            r=50, 
            t=50, 
            b=350, 
        ),
    )

    # Add tables for missing and extra peaks
    # Missing peaks table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>2θ</b>", "<b>Intensity</b>"],
                font=dict(size=14, color="white"),
                fill_color="#f9726a",  # Color for missing peaks
                align="center",
            ),
            cells=dict(
                values=[
                    [row["2θ"] for row in missing_peaks_table], 
                    [row["Intensity"] for row in missing_peaks_table],
                ],
                font=dict(size=14),
                fill_color="lavender",
                align="center",
            ),
            domain=dict(x=[0, 0.45], y=[0, 0.22]), 
        )
    )

    # Extra peaks table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>2θ</b>", "<b>Intensity</b>"],
                font=dict(size=14, color="white"),
                fill_color="#335da0",  # Color for extra peaks
                align="center",
            ),
            cells=dict(
                values=[
                    [row["2θ"] for row in extra_peaks_table], 
                    [row["Intensity"] for row in extra_peaks_table],
                ],
                font=dict(size=14),
                fill_color="lavender",
                align="center",
            ),
            domain=dict(x=[0.55, 1], y=[0, 0.22]),  
        )
    )
  
    fig.add_hline(y=0, line_width=1)

    # Add ticks
    fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor="black", ticklen=10)
    fig.update_yaxes(ticks="outside", tickwidth=1, tickcolor="black", ticklen=10)

    return fig 

def celsius_to_kelvin(celsius):
    """
    Convert Celsius to Kelvin.

    Parameters:
    celsius (float or int): Temperature in degrees Celsius.

    Returns:
    float: Temperature in Kelvin.
    """
    return str(float(celsius) + 273.15)

def remove_elements(all_cifs, elements_to_remove):
    """
    Removes elements from all_cifs that match any element in elements_to_remove after transformations.
    
    Args:
        all_cifs (list): A list of PosixPath objects.
        elements_to_remove (list): A list of strings representing elements to remove after transformation.
    
    Returns:
        list: A filtered list of PosixPath objects from all_cifs.
    """
    transformed_all_cifs = transform_all_cifs(all_cifs)
    
    return [
        all_cifs[i] for i in range(len(all_cifs))
        if transformed_all_cifs[i] not in elements_to_remove
    ]

def include_elements(all_cifs, elements_to_include):
    """
    Filters all_cifs to include only elements that match any element in elements_to_include after transformations.

    Args:
        all_cifs (list): A list of PosixPath objects.
        elements_to_include (list): A list of lists containing elements to include.

    Returns:
        list: A filtered list of PosixPath objects from all_cifs.
    """
    # Exclude elements that are NOT in elements_to_include
    elements_to_include= [element.split('_(icsd')[0] for element in elements_to_include]

    transformed_all_cifs = transform_all_cifs(all_cifs)
    return [
        all_cifs[i] for i in range(len(all_cifs))
        if transformed_all_cifs[i] in elements_to_include
    ]
    

def cleanup_phases(phases):
    new_phases = []
    for phase in phases:
        split_at_first_underscore = phase.split('_', 1)
        chemical_formula = split_at_first_underscore[0]
        space_group_number = split_at_first_underscore[1].split('_', 1)[0]
        comp = Composition(chemical_formula)
        combined = comp.get_integer_formula_and_factor(max_denominator=10)[0] + "_" + space_group_number
        new_phases.append(str(combined))
    return new_phases

def cleanup_phases_only_formula(phases):
    new_phases = []
    for phase in phases:
        split_at_first_underscore = phase.split('_', 1)
        chemical_formula = split_at_first_underscore[0]
        comp = Composition(chemical_formula)
        combined = comp.get_integer_formula_and_factor(max_denominator=10)[0]
        new_phases.append(str(combined))
    return new_phases

def remove_cifs_prefix(file_list):
    """
    Removes the 'cifs/' prefix from each string in the list.
    
    Args:
        file_list (list): A list of strings with 'cifs/' prefixes.
        
    Returns:
        list: A new list with 'cifs/' prefixes removed.
    """
    return [item.replace('cifs/', '', 1)for item in file_list]

def remove_cifs_suffix(file_list):
    """
    Removes the 'cifs/' prefix from each string in the list.
    
    Args:
        file_list (list): A list of strings with 'cifs/' prefixes.
        
    Returns:
        list: A new list with 'cifs/' prefixes removed.
    """
    return [item.replace('.cif', '', 1)for item in file_list]


def extract_icsd_from_key(phase_key: str) -> str | None:
    """
    Extract the ICSD id from phase keys like 'ZrO2_14_(icsd_157403)-0' or 'Zn1.96O2_186_(icsd_13952)-None'.
    Returns the digits after 'icsd_' or 'icsd-', or None if no ICSD id is found.
    """
    m = re.search(r'icsd[_-]?(\d+)', phase_key, re.I)
    return m.group(1) if m else None


def strip_phase_identifier(phase_name: str) -> str:
    """
    Remove the ICSD suffix from a phase name: everything from '_(' onward is dropped.
    Example: 'V2O3_167_(icsd_1869)-0' -> 'V2O3_167'.
    """
    return re.split(r'_\(', phase_name)[0]


def transform_all_cifs(all_cifs):
    """
    Transforms the all_cifs list into a comparable format by removing the 'cifs/' prefix and cleaning up phases.
    
    Args:
        all_cifs (list): A list of PosixPath objects pointing to CIF files.
    
    Returns:
        list: A new list of transformed strings representing cleaned-up phases.
    """
    transformed = []
    for cif in all_cifs:
        # Convert PosixPath to string and remove the 'cifs/' prefix
        cif_str = str(cif).replace('cifs/', '')
        # Apply cleanup_phases logic
        split_at_first_underscore = cif_str.split('_', 1)
        chemical_formula = split_at_first_underscore[0]
        space_group_number = split_at_first_underscore[1].split('_', 1)[0]
        comp = Composition(chemical_formula)
        combined = comp.get_integer_formula_and_factor(max_denominator=10)[0] + "_" + space_group_number
        transformed.append(str(combined))
    return transformed


def cleanup_cifs():
    folder_name = "cifs"
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
        print(f"Folder '{folder_name}' has been deleted.")
    else:
        print(f"Folder '{folder_name}' does not exist.")

def type_of_furnace(furnace):
    if furnace == 'BF':
        furnace = 'Box furnace with ambient air'
    elif furnace == 'TF-Ar':
        furnace = 'Tube furnace with flowing Argon (flow rate unknown)'
    elif furnace == 'TF-Ar + H2':
        furnace = 'Tube furnace with flowing Argon and Hydrogen (flow rate unknown)'
    elif furnace == 'TF-O2':
        furnace = 'Tube furnace with flowing Oxygen (flow rate unknown)'
    return furnace

def save_xrd_plots(search_results, final_results, interpretation_number, pattern_path, missing_peaks, extra_peaks, target): 
    base_name = os.path.basename(pattern_path)
    base_name = os.path.splitext(base_name)[0]
    try:
        if "-" in base_name and "_" in base_name:
            first_part = base_name.split('-')[0]
            second_part = base_name.split('-')[1].split('_')[0]
            project_number = first_part + "_" + second_part
        elif "_" in base_name:
            parts = base_name.split('_')
            if len(parts) >= 2:
                project_number = parts[0] + "_" + parts[1]
            else:
                project_number = base_name
        elif "-" in base_name:
            parts = base_name.split('-')
            if len(parts) >= 2:
                project_number = parts[0] + "_" + parts[1]
            else:
                project_number = base_name
        else:
            project_number = base_name

    except IndexError:
        project_number = "Invalid format"


    # Ensure the output directory exists
    # output_dir = f"../data/xrd_data/xrd_analysis/{target}/{project_number}"
    output_dir = get_output_dir(target, project_number)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final results plot
    
    final_fig = visualize(final_results,False,missing_peaks, extra_peaks)

    # Create the filename for final results
    interpretation_number = int(interpretation_number.replace('I_', ''))

    final_filename_html = f"{project_number}_I_{interpretation_number}.html"
    final_filename_png = f"{project_number}_I_{interpretation_number}.png"
    final_file_path_html = os.path.join(output_dir, final_filename_html)
    final_file_path_png = os.path.join(output_dir, final_filename_png)

    # Save interactive HTML
    final_fig.write_html(final_file_path_html)

    # Save static PNG with larger size and higher resolution
    try:
        final_fig.write_image(final_file_path_png, width=1600, height=900, scale=3)
        print(f"Final result plot saved as: {final_filename_png}")
    except ValueError as e:
        print(f"[Warning] Failed to export PNG for {final_filename_png}: {e}")
        # Optionally: save fallback format like HTML or SVG
        fallback_svg = final_file_path_png.replace(".png", ".svg")
        try:
            final_fig.write_image(fallback_svg)
            print(f"Saved fallback SVG for failed PNG: {fallback_svg}")
        except Exception as svg_err:
            print(f"[Error] Fallback SVG export also failed: {svg_err}")
    
def importance_factor_calculation(interpretations):
    """
    Calculate P(S | I) by directly normalizing the ratio (score / Rwp)
    using min-max scaling.

    Args:
        interpretations (dict): A dictionary where each key is an interpretation name and 
                                each value contains at least "rwp" and "score".

    Returns:
        dict: Updated interpretations with scaled interpretation_importance.
    """
    # Extract Rwp and Score values
    rwp_values = np.array([data["rwp"] for data in interpretations.values()])
    score_values = np.array([data["score"] for data in interpretations.values()])

    # Compute the ratio score / Rwp
    ratio_values = score_values / (rwp_values + 1e-8)  # Avoid division by zero

    # Min-Max Normalize the ratio
    min_ratio, max_ratio = ratio_values.min(), ratio_values.max()
   # TODO THINk about min_ration value when all rwp and score are the smallest I=0%
    if max_ratio - min_ratio > 0:
        normalized_importance = (ratio_values - min_ratio) / (max_ratio - min_ratio) * 100
    else:
        normalized_importance = np.zeros_like(ratio_values)  # If all values are the same, set all to 0

    # Store in interpretations
    for interp, data, imp_value in zip(interpretations.keys(), interpretations.values(), normalized_importance):
        data["interpretation_importance"] = imp_value

    return interpretations
def calculate_prior_probability(interpretations, w_llm=1, w_bscore=1):
    # Update each interpretation with a new field "prior_probability"
    for interpretation_name, interpretation in interpretations.items():
        interpretation["prior_probability"] = (
           (interpretation["LLM_interpretation_likelihood"]*w_llm + interpretation["balance_score"]*w_bscore)/(w_llm + w_bscore)
        )
    return interpretations

def calculate_posterior_probability_of_interpretation(interpretations):
    """
    Calculate the posterior probabilities P(In | S) for all interpretations
    using balance_score and interpretation_importance.
    
    Parameters:
        interpretations (dict or list): A dictionary of interpretations or 
                                        a list containing a single dictionary.

    Returns:
        dict: Updated interpretations with added "posterior_probability" field.
    """
    # Extract dictionary if interpretations is a list
    if isinstance(interpretations, list) and len(interpretations) == 1:
        interpretations = interpretations[0]  # Extract the dictionary

    if not isinstance(interpretations, dict):
        raise ValueError("Interpretations should be a dictionary or a list containing a dictionary.")

    # Step 1: Compute joint = prior * fit_quality
    joint_probabilities = {}
    for name, interp in interpretations.items():
        joint = interp["prior_probability"] * interp["fit_quality"]
        joint_probabilities[name] = joint

    # Step 2: Normalize to get posterior_probability
    total_joint = sum(joint_probabilities.values())
    for name in interpretations:
        if total_joint > 0:
            interpretations[name]["posterior_probability"] = joint_probabilities[name] / total_joint
        else:
            interpretations[name]["posterior_probability"] = 0.0

    # Optional: Downweight factor for untrustworthy results
    TRUST_PENALTY_FACTOR = 0.1
    for interpretation_name, interpretation in interpretations.items():
        is_trustworthy = interpretation.get("trustworthy", True)
        penalty = 1.0 if is_trustworthy else TRUST_PENALTY_FACTOR
        interpretation["adjusted_posterior_probability"] = (
            interpretation["posterior_probability"] * penalty
        )
    return interpretations

def calculate_phase_probabilities(interpretations):
    """
    Calculate and scale the probability of each individual phase P(f_i | S) between 0 and 100.

    Parameters:
        interpretations (dict or list): Dictionary containing interpretations with `posterior_probability`
                                        and `phases` (phases present in each interpretation).

    Returns:
        dict: Dictionary with the probability of each phase P(f_i | S), scaled between 0 and 100.
    """
    # Extract dictionary if interpretations is a list
    if isinstance(interpretations, list) and len(interpretations) == 1:
        interpretations = interpretations[0]  # Extract the dictionary

    if not isinstance(interpretations, dict):
        raise ValueError("Interpretations should be a dictionary or a list containing a dictionary.")

    phase_probabilities = {}

    # Flatten all unique phases across interpretations
    all_phases = set(phase for interp in interpretations.values() for phase in interp["phases"])
    # Calculate raw phase probabilities
    for phase in all_phases:
        phase_probability = 0
        for interpretation_name, interpretation in interpretations.items():
            if phase in interpretation["phases"]:
                # Add P(I_n | S) to the phase's probability if it's in the interpretation
                phase_probability += interpretation.get("posterior_probability", 0)
        phase_probabilities[phase] = phase_probability*100  # Store raw value
 
    return phase_probabilities
   


def normalize_rwp_for_sample(interpretations, max_rwp=60):
    """Apply softclip linear normalization to RWP values in the interpretation."""
    if not isinstance(interpretations, dict):
        raise ValueError("Expected a dict of interpretations.")

    for key, interp in interpretations.items():
        rwp = interp.get("rwp")
        if rwp is not None:
            # Softclip linear: score = 1 - (rwp / max_rwp), clipped to [0, 1]
            interp["normalized_rwp"] = float(np.clip((max_rwp - rwp) / max_rwp, 0, 1))
        else:
            interp["normalized_rwp"] = None

    return interpretations



def scaled_sigmoid(s, k=7, center=0.2):
    """Sigmoid scaled so that score=1 maps exactly to normalized=1."""
    raw = 1 / (1 + np.exp(-k * (s - center)))
    max_val = 1 / (1 + np.exp(-k * (1 - center)))
    return raw / max_val

def normalize_scores_for_sample(interpretations, k=3, center=0.3):
    """Apply scaled sigmoid normalization to score values in the interpretation."""
    if not isinstance(interpretations, dict):
        raise ValueError("Expected a dict of interpretations.")

    for key, interp in interpretations.items():

        score = interp.get("score")
        if score is not None:
            interp["normalized_score"] = float(scaled_sigmoid(score, k=k, center=center))
        else:
            interp["normalized_score"] = None

    return interpretations


def calculate_fit_quality(interpretations, w_rwp=1, w_score=1):
    """
    Compute a combined fit quality score per interpretation from normalized RWP and normalized score.

    Adds/overwrites `fit_quality` for each interpretation:
      fit_quality = (normalized_rwp*w_rwp + normalized_score*w_score) / (w_rwp + w_score)
    """
    for _, interpretation in interpretations.items():
        interpretation["fit_quality"] = (
            interpretation["normalized_rwp"] * w_rwp
            + interpretation["normalized_score"] * w_score
        ) / (w_rwp + w_score)
    return interpretations


def flag_interpretation_trustworthiness_(interpretations: dict) -> dict:
    """
    Adds a 'trustworthy' field to each interpretation in the dictionary based on custom flagging rules.
    
    Parameters:
        interpretations (dict): A dictionary of interpretation entries (I_1, I_2, ...) where each entry is a dictionary of properties.
    
    Returns:
        dict: The updated dictionary with a new key 'trustworthy' added to each interpretation.
    """
    
    for key, interp in interpretations.items():
        balance_score = interp.get("balance_score", 1.0)
        llm = interp.get("LLM_interpretation_likelihood", 1.0)
        signal_above_bkg_score =interp.get("signal_above_bkg_score", 0.0)
        bkg_overshoot_score =interp.get("bkg_overshoot_score", 0.0)
        peak_match_score = interp.get("normalized_score",1.0)

        # Apply your custom flags
        flag1 = llm <= 0.4
        if signal_above_bkg_score !=0.0:
            flag2 = signal_above_bkg_score < 9000
        else: 
            flag2 = False
        if bkg_overshoot_score !=0.0:
            flag3 = bkg_overshoot_score > 1200
        else: 
            flag3 = False

        if bkg_overshoot_score !=0.0 and signal_above_bkg_score !=0.0:
            flag4 = signal_above_bkg_score/bkg_overshoot_score < 15
        else: 
            flag4 = False

        flag5 = balance_score < 0.6
        # flag6 = peak_match_score < 0.6

        # If any flag is True, it's not trustworthy
        interp["trustworthy"] = not (flag1 or flag2 or flag3 or flag4 or flag5 ) #or flag6)

    return interpretations

def flag_interpretation_trustworthiness(
    interpretations: dict,
    trust_threshold: float = 0.65
) -> dict:
    """
    Adds a boolean 'trustworthy' field to each interpretation based solely on trust_score.

    Parameters:
        interpretations (dict): Dictionary of interpretation entries (I_1, I_2, ...).
        trust_threshold (float): Minimum trust_score required to be considered trustworthy.

    Returns:
        dict: Updated interpretations with 'trustworthy' key added.
    """

    for key, interp in interpretations.items():
        try:
            trust_score = float(interp.get("trust_score", 0.0))
        except (TypeError, ValueError):
            trust_score = 0.0

        interp["trustworthy"] = trust_score >= trust_threshold

    return interpretations
def compute_trust_score(
    interpretations: dict,
    *,
    llm_ref=0.41,
    signal_ref=9000.0,
    overshoot_ref=1200.0,
    ratio_ref=15.0,
    balance_ref=0.6,
    rwp_ref=15.0,              # NEW: ideal rwp <= 15
    include_peak_match=False,  # optional, off by default
    peak_match_ref=0.6,
    decimals=6
) -> dict:
    """
    Adds interp['trust_score'] in [0,1] using soft penalties.

    Criteria (penalty in [0,1], 0=good, 1=bad):
      1) LLM likelihood: ideal >= llm_ref
      2) Signal above background: ideal >= signal_ref
      3) Background overshoot: ideal <= overshoot_ref
      4) Signal/overshoot ratio: ideal >= ratio_ref
      5) Balance score: ideal >= balance_ref
      6) Rwp: ideal <= rwp_ref   (penalize only if rwp > rwp_ref)

    Optionally add peak match (normalized_score) as an extra criterion if include_peak_match=True.

    Trust score = 1 - mean(penalties).
    """

    for key, interp in interpretations.items():
        try:
            llm = float(interp.get("LLM_interpretation_likelihood", 1.0))
            signal = float(interp.get("signal_above_bkg_score", 10000.0))
            overshoot = float(interp.get("bkg_overshoot_score", 0.0))
            balance = float(interp.get("balance_score", 1.0))
            rwp = float(interp.get("rwp", 0.0))  # NEW
            score = float(interp.get("normalized_score", 1.0))  # parsed for optional use
        except (TypeError, ValueError):
            interp["trust_score"] = 0.0
            continue

        # 1) LLM likelihood (ideal >= llm_ref)
        penalty_llm = max(0.0, min(1.0, (llm_ref - llm) / llm_ref))

        # 2) Signal above background (ideal >= signal_ref)
        penalty_signal = max(0.0, min(1.0, (signal_ref - signal) / signal_ref))

        # 3) Background overshoot (ideal <= overshoot_ref)
        penalty_overshoot = (
            max(0.0, min(1.0, (overshoot - overshoot_ref) / overshoot_ref))
            if overshoot > 0 else 0.0
        )

        # 4) Signal / overshoot ratio (ideal >= ratio_ref)
        if overshoot > 0:
            ratio = signal / overshoot
            penalty_ratio = max(0.0, min(1.0, (ratio_ref - ratio) / ratio_ref))
        else:
            penalty_ratio = 0.0  # no penalty if overshoot is 0

        # 5) Balance score (ideal >= balance_ref)
        penalty_balance = max(0.0, min(1.0, (balance_ref - balance) / balance_ref))

        # 6) Rwp (ideal <= rwp_ref)  <-- NEW
        # scale linearly: at rwp_ref => 0; at 2*rwp_ref => 1; above that clipped to 1
        penalty_rwp = max(0.0, min(1.0, (rwp - rwp_ref) / rwp_ref))

        penalties = [
            penalty_llm,
            penalty_signal,
            penalty_overshoot,
            penalty_ratio,
            penalty_balance,
            penalty_rwp,
        ]

        # Optional: Peak match score (ideal >= peak_match_ref)
        if include_peak_match:
            penalty_peak = max(0.0, min(1.0, (peak_match_ref - score) / peak_match_ref))
            penalties.append(penalty_peak)

        total_penalty = float(np.mean(penalties))
        interp["trust_score"] = round(max(0.0, 1.0 - total_penalty), decimals)

    return interpretations
def compute_trust_score_(interpretations: dict) -> dict:
    """
    Adds a 'trust_score' field to each interpretation, ranging from 0 (not trustworthy) to 1 (fully trustworthy),
    based on six soft criteria:
        - LLM likelihood (>= 0.4)
        - Signal above background score (>= 9000)
        - Background overshoot score (<= 1200)
        - Signal-to-overshoot ratio (>= 15)
        - Balance score (>= 0.6)
        - Peak match score (normalized_score >= 0.6)

    Trust score is 1 - average of penalties (each in [0, 1]).
    """

    for key, interp in interpretations.items():
        try:
            llm = float(interp.get("LLM_interpretation_likelihood", 1.0))
            signal = float(interp.get("signal_above_bkg_score", 10000.0))
            overshoot = float(interp.get("bkg_overshoot_score", 0.0))
            balance = float(interp.get("balance_score", 1.0))
            # score = float(interp.get("normalized_score", 1.0))
        except (TypeError, ValueError):
            interp["trust_score"] = 0.0
            continue

        # Flag 1: LLM likelihood (ideal > 0.4)
        penalty_llm = max(0.0, min(1.0, (0.41 - llm) / 0.41))

        # Flag 2: Signal above background (ideal >= 9000)
        penalty_signal = max(0.0, min(1.0, (9000 - signal) / 9000))

        # Flag 3: Background overshoot (ideal <= 1200)
        penalty_overshoot = max(0.0, min(1.0, (overshoot - 1200) / 1200)) if overshoot > 0 else 0.0

        # Flag 4: Signal / overshoot ratio (ideal >= 15)
        if overshoot > 0:
            ratio = signal / overshoot
            penalty_ratio = max(0.0, min(1.0, (15 - ratio) / 15))
        else:
            penalty_ratio = 0.0  # no penalty if overshoot is 0

        # Flag 5: Balance score (ideal >= 0.6)
        penalty_balance = max(0.0, min(1.0, (0.6 - balance) / 0.6))

        # Flag 6: Peak match score (ideal >= 0.6)
        # penalty_score = max(0.0, min(1.0, (0.6 - score) / 0.6))

        # Aggregate penalty and compute trust score
        total_penalty = (
            penalty_llm + penalty_signal + penalty_overshoot +
            penalty_ratio + penalty_balance #+ penalty_score
        ) / 6.0

        interp["trust_score"] = round(max(0.0, 1.0 - total_penalty), 3)

    return interpretations

def calculate_excess_bkg(plot_data, peak_window=2, top_n_peaks=3, low_angle=(20,35), high_angle=60): #10, 40
    observed = np.asarray(plot_data.y_obs)
    background = np.asarray(plot_data.y_bkg)
    angles = np.asarray(plot_data.x)

    def max_localized_excess(region_mask):
        # localized_excess = np.maximum(0, background[region_mask] - observed[region_mask])
        # total_signal = np.sum(observed[region_mask])
        # max_excess = np.max(localized_excess)
        # return (max_excess * 100 / total_signal) if total_signal != 0 else np.nan
        max_local_excess = np.max(np.maximum(0, background[region_mask] - observed[region_mask]))
        local_intensity = observed[region_mask][np.argmax(np.maximum(0, background[region_mask] - observed[region_mask]))]

        max_excess_local_percentage = (max_local_excess * 100 / local_intensity) if local_intensity != 0 else np.nan
        return max_excess_local_percentage

    # Option 1: High-intensity peak regions
    top_peaks_indices = np.argsort(observed)[-top_n_peaks:]
    peak_region_mask = np.zeros_like(observed, dtype=bool)

    for idx in top_peaks_indices:
        peak_angle = angles[idx]
        peak_region_mask |= (angles >= peak_angle - peak_window) & (angles <= peak_angle + peak_window)

    max_high_intensity = max_localized_excess(peak_region_mask)

    # Option 2: Low-angle region
    low_angle_mask = (angles >= low_angle[0]) & (angles <= low_angle[1])
    max_low_angle = max_localized_excess(low_angle_mask)

    # Option 3: High-angle region
    high_angle_mask = angles >= high_angle
    max_high_angle = max_localized_excess(high_angle_mask)

    return {
        'high_intensity_peaks': max_high_intensity,
        'low_angle_region': max_low_angle,
        'high_angle_region': max_high_angle
    }

def signal_above_bkg_score(plot_data, angle_window=(10, 70)):
    observed = np.asarray(plot_data.y_obs)
    background = np.asarray(plot_data.y_bkg)
    observed= np.asarray(observed)
    background = np.asarray(background)
    angles = np.asarray(plot_data.x)
    region_mask = (angles >= angle_window[0]) & (angles <= angle_window[1])
    obs, bkg, theta = observed[region_mask], background[region_mask], angles[region_mask]

    delta_theta = theta[-1] - theta[0]
    score = np.sum(np.maximum(obs, bkg) - bkg) / (delta_theta + 1e-8)
    # score1_norm = score1 / np.max(plot_data.y_obs)
    return score

def bkg_overshoot_score(plot_data, angle_window=(10, 70)):
    observed = np.asarray(plot_data.y_obs)
    background = np.asarray(plot_data.y_bkg)
    angles = np.asarray(plot_data.x)

    region_mask = (angles >= angle_window[0]) & (angles <= angle_window[1])
    obs, bkg, theta = observed[region_mask], background[region_mask], angles[region_mask]

    delta_theta = theta[-1] - theta[0]
    score = np.sum(np.maximum(bkg - obs, 0)) / (delta_theta + 1e-8)
    return score


def estimate_baseline(y, window_size=50):
    """A non-parametric baseline: the minimum in each sliding window."""
    return minimum_filter1d(y, size=window_size, mode='reflect')

def extreme_diff(plot_data, angle_window=(10,70),
                 p_low=5, p_high=95):
    angles = np.asarray(plot_data.x)
    mask = (angles >= angle_window[0]) & (angles <= angle_window[1])
    obs = np.asarray(plot_data.y_obs)[mask]
    bkg = np.asarray(plot_data.y_bkg)[mask]

    low_obs  = np.percentile(obs,  p_low)   # e.g. 5th percentile
    high_bkg = np.percentile(bkg, p_high)  # e.g. 95th percentile
    
    return low_obs - high_bkg

def bkg_underfit_score(plot_data,
                       angle_window=(10, 70),
                       perc=10):
    """
    Returns the `perc`-th percentile of (obs - bkg) in the window.
    If this is *big* (> noise level), your background is too low
    across most of the pattern.
    
    Parameters
    ----------
    plot_data : object with attributes x, y_obs, y_bkg
    angle_window : tuple of (θ_min, θ_max)
    perc : percentile (0–100) to use; e.g. 10 means the 10th percentile
    """
    angles = np.asarray(plot_data.x)
    mask   = (angles >= angle_window[0]) & (angles <= angle_window[1])
    obs    = np.asarray(plot_data.y_obs)[mask]
    bkg    = np.asarray(plot_data.y_bkg)[mask]

    diffs = obs - bkg
    return np.percentile(diffs, perc)

def plot_phase_and_interpretation_probabilities(interpretations, project_number, df, target):
    """
    Plot phase probabilities and interpretation probabilities.
    - Style: Bigger bars, tighter spacing, Right-Aligned labels (matching phases).
    - Palette: Specific user-defined NPG colors + Deep Teal/Green highlights.
    - Logic: Original precursor detection.
    """
    
    # --- Helper Functions ---
    def clean_phase_name(phase):
        return phase.split("_(icsd")[0]

    def cleanup_phases(phase_list):
        return phase_list

    def clean_chemical_formula(name):
        return re.sub(r"\s*\(.*?\)", "", name).strip()

    def format_phase_display(name):
        # Converts 'Phase_123' to 'Phase (sg #123)'
        match = re.search(r"(.+)_(\d+)$", name)
        if match:
            return f"{match.group(1)} (sg #{match.group(2)})"
        return name

    
    phase_color_palette = ["#91D1C2","#00468B", "#F3E5AB",  "#D191A0",  
        "#b2b223",  "#98a8b8", ]
    
    interpretation_colors = {
    "I_1": "#3C5488",
    "I_2": "#7E8DA4", 
    "I_3": "#BFD3E6",
    "I_4": "#D9D9D9",
    "I_5": "#f3d57f", 
    "I_6": "#F39B7F"
    
}

    default_interpretation_color = "#4C566A"


    # Highlight Colors
    color_AIF_best  = "#005F73"  
    color_Rwp_lowest = "#00A087"  
  

    # --- Precursor Logic (Original) ---
    precursors_raw = df['Precursors'].values[0]
    if isinstance(precursors_raw, str) and precursors_raw.startswith("["):
        precursors_list = ast.literal_eval(precursors_raw)
    else:
        precursors_list = precursors_raw
        
    precursors = set(clean_chemical_formula(p) for p in precursors_list)

    def is_precursor(phase_name):
        base_phase = phase_name.split("_")[0]
        # Original logic: exact split match
        return any(base_phase == precursor.split("_")[0] for precursor in precursors)
    
    # Clean phases
    for interp in interpretations:
        interpretations[interp]["phases"] = cleanup_phases([clean_phase_name(p) for p in interpretations[interp]["phases"]])

    unique_phases = sorted(set(clean_phase_name(phase) for interp in interpretations.values() for phase in interp["phases"]))
    unique_phases = cleanup_phases(unique_phases)
    phase_colors = dict(zip(unique_phases, phase_color_palette[:len(unique_phases)]))
    

    # Compute probabilities
    phase_probabilities = calculate_phase_probabilities(interpretations)
    
    interpretation_names = list(interpretations.keys())
    posterior_values = [interpretations[interp]["posterior_probability"] * 100 for interp in interpretation_names]
    rwp_values = [interpretations[interp]["rwp"] for interp in interpretation_names]
    
    sorted_indices = np.argsort(posterior_values)[::-1]
    sorted_interpretations = [interpretation_names[i] for i in sorted_indices]
    sorted_posterior_values = [posterior_values[i] for i in sorted_indices]
    sorted_rwp_values = [rwp_values[i] for i in sorted_indices]

    I_best = sorted_interpretations[np.argmax(sorted_posterior_values)]
    I_dara = sorted_interpretations[np.argmin(sorted_rwp_values)]
    
    # Assign lowest Rwp color for legend consistency
    interpretation_colors[I_dara] = color_Rwp_lowest

    # -------------------------------------------------------------------------
    # PLOT 1: Interpretation Probabilities
    # -------------------------------------------------------------------------
    
    sorted_weight_fractions = [interpretations[interp]["weight_fraction"] for interp in sorted_interpretations]
    sorted_phases = [interpretations[interp]["phases"] for interp in sorted_interpretations]

    # STYLE: Tighter vertical spacing (0.6 factor) + wider figure (11)
    fig, ax = plt.subplots(figsize=(11, max(4, len(sorted_interpretations) * 0.6)))
    plt.subplots_adjust(bottom=0.151)
    plt.subplots_adjust(left=0.25, right=0.95)

    bottom_bar = np.zeros(len(sorted_interpretations))
    trust_flags = {interp: interpretations[interp].get("trustworthy", True) for interp in sorted_interpretations}
    total_bar_widths = defaultdict(float)

    for phase_idx, phase in enumerate(unique_phases):
        fraction_values = [
            (weights[phases.index(phase)] if phase in phases else 0) * post_prob / 100
            for phases, weights, post_prob in zip(sorted_phases, sorted_weight_fractions, sorted_posterior_values)
        ]

        bars = ax.barh(
            [f"Interpretation_{name.split('_')[1]}" for name in sorted_interpretations[::-1]],
            fraction_values[::-1],
            left=bottom_bar[::-1],
            color=phase_colors[phase],
            edgecolor="none",
            height=0.8 
        )
        bottom_bar += fraction_values
        
        for bar, interp_name in zip(bars, sorted_interpretations[::-1]):
             total_bar_widths[interp_name] += bar.get_width()

    # --- Labels ---
    ax.set_yticks([]) 
    ax.axvline(0, color="black", linewidth=1, zorder=3)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    trans = ax.get_yaxis_transform() 

    for y_idx, interp_name in enumerate(sorted_interpretations[::-1]):
        is_trustworthy = trust_flags[interp_name]
        symbol = "✔" if is_trustworthy else "✘"
        symbol_color = "green" if is_trustworthy else "red"
        
        display_name = f"Interpretation {interp_name.split('_')[1]}"
        text_color = color_Rwp_lowest if interp_name == I_dara else (color_AIF_best if interp_name == I_best else "black")
        weight = "bold"

        # STYLE: Right Aligned Labels (Closer to axis, matching Phase plot)
        # 1. Name
        ax.text(-0.02, y_idx, display_name, color=text_color, 
                fontsize=14, fontweight=weight, ha='right', va='center', transform=trans)

        # 2. Symbol (Dynamic offset based on name length to sit "in front")
        # Heuristic: ~0.012 units per character is a rough estimate for fontsize 14
        symbol_offset = -0.02 - (len(display_name) * 0.013) - 0.02
        ax.text(symbol_offset, y_idx, symbol, color=symbol_color, 
                fontsize=16, fontweight='bold', ha='right', va='center', transform=trans)

        # 3. Subtitles
        if interp_name == I_dara:
            ax.text(-0.02, y_idx - 0.35, "(Lowest Rwp)", color="black", 
                    fontsize=12, ha='right', va='top', transform=trans)
        if interp_name == I_best:
             ax.text(-0.02, y_idx - 0.35, "(AIF)", color="black", 
                    fontsize=12, ha='right', va='top', transform=trans)

    # Percentages
    for interp_name, value in zip(sorted_interpretations[::-1], sorted_posterior_values[::-1]):
         y_pos = sorted_interpretations[::-1].index(interp_name)
         ax.text(total_bar_widths[interp_name] + 1.5, y_pos, f"{value:.1f}%", va="center", ha="left", fontsize=12, color="black")

    ax.set_xlabel("Interpretation Probability (%)", fontsize=14)
    ax.set_xlim(left=0, right=max(sorted_posterior_values) + 15)
    ax.tick_params(axis="x", labelsize=12)
    
    # Legend
    legend_handles = []
    for phase in unique_phases:
        display = format_phase_display(phase)
        if is_precursor(phase):
            display += "\n(unreacted precursor)"
        
        handle = plt.Line2D([0], [0], color=phase_colors.get(phase, "black"), lw=6, label=display)
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="lower right", fontsize=10, frameon=True, edgecolor="black")

    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir, "interpretation_probabilities.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 2: Phase Probabilities
    # -------------------------------------------------------------------------

    sorted_phases_by_prob = sorted(phase_probabilities.items(), key=lambda x: x[1], reverse=True)
    cleaned_phases_list, probabilities = zip(*[(clean_phase_name(p), v) for p, v in sorted_phases_by_prob]) if sorted_phases_by_prob else ([], [])

    # STYLE: Tighter vertical spacing matching Plot 1
    fig, ax = plt.subplots(figsize=(11, max(4, len(cleaned_phases_list) * 0.6)))
    plt.subplots_adjust(bottom=0.18)
    plt.subplots_adjust(left=0.25, right=0.95)

    bottom_bar = np.zeros(len(cleaned_phases_list))
    total_bar_widths_phase = defaultdict(float)

    for interp in sorted_interpretations:
        fraction_values = [
            (interpretations[interp]["posterior_probability"] * 100 if phase in [clean_phase_name(p) for p in interpretations[interp]["phases"]] else 0)
            for phase in cleaned_phases_list
        ]
        
        color = color_AIF_best if interp == I_best else interpretation_colors.get(interp, default_interpretation_color)
        hatch = "xx" if interp == I_best else ""

        bars = ax.barh(cleaned_phases_list[::-1], fraction_values[::-1], left=bottom_bar[::-1], 
                       color=color, hatch=hatch, height=0.8) # STYLE: Bigger bars
        
        for bar, phase in zip(bars, cleaned_phases_list[::-1]):
            total_bar_widths_phase[phase] += bar.get_width()

        bottom_bar += fraction_values

    # --- Labels ---
    ax.set_yticks([]) 
    ax.axvline(0, color="black", linewidth=1, zorder=3)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    trans = ax.get_yaxis_transform()

    for y_idx, phase in enumerate(cleaned_phases_list[::-1]):
        display_name = format_phase_display(phase)
        
        # Name
        ax.text(-0.02, y_idx, display_name, color="black", 
                fontsize=14, fontweight='bold', ha='right', va='center', transform=trans)
        
        # Subtitle
        if is_precursor(phase):
            ax.text(-0.02, y_idx - 0.35, "(unreacted precursor)", color="black", 
                    fontsize=11,  ha='right', va='top', transform=trans)

    # Value Labels
    for phase in cleaned_phases_list:
        y_pos = cleaned_phases_list[::-1].index(phase)
        total_w = total_bar_widths_phase[phase]
        ax.text(total_w + 1.5, y_pos, f"{total_w:.1f}%", va="center", ha="left", fontsize=12)

    ax.set_xlabel("Phase Probability (%)", fontsize=14)
    ax.set_xlim(left=0, right=max(probabilities) + 15)
    ax.tick_params(axis="x", labelsize=12)

    # Legend
    legend_handles = []
    for interp in interpretation_names:
        color = color_AIF_best if interp == I_best else interpretation_colors.get(interp, default_interpretation_color)
        
        # Use Lowest Rwp color for that specific interpretation
        if interp == I_dara:
            color = color_Rwp_lowest
            
        interp_num = interp.split("_")[1]
        interp_label = f"Interp {interp_num}"

        label_txt = (
            f"{interp_label}\n(AIF)"
            if interp == I_best
            else (f"{interp_label}\n(Lowest Rwp)" if interp == I_dara else interp_label)
        )
        if interp == I_best:
            patch = mpatches.Patch(facecolor=color, edgecolor="black", hatch="xx", label=label_txt, linewidth=0)
        else:
            patch = mpatches.Patch(facecolor=color, edgecolor="white", label=label_txt, linewidth=1.5)
        legend_handles.append(patch)

    ax.legend(handles=legend_handles, loc="lower right", fontsize=10, frameon=True, edgecolor="black")
    
    output_path = os.path.join(output_dir, "phases_probabilities.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

def plot_metrics_contribution(
    interpretations, project_number, target
):
    """Draw horizontal bars (posterior %) with 4 colored segments (Rwp, score, balance, LLM). Same colors as v3."""
    # Weights (same as in v3)
    w_b, w_llm = 0.7, 0.5
    w_score, w_rwp = 0.5, 1.0

    components = ["Normalized Rwp", "Normalized score", "Balance score", "LLM"]
    colors = ["#a6d854", "lightblue", "mediumseagreen", "teal"]

    bar_height = 0.4
    spacing = 0.25

    # Sort interpretations by posterior (ascending for plotting)
    sorted_items = sorted(
        interpretations.items(),
        key=lambda x: x[1]["posterior_probability"],
        reverse=True
    )[::-1]

    y_positions = [i * (bar_height + spacing) for i in range(len(sorted_items))]

    fig, ax = plt.subplots(figsize=(12, 6))

    max_posterior = max(d["posterior_probability"] for _, d in sorted_items) * 100

    # Identify special interpretations for coloring labels
    lowest_rwp_key = max(sorted_items, key=lambda x: x[1]["normalized_rwp"])[0]
    highest_post_key = max(sorted_items, key=lambda x: x[1]["posterior_probability"])[0]

    for idx, (interp_key, data) in enumerate(sorted_items):
        label = f"Interpretation_{interp_key.split('_')[1]}"

        bal = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]
        score = data["normalized_score"]
        rwp = data["normalized_rwp"]
        posterior = data["posterior_probability"]

        # Posterior value (as %)
        posterior_val = posterior * 100.0

        # ---- Compute component contributions (same as v3) ----
        prior = (bal * w_b + llm * w_llm) / (w_b + w_llm)
        fit_quality = (score * w_score + rwp * w_rwp) / (w_score + w_rwp)

        C_balance = (bal * w_b / (w_b + w_llm)) * prior
        C_llm = (llm * w_llm / (w_b + w_llm)) * prior
        C_score = (score * w_score / (w_score + w_rwp)) * fit_quality
        C_rwp = (rwp * w_rwp / (w_score + w_rwp)) * fit_quality

        raw_contributions = [C_rwp, C_score, C_balance, C_llm]
        total_raw = sum(raw_contributions)

        # Scale so that the 4 segments sum to posterior_val (% points)
        if total_raw > 0:
            scale = posterior_val / total_raw
            seg_lengths = [c * scale for c in raw_contributions]  # percentage points
        else:
            seg_lengths = [0.0] * 4

        # ---- Draw stacked bar ----
        left = 0.0
        for seg_val, color in zip(seg_lengths, colors):
            ax.barh(
                [y_positions[idx]],
                [seg_val],
                left=left,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=1.2,
            )
            left += seg_val

        # Interpretation label on the left
        if interp_key == highest_post_key:
            label_color = "darkcyan"
        elif interp_key == lowest_rwp_key:
            label_color = "cyan"
        else:
            label_color = "black"

        ax.text(
            -1.5,
            y_positions[idx],
            label,
            va="center",
            ha="right",
            fontsize=16,
            fontweight="bold",
            color=label_color,
        )

        # Posterior text above the bar
        ax.text(
            posterior_val,
            y_positions[idx] + bar_height / 2 + 0.02,
            f"{posterior_val:.1f}%",
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

        # ---- Text summary of contributions to the right of the bar ----
        # seg_lengths are already in percentage points of posterior
        rwp_pp, score_pp, bal_pp, llm_pp = seg_lengths

        summary_text = (
            f"Rwp {rwp_pp:.1f}%, "
            f"Score {score_pp:.1f}%, "
            f"Balance {bal_pp:.1f}%, "
            f"LLM {llm_pp:.1f}%"
        )
    
    # Formatting
    ax.set_yticks([])
    ax.set_xlim(0, max_posterior)  # extra room on the right for text
    ax.set_title(
        "Posterior Probability per Interpretation\nwith Metric Contributions",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Posterior (%)", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        components,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        fontsize=14,
        frameon=False,
    )
    plt.tight_layout()

    # Save figure
    output_dir = get_output_dir(target, project_number)
    save_path = os.path.join(
        output_dir, "metrics_contribution_breakdown.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()