import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import ast 
from pymatgen.core import Composition
import re
import shutil
from pathlib import Path
import plotly.graph_objects as go
from dara.result import RefinementResult
import matplotlib.patches as mpatches
from scipy.ndimage import minimum_filter1d

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

def plot_interpretation_probabilities_with_statistical(interpretations, project_number, target):
    """
    Plot prior, statistical (fit), and posterior probabilities per interpretation.
    """
    if isinstance(interpretations, list) and len(interpretations) == 1:
        interpretations = interpretations[0]  # Extract the dictionary

    if not isinstance(interpretations, dict):
        raise ValueError("Interpretations should be a dictionary or a list containing a dictionary.")

    # Extract values
    interpretation_names = list(interpretations.keys())
    posterior_values = [interpretations[interp]["posterior_probability"] * 100 for interp in interpretation_names]

    sorted_indices = np.argsort(posterior_values)[::-1]  # Sort from highest to lowest
    sorted_names = [interpretation_names[i] for i in sorted_indices]
    sorted_posterior_values = [posterior_values[i] for i in sorted_indices]

    sorted_prior_values = [interpretations[interp]["prior_probability"] * 100 for interp in sorted_names]
    sorted_statistical_values = [interpretations[interp]["fit_quality"] * 100 for interp in sorted_names] #TODO FROM X TO interpretation_importance

    # Convert names to "Interpretation_X" format
    sorted_labels = [f"Interpretation_{name.split('_')[1]}" for name in sorted_names]

    x = np.arange(len(sorted_labels))  # Label locations
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars for prior probabilities
    rects1 = ax.bar(x - width, sorted_prior_values, width, label="P(I_n) - Prior (Balance Score)")

    # Bars for statistical factors
    rects2 = ax.bar(x, sorted_statistical_values, width, label="P(S | I_n) - Statistical Factor (Interpretation Importance)")

    # Bars for posterior probabilities
    rects3 = ax.bar(x + width, sorted_posterior_values, width, label="P(I_n | S) - Posterior")

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel("Interpretations", fontsize=14)
    ax.set_ylabel("Probability (%)", fontsize=14)
    ax.set_title(f"Probabilities for Interpretations (Project {project_number})", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, fontsize=12, rotation=45, ha="right")
    ax.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0, 0))

    # Annotate bars with their values
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"probability_of_interpretation.png")
    plt.savefig(output_path, format='png', dpi=300)
    plt.close(fig)

    print(f"Probability Plot saved to: {output_path}")
   

    interpretation_names = list(interpretations.keys())
    # Extract the required probabilities and scale them to percentages
    llm_likelihood_values = [interpretations[interp]["LLM_interpretation_likelihood"] * 100 for interp in interpretation_names]
    balance_score_values = [interpretations[interp]["balance_score"] * 100 for interp in interpretation_names]
    prior_values = [interpretations[interp]["prior_probability"] * 100 for interp in interpretation_names]

    # # Convert interpretation names to Interpretation_1, Interpretation_2, etc.
    interpretation_labels = [f"Interpretation_{key.split('_')[1]}" for key in interpretation_names]

    # Define color scheme
    color_llm = "paleturquoise"  # Blue for LLM Likelihood
    color_balance = "darkcyan"  # Orange for Balance Score
    color_prior = "skyblue"  # Green for Prior Probability

    x = np.arange(len(interpretation_names))  # Label locations
    width = 0.3  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    interpretation_labels = interpretation_labels[::-1]
    llm_likelihood_values = llm_likelihood_values[::-1]
    balance_score_values = balance_score_values[::-1]
    prior_values = prior_values[::-1]

    # Plot bars for each probability type
    bars1 = ax.barh(x - width, llm_likelihood_values, width, label="LLM Likelihood", color=color_llm)
    bars2 = ax.barh(x, balance_score_values, width, label="Balance Score", color=color_balance)
    bars3 = ax.barh(x + width, prior_values, width, label="Prior Probability", color=color_prior)

    # Set labels and styling
    ax.set_xlabel("Probability (%)", fontsize=14)
    ax.set_yticks(x)
    ax.set_yticklabels(interpretation_labels, fontsize=12)
    ax.set_title(f"Prior Probabilities for Interpretations (Project {project_number})", fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars with values
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.1f}%",
                        xy=(width + 1, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10)

    # Add legend
    ax.legend(fontsize=12, loc="lower left")

    plt.tight_layout()
    
    # Save the plot
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"prior_probabilities.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close(fig)
    
    print(f"✅ Prior Probabilities Plot saved to: {output_path}")
    posterior_values = [interpretations[interp]["posterior_probability"] * 100 for interp in interpretation_names]

    # Extract simplified phase names
    simplified_phases = [
        "\n".join([phase.split("_(")[0] for phase in interpretations[interp]["phases"]])
        for interp in interpretation_names
    ]
    sorted_indices = np.argsort(posterior_values)[::-1]  # Sort in descending order

    # Apply sorting to all relevant lists
    sorted_interpretation_names = [interpretation_names[i] for i in sorted_indices]
    sorted_posterior_values = [posterior_values[i] for i in sorted_indices]
    sorted_simplified_phases = [simplified_phases[i] for i in sorted_indices]


    # Define color scheme
    highlight_interpretation = "cyan"  # First interpretation highlighted
    neutral_color = "#4C566A"          # Neutral color for others

    # Adjust figure size dynamically based on number of interpretations
    fig_height = max(4, len(interpretation_names) * 0.4)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    # Create horizontal bar chart
    interpretation_names = [f"Interpretation_{key.split('_')[1]}" for key in interpretations.keys()]
    sorted_interpretation_labels = [f"Interpretation_{key.split('_')[1]}" for key in sorted_interpretation_names]
    colors = [highlight_interpretation if name =="Interpretation_1" else neutral_color for name in sorted_interpretation_labels]
   
    # Create horizontal bar chart
    bars = ax.barh(sorted_interpretation_labels[::-1], sorted_posterior_values[::-1], color=colors[::-1])
    
    # Set labels and styling
    ax.set_xlabel("Interpretation Probability (%)", fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(posterior_values) + 5)  # Ensures spacing on the right
    ax.set_yticks(range(len(sorted_interpretation_labels)))  # Ensure ticks are set
    ax.set_yticklabels(sorted_interpretation_labels[::-1], fontsize=18)

    # Adjust font size dynamically
    fontsize = max(8, 14 - len(interpretation_names) * 0.2)

    # Annotate bars with probability values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.1f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
        )

    plt.tight_layout()
    output_dir = get_output_dir(target, project_number)
    output_path_2 = os.path.join(output_dir,"posterior_probabilities.png")
    plt.savefig(output_path_2, format="png", dpi=300)
    plt.close(fig)
    print(f"Posterior Probabilities Plot saved to: {output_path_2}")


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

     # Step 1: Compute unnormalized posterior = prior * fit_quality
    joint_probabilities = {}
    for name, interp in interpretations.items():
        joint = interp["prior_probability"] * interp["fit_quality"]
        interpretations[name]["unnormalized_posterior"] = joint
        joint_probabilities[name] = joint

    # Step 2: Normalize to get posterior_probability
    total_joint = sum(joint_probabilities.values())
    for name in interpretations:
        if total_joint > 0:
            interpretations[name]["posterior_probability"] = (
                interpretations[name]["unnormalized_posterior"] / total_joint
            )
        else:
            interpretations[name]["posterior_probability"] = 0.0

    # Optional: Downweight factor for untrustworthy results
    TRUST_PENALTY_FACTOR = 0.1  # You can tune this

    for interpretation_name, interpretation in interpretations.items():
        is_trustworthy = interpretation.get("trustworthy", True)
        penalty = 1.0 if is_trustworthy else TRUST_PENALTY_FACTOR
        interpretation["adjusted_posterior_probability"] = (
            interpretation["posterior_probability"] * penalty
        )
    return interpretations  # Return updated interpretations dictionary

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
   
def plot_phase_probabilities(interpretations, phase_probabilities, df, project_number, target):
    """
    Plot phase probabilities as a stacked bar chart, where phases belonging to multiple categories
    are represented with stacked colors.

    Parameters:
        interpretations (dict): Dictionary containing interpretations and their phase data.
        df (DataFrame): DataFrame containing synthesis precursor information.
        project_number (str): Identifier for saving the plot.

    Returns:
        None
    """
    # Function to clean phase names
    def clean_phase_name(phase):
        return re.sub(r'_\(icsd_\d+\)-[\w]+', '', phase)  


    # Create defaultdict to sum probabilities
    cleaned_phase_probabilities = defaultdict(float)
    for phase, prob in phase_probabilities.items():
        cleaned_name = clean_phase_name(phase)
        cleaned_phase_probabilities[cleaned_name] += prob  
    phase_probabilities = cleaned_phase_probabilities

    if not phase_probabilities:
        print("No valid phases to plot.")
        return

    # Extract precursors from df and clean formatting
    precursors_raw = df['Precursors'].values[0]  
    # precursors_raw = df['Precursors.1'].values[0]  
    if isinstance(precursors_raw, str) and precursors_raw.startswith("["):
        precursors_list = ast.literal_eval(precursors_raw)  
    else:
        precursors_list = precursors_raw  

    def clean_chemical_formula(name):
        return re.sub(r"\s*\(.*?\)", "", name).strip()  

    precursors = set(clean_chemical_formula(p) for p in precursors_list)

    # Extract phases from Interpretation 1
    first_interpretation_phases = set()
    if "I_1" in interpretations and isinstance(interpretations["I_1"], dict):
        first_interpretation_phases = set(
            [phase.split("_(icsd")[0] for phase in interpretations["I_1"].get("phases", [])]
        )

    sorted_phases = sorted(phase_probabilities.items(), key=lambda x: x[1], reverse=True)
    phases, probabilities = zip(*sorted_phases) if sorted_phases else ([], [])

    cleaned_phases = cleanup_phases_only_formula(phases)
    
    # Define color scheme
    highlight_interpretation = "cyan"  
    highlight_unreacted = "yellow"     
    neutral_color = "#4C566A"          
    unreacted= False
    # Identify which phases belong to which categories
    phase_categories = {
        phase: {
            "interpretation_1": phase in first_interpretation_phases,
            "unreacted_precursor": cleaned_phase in precursors
        }
        for phase, cleaned_phase in zip(phases, cleaned_phases)
    }

    fig_height = max(4, len(phases) * 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Create stacked horizontal bars
    for i, (phase, prob) in enumerate(zip(phases[::-1], probabilities[::-1])):
        categories = phase_categories[phase]
        bar_bottom = 0  # Start stacking from zero

        if categories["interpretation_1"] and categories["unreacted_precursor"]:
            # Split 50/50 between cyan and yellow
            ax.barh(i, prob * 0.5, left=bar_bottom, color=highlight_unreacted)
            bar_bottom += prob * 0.5
            ax.barh(i, prob * 0.5, left=bar_bottom, color=highlight_interpretation)

        elif categories["interpretation_1"]:
            # Full cyan bar
            ax.barh(i, prob, left=bar_bottom, color=highlight_interpretation)

        elif categories["unreacted_precursor"]:
            unreacted= True
            # Split 50/50 between yellow (precursor) and neutral (other phase)
            ax.barh(i, prob * 0.5, left=bar_bottom, color=highlight_unreacted)
            bar_bottom += prob * 0.5
            ax.barh(i, prob * 0.5, left=bar_bottom, color=neutral_color)

        else:
            # Full neutral color
            ax.barh(i, prob, left=bar_bottom, color=neutral_color)

        # Add probability label
        ax.text(prob + 1, i, f"{prob:.1f}%", va="center", fontsize=12)
   

    # Set labels and remove unnecessary spines
    ax.set_yticks(range(len(phases[::-1])))
    ax.set_yticklabels(cleanup_phases(phases[::-1]), fontsize=18)
    ax.set_xlabel("Phase Probability (%)", fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(probabilities) + 30)
    
    # Add legend explaining the colors
    if unreacted:
        legend_patches = [
        plt.Line2D([0], [0], color=highlight_interpretation, lw=6, label="Phases in Interpretation_1"),
        plt.Line2D([0], [0], color=highlight_unreacted, lw=6, label="Unreacted precursor"),
        plt.Line2D([0], [0], color=neutral_color, lw=6, label="Phases in other interpretations")
    ]
    else:
        legend_patches = [
        plt.Line2D([0], [0], color=highlight_interpretation, lw=6, label="Phases in Interpretation_1"),
        plt.Line2D([0], [0], color=neutral_color, lw=6, label="Phases in other interpretations")
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=False)

    plt.tight_layout()

    # Save the plot
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"phase_probabilities.png")
    # output_path = f"../data/xrd_data/xrd_analysis/{target}/{project_number}/phase_probabilities.png"
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

    print(f"✅ Phase probabilities plot saved to: {output_path}")


def plot_phase_and_interpretation_probabilities_adjusted(interpretations, project_number, df, target):
    """
    Same as original plot function, but uses adjusted_posterior_probability and visually flags untrustworthy interpretations.
    """
   

    interpretation_colors = {
        "I_1": "cyan",
        "I_2": "#955196",
        "I_3": "#dd5182",
        "I_4": "#ff6e54",
        "I_5": "#ffa600",
    }
    default_interpretation_color = "#4C566A"
    phase_color_palette = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a",
                           "#ffee65", "#beb9db", "#fdcce5", "#7596ee", "#8bf953",
                           "#d779bc", "#ffe240", "#dcff4b", "#ffce56", "#a1c9f4"]

    def clean_phase_name(phase):
        return phase.split("_(icsd")[0]

    def clean_chemical_formula(name):
        return re.sub(r"\s*\(.*?\)", "", name).strip()

    precursors_raw = df['Precursors'].values[0]
    # precursors_raw = df['Precursors.1'].values[0]
    if isinstance(precursors_raw, str) and precursors_raw.startswith("["):
        precursors_list = ast.literal_eval(precursors_raw)
    else:
        precursors_list = precursors_raw
    precursors = set(clean_chemical_formula(p) for p in precursors_list)

    for interp in interpretations:
        interpretations[interp]["phases"] = cleanup_phases([clean_phase_name(p) for p in interpretations[interp]["phases"]])

    unique_phases = sorted(set(clean_phase_name(phase) for interp in interpretations.values() for phase in interp["phases"]))
    unique_phases = cleanup_phases(unique_phases)
    phase_colors = dict(zip(unique_phases, phase_color_palette[:len(unique_phases)]))
    phase_probabilities = calculate_phase_probabilities(interpretations)

    interpretation_names = list(interpretations.keys())
    adjusted_posteriors = [interpretations[interp]["adjusted_posterior_probability"] * 100 for interp in interpretation_names]
    sorted_indices = np.argsort(adjusted_posteriors)[::-1]
    sorted_interpretations = [interpretation_names[i] for i in sorted_indices]
    sorted_post_values = [adjusted_posteriors[i] for i in sorted_indices]

    sorted_weight_fractions = [interpretations[interp]["weight_fraction"] for interp in sorted_interpretations]
    sorted_phases = [interpretations[interp]["phases"] for interp in sorted_interpretations]
    trust_flags = {interp: interpretations[interp].get("trustworthy", True) for interp in sorted_interpretations}

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_interpretations) * 0.4)))
    bottom_bar = np.zeros(len(sorted_interpretations))

    for phase in unique_phases:
        fraction_values = [
            (weights[phases.index(phase)] if phase in phases else 0) * post_prob / 100
            for phases, weights, post_prob in zip(sorted_phases, sorted_weight_fractions, sorted_post_values)
        ]
        bars = ax.barh(
            [f"Interpretation_{name.split('_')[1]}" for name in sorted_interpretations[::-1]],
            fraction_values[::-1],
            left=bottom_bar[::-1],
            color=phase_colors[phase],
            edgecolor=["red" if not trust_flags[name] else "black" for name in sorted_interpretations[::-1]],
            # alpha=[0.4 if not trust_flags[name] else 1.0 for name in sorted_interpretations[::-1]],
            hatch=['//' if not trust_flags[name] else '' for name in sorted_interpretations[::-1]],
            linewidth=1.0,
            label=phase
        )
        bottom_bar += fraction_values

    ax.set_xlabel("Adjusted Interpretation Probability (%)", fontsize=14)
    ax.set_xlim(0, max(sorted_post_values) + 20)
    ax.set_yticks(range(len(sorted_interpretations)))
    ax.set_yticklabels([f"Interpretation_{name.split('_')[1]}" for name in sorted_interpretations[::-1]], fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for ticklabel, interp in zip(ax.get_yticklabels(), sorted_interpretations[::-1]):
        ticklabel.set_color(interpretation_colors.get(interp, default_interpretation_color))
        if not trust_flags[interp]:
            ticklabel.set_alpha(0.5)

    plt.tight_layout()
    # plt.savefig(f"../data/xrd_data/xrd_analysis/{target}/{project_number}/adjusted_posterior_probabilities.png", dpi=300)
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"adjusted_posterior_probabilities.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

def plot_phase_and_interpretation_unnormalized_probabilities_newstyle(interpretations, project_number, df, target):
    """
    Plot phase probabilities and interpretation probabilities using distinct colors
    for interpretations and phases.
    """
    # Define colors for interpretations
   
    interpretation_colors = {
        "I_1": "#5778a4",
        "I_2": "#955196",
        "I_3": "#dd5182",
        "I_4": "#ff6e54",
        "I_5": "#ffa600",
        }

    phase_color_palette = [
    "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", 
    "#7596ee", "#8bf953", "#d779bc", "#ffe240", "#dcff4b", "#ffce56", 
    "#a1c9f4", "#ff9f9b", "#8c7aa9", "#f4a582", "#6bb0bc", "#f4e3b2", "#a4d7d1", "#c4b7b7", "#e2a29b", "#a6cfcf"
]
    default_interpretation_color = "#4C566A"  # For additional interpretations
    
    # Function to clean phase names
    def clean_phase_name(phase):
        return phase.split("_(icsd")[0]

    
    # Extract precursors from df and clean formatting
    precursors_raw = df['Precursors'].values[0]  
    # precursors_raw = df['Precursors.1'].values[0]  
    if isinstance(precursors_raw, str) and precursors_raw.startswith("["):
        precursors_list = ast.literal_eval(precursors_raw)  
    else:
        precursors_list = precursors_raw  

    def clean_chemical_formula(name):
        return re.sub(r"\s*\(.*?\)", "", name).strip()  

    precursors = set(clean_chemical_formula(p) for p in precursors_list)
   
    for interp in interpretations:
        interpretations[interp]["phases"] = cleanup_phases([clean_phase_name(p) for p in interpretations[interp]["phases"]])
    # Extract all unique phases and assign colors
    unique_phases = sorted(set(clean_phase_name(phase) for interp in interpretations.values() for phase in interp["phases"]))
    print("The unique phases :", unique_phases)
    unique_phases = cleanup_phases(unique_phases)
   
    phase_colors = dict(zip(unique_phases, phase_color_palette[:len(unique_phases)]))
    print("The phase colors are: ", phase_colors)
    # Compute phase probabilities
    phase_probabilities = calculate_phase_probabilities(interpretations)
    
    # Compute interpretation probabilities
    interpretation_names = list(interpretations.keys())
    posterior_values = [interpretations[interp]["unnormalized_posterior"] * 100 for interp in interpretation_names]
    rwp_values = [interpretations[interp]["rwp"] for interp in interpretation_names]
    sorted_indices = np.argsort(posterior_values)[::-1]
    sorted_interpretations = [interpretation_names[i] for i in sorted_indices]
    sorted_posterior_values = [posterior_values[i] for i in sorted_indices]
    sorted_rwp_values = [rwp_values[i] for i in sorted_indices]
    # Find the interpretation with the highest probability
    I_best = sorted_interpretations[np.argmax(sorted_posterior_values)]
    I_dara = sorted_interpretations[np.argmin(sorted_rwp_values)]
    print("The selected I_dara is:", I_dara)

    # Dynamically assign cyan to I_dara
    interpretation_colors[I_dara] = "cyan"
    # Assign colors to y-axis labels based on interpretation names
    sorted_label_colors = [interpretation_colors.get(interp, default_interpretation_color) for interp in sorted_interpretations]
    
    # Prepare data for interpretation probability plot
    sorted_weight_fractions = [interpretations[interp]["weight_fraction"] for interp in sorted_interpretations]

    sorted_phases = [cleanup_phases([clean_phase_name(p) for p in interpretations[interp]["phases"]]) for interp in sorted_interpretations]

    
    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_interpretations) * 0.4)))
    
    # Create stacked bar chart for interpretations
    bottom_bar = np.zeros(len(sorted_interpretations))
    
 
    # Precompute trustworthiness flags
    trust_flags = {
        interp: interpretations[interp].get("trustworthy", True)
        for interp in sorted_interpretations
    }

    for phase_idx, phase in enumerate(unique_phases):
        fraction_values = [
            (weights[phases.index(phase)] if phase in phases else 0) * post_prob / 100
            for phases, weights, post_prob in zip(sorted_phases, sorted_weight_fractions, sorted_posterior_values)
        ]

        # Style elements based on trustworthiness
        # alpha_vals = [0.4 if not trust_flags[name] else 1.0 for name in sorted_interpretations[::-1]]
        # edge_colors = ["bold red" if not trust_flags[name] else "bold black" for name in sorted_interpretations[::-1]]
        edge_colors = ["red" if not trust_flags[name] else "none" for name in sorted_interpretations[::-1]]
        # hatch_patterns = ["//" if not trust_flags[name] else "" for name in sorted_interpretations[::-1]]

        bars = ax.barh(
            [f"Interpretation_{name.split('_')[1]}" for name in sorted_interpretations[::-1]],
            fraction_values[::-1],
            left=bottom_bar[::-1],
            color=phase_colors[phase],
            # edgecolor=edge_colors,
            label=phase if phase_idx < 5 else ""
        )
        bottom_bar += fraction_values
    ax.set_xlabel("Interpretation Probability (%)", fontsize=14)
    ax.set_xlim(0, max(sorted_posterior_values) + 20)
    ax.set_yticks(range(len(sorted_interpretations)))

    def strikethrough(text):
        return ''.join(char + '\u0336' for char in text)

    ytick_labels = [
    strikethrough(f"Interpretation_{name.split('_')[1]}")
    if not trust_flags[name]
    else f"Interpretation_{name.split('_')[1]}"
    for name in sorted_interpretations[::-1]
    ]

    ax.set_yticks(range(len(sorted_interpretations)))
    ax.set_yticklabels(ytick_labels, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for ticklabel, color in zip(ax.get_yticklabels(), sorted_label_colors[::-1]):
        ticklabel.set_color(color)

    # Dictionary to store total width per interpretation (row)
    total_bar_widths = defaultdict(float)

    # First loop: Compute total width per interpretation
    for bar, interp_name in zip(ax.patches, sorted_interpretations * len(unique_phases)):
        total_bar_widths[interp_name] += bar.get_width()  # Sum segment widths

    # Second loop: Place text at the end of the full stacked bar
    for bar, interp_name, value in zip(ax.patches, sorted_interpretations * len(unique_phases), sorted_posterior_values[::-1] * len(unique_phases)):
        if total_bar_widths[interp_name] == bar.get_x() + bar.get_width():  # Ensure we place text only once per row
            ax.text(total_bar_widths[interp_name] + 1,  # Move text slightly beyond total width
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    va="center", ha="left", fontsize=12)
    
    # Function to check if a phase is in precursors (ignoring _<numbers>)
    def is_precursor(phase_name):
        base_phase = phase_name.split("_")[0]  # Extract chemical formula before '_'
        return any(base_phase == precursor.split("_")[0] for precursor in precursors)

    # Create legend handles
    legend_handles = [
        plt.Line2D([0], [0], color=phase_colors.get(phase, "black"), lw=6,
                label=f"*{phase}" if is_precursor(phase) else phase)  
        for phase in unique_phases
    ]
    
    yticklabels = ax.get_yticklabels()
    for label, interp_name in zip(yticklabels, sorted_interpretations[::-1]):  # Match labels to interpretations
        label.set_fontweight("bold")  # Make bold
        if interp_name == I_dara: 
            label.set_color("cyan") 
            # label.set_color(interpretation_colors.get(interp_name, "black"))  # Keep assigned color for I_1
        elif interp_name == I_best:  
            label.set_color("darkcyan")  # Highlight best interpretation
        else:
            label.set_color("black")  # Default color for others
    

    # Pass the updated handles to ax.legend()
    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=True, edgecolor="black", facecolor="white", framealpha=1, fancybox=True)
    plt.tight_layout()
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"unnormalized_posterior_probabilities_polychromatic.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()
    
    # Create phase probabilities plot
    sorted_phases_by_prob = sorted(phase_probabilities.items(), key=lambda x: x[1], reverse=True)
    cleaned_phases, probabilities = zip(*[(clean_phase_name(p), v) for p, v in sorted_phases_by_prob]) if sorted_phases_by_prob else ([], [])

    fig, ax = plt.subplots(figsize=(8, max(4, len(cleaned_phases) * 0.4)))

    # Dictionary to track total bar width per phase
    total_bar_widths = defaultdict(float)
    
    # First loop: Compute total width per phase
    bottom_bar = np.zeros(len(cleaned_phases))
    for interp_idx, interp in enumerate(sorted_interpretations):
        fraction_values = [
            (interpretations[interp]["posterior_probability"] * 100 if phase in [clean_phase_name(p) for p in interpretations[interp]["phases"]] else 0)
            for phase in cleaned_phases
        ]
        highlight_color = "darkcyan" if interp == I_best else interpretation_colors.get(interp, default_interpretation_color)
        hatch_pattern = "xx" if interp == I_best else ""  # Add cross-hatched pattern for I_best
        
        bars = ax.barh(cleaned_phases[::-1], fraction_values[::-1], left=bottom_bar[::-1], 
                   color=highlight_color, label=interp if interp_idx < 5 else "", hatch=hatch_pattern)
        
        # Update total width tracker
        for bar, phase in zip(bars, cleaned_phases):
            total_bar_widths[phase] += bar.get_width()

        bottom_bar += fraction_values
    

    # Only this loop should be placing the text
    for phase in cleaned_phases:
        total_width = total_bar_widths[phase]  # Get total width of stacked bar
        phase_idx = cleaned_phases.index(phase)  # Correctly map phase to index
        ax.text(total_width + 1,  # Move text slightly beyond bar
                phase_idx,  # Ensure text is on the correct row
                f"{total_width:.1f}%", 
                va="center", ha="left", fontsize=12)

    
    ax.set_xlabel("Phase Probability (%)", fontsize=14)
    ax.set_xlim(0, max(probabilities) + 30)
    ax.set_yticks(range(len(cleaned_phases)))
    ax.set_yticklabels(
    [
        f"*{phase}" if phase.split("_")[0] in precursors else phase 
        for phase in cleaned_phases[::-1]
    ],
    fontsize=14, 
    fontweight='bold'
)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Apply y-tick colors
    for ticklabel, phase in zip(ax.get_yticklabels(), cleaned_phases[::-1]):
        ticklabel.set_color(phase_colors.get(phase, "black"))

    legend_handles = []

    for interp in interpretation_names:
        color = "darkcyan" if interp == I_best else interpretation_colors.get(interp, default_interpretation_color)
        
        # Use a rectangle (Patch) for ALL interpretations for uniform size
        if interp == I_best:
            patch = mpatches.Patch(facecolor=color, edgecolor="black", hatch="xx", label=interp, linewidth=0)
        else:
            patch = mpatches.Patch(facecolor=color, edgecolor="white", label=interp, linewidth=1.5)

        legend_handles.append(patch)

    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        label.set_fontweight("bold")  # Make bold
        label.set_color("black")  # Force black color

    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=True, edgecolor="black", facecolor="white", framealpha=1, fancybox=True)

    plt.tight_layout()
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"unnormalized_phase_probabilities_polychromatic.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()



def plot_phase_and_interpretation_probabilities_newstyle(interpretations, project_number, df, target):
    """
    Plot phase probabilities and interpretation probabilities using distinct colors
    for interpretations and phases.
    """
    # Define colors for interpretations
   
    interpretation_colors = {
        "I_1": "#5778a4",
        "I_2": "#955196",
        "I_3": "#dd5182",
        "I_4": "#ff6e54",
        "I_5": "#ffa600",
        }

    phase_color_palette = [
    "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", 
    "#7596ee", "#8bf953", "#d779bc", "#ffe240", "#dcff4b", "#ffce56", 
    "#a1c9f4", "#ff9f9b", "#8c7aa9", "#f4a582", "#6bb0bc", "#f4e3b2", "#a4d7d1", "#c4b7b7", "#e2a29b", "#a6cfcf"
]
    default_interpretation_color = "#4C566A"  # For additional interpretations
    
    # Function to clean phase names
    def clean_phase_name(phase):
        return phase.split("_(icsd")[0]

    
    # Extract precursors from df and clean formatting
    precursors_raw = df['Precursors'].values[0]  
    # precursors_raw = df['Precursors.1'].values[0]  
    if isinstance(precursors_raw, str) and precursors_raw.startswith("["):
        precursors_list = ast.literal_eval(precursors_raw)  
    else:
        precursors_list = precursors_raw  

    def clean_chemical_formula(name):
        return re.sub(r"\s*\(.*?\)", "", name).strip()  

    precursors = set(clean_chemical_formula(p) for p in precursors_list)
   
    for interp in interpretations:
        interpretations[interp]["phases"] = cleanup_phases([clean_phase_name(p) for p in interpretations[interp]["phases"]])
    # Extract all unique phases and assign colors
    unique_phases = sorted(set(clean_phase_name(phase) for interp in interpretations.values() for phase in interp["phases"]))
    print("The unique phases :", unique_phases)
    unique_phases = cleanup_phases(unique_phases)
   
    phase_colors = dict(zip(unique_phases, phase_color_palette[:len(unique_phases)]))
    print("The phase colors are: ", phase_colors)
    # Compute phase probabilities
    phase_probabilities = calculate_phase_probabilities(interpretations)
    
    # Compute interpretation probabilities
    interpretation_names = list(interpretations.keys())
    posterior_values = [interpretations[interp]["posterior_probability"] * 100 for interp in interpretation_names]
    rwp_values = [interpretations[interp]["rwp"] for interp in interpretation_names]
    sorted_indices = np.argsort(posterior_values)[::-1]
    sorted_interpretations = [interpretation_names[i] for i in sorted_indices]
    sorted_posterior_values = [posterior_values[i] for i in sorted_indices]
    sorted_rwp_values = [rwp_values[i] for i in sorted_indices]
    print("Here on the plot my rwp values are:", rwp_values)
    print("the sorted rwp values are: ", sorted_rwp_values)
    # Find the interpretation with the highest probability
    I_best = sorted_interpretations[np.argmax(sorted_posterior_values)]
    I_dara = sorted_interpretations[np.argmin(sorted_rwp_values)]
    print("the selected I dara is : ", I_dara)
    # Assign cyan to I_dara dynamically
    interpretation_colors[I_dara] = "cyan"

    # Assign colors to y-axis labels based on interpretation names
    sorted_label_colors = [interpretation_colors.get(interp, default_interpretation_color) for interp in sorted_interpretations]
    
    # Prepare data for interpretation probability plot
    sorted_weight_fractions = [interpretations[interp]["weight_fraction"] for interp in sorted_interpretations]

    sorted_phases = [cleanup_phases([clean_phase_name(p) for p in interpretations[interp]["phases"]]) for interp in sorted_interpretations]

    
    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_interpretations) * 0.4)))
    
    # Create stacked bar chart for interpretations
    bottom_bar = np.zeros(len(sorted_interpretations))
    
 
    # Precompute trustworthiness flags
    trust_flags = {
        interp: interpretations[interp].get("trustworthy", True)
        for interp in sorted_interpretations
    }

    for phase_idx, phase in enumerate(unique_phases):
        fraction_values = [
            (weights[phases.index(phase)] if phase in phases else 0) * post_prob / 100
            for phases, weights, post_prob in zip(sorted_phases, sorted_weight_fractions, sorted_posterior_values)
        ]

        # Style elements based on trustworthiness
        # alpha_vals = [0.4 if not trust_flags[name] else 1.0 for name in sorted_interpretations[::-1]]
        # edge_colors = ["bold red" if not trust_flags[name] else "bold black" for name in sorted_interpretations[::-1]]
        edge_colors = ["red" if not trust_flags[name] else "none" for name in sorted_interpretations[::-1]]
        # hatch_patterns = ["//" if not trust_flags[name] else "" for name in sorted_interpretations[::-1]]

        bars = ax.barh(
            [f"Interpretation_{name.split('_')[1]}" for name in sorted_interpretations[::-1]],
            fraction_values[::-1],
            left=bottom_bar[::-1],
            color=phase_colors[phase],
            # edgecolor=edge_colors,
            label=phase if phase_idx < 5 else ""
        )
        bottom_bar += fraction_values
    ax.set_xlabel("Interpretation Probability (%)", fontsize=14)
    ax.set_xlim(0, max(sorted_posterior_values) + 20)
    ax.set_yticks(range(len(sorted_interpretations)))

    def strikethrough(text):
        return ''.join(char + '\u0336' for char in text)

    ytick_labels = [
    strikethrough(f"Interpretation_{name.split('_')[1]}")
    if not trust_flags[name]
    else f"Interpretation_{name.split('_')[1]}"
    for name in sorted_interpretations[::-1]
    ]

    ax.set_yticks(range(len(sorted_interpretations)))
    ax.set_yticklabels(ytick_labels, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for ticklabel, color in zip(ax.get_yticklabels(), sorted_label_colors[::-1]):
        ticklabel.set_color(color)

    # Dictionary to store total width per interpretation (row)
    total_bar_widths = defaultdict(float)

    # First loop: Compute total width per interpretation
    for bar, interp_name in zip(ax.patches, sorted_interpretations * len(unique_phases)):
        total_bar_widths[interp_name] += bar.get_width()  # Sum segment widths

    # Second loop: Place text at the end of the full stacked bar
    for bar, interp_name, value in zip(ax.patches, sorted_interpretations * len(unique_phases), sorted_posterior_values[::-1] * len(unique_phases)):
        if total_bar_widths[interp_name] == bar.get_x() + bar.get_width():  # Ensure we place text only once per row
            ax.text(total_bar_widths[interp_name] + 1,  # Move text slightly beyond total width
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    va="center", ha="left", fontsize=12)
    
    # Function to check if a phase is in precursors (ignoring _<numbers>)
    def is_precursor(phase_name):
        base_phase = phase_name.split("_")[0]  # Extract chemical formula before '_'
        return any(base_phase == precursor.split("_")[0] for precursor in precursors)

    # Create legend handles
    legend_handles = [
        plt.Line2D([0], [0], color=phase_colors.get(phase, "black"), lw=6,
                label=f"*{phase}" if is_precursor(phase) else phase)  
        for phase in unique_phases
    ]
    
    yticklabels = ax.get_yticklabels()
    for label, interp_name in zip(yticklabels, sorted_interpretations[::-1]):  # Match labels to interpretations
        label.set_fontweight("bold")  # Make bold
        if interp_name == I_dara: 
            label.set_color("cyan") 
        # if interp_name == "I_1":  
        #     label.set_color(interpretation_colors.get(interp_name, "black"))  # Keep assigned color for I_1
        elif interp_name == I_best:  
            label.set_color("darkcyan")  # Highlight best interpretation
        else:
            label.set_color("black")  # Default color for others
    

    # Pass the updated handles to ax.legend()
    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=True, edgecolor="black", facecolor="white", framealpha=1, fancybox=True)
    plt.tight_layout()
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"posterior_probabilities_polychromatic.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()
    
    # Create phase probabilities plot
    sorted_phases_by_prob = sorted(phase_probabilities.items(), key=lambda x: x[1], reverse=True)
    cleaned_phases, probabilities = zip(*[(clean_phase_name(p), v) for p, v in sorted_phases_by_prob]) if sorted_phases_by_prob else ([], [])

    fig, ax = plt.subplots(figsize=(8, max(4, len(cleaned_phases) * 0.4)))

    # Dictionary to track total bar width per phase
    total_bar_widths = defaultdict(float)
    
    # First loop: Compute total width per phase
    bottom_bar = np.zeros(len(cleaned_phases))
    for interp_idx, interp in enumerate(sorted_interpretations):
        fraction_values = [
            (interpretations[interp]["posterior_probability"] * 100 if phase in [clean_phase_name(p) for p in interpretations[interp]["phases"]] else 0)
            for phase in cleaned_phases
        ]
        highlight_color = "darkcyan" if interp == I_best else interpretation_colors.get(interp, default_interpretation_color)
        hatch_pattern = "xx" if interp == I_best else ""  # Add cross-hatched pattern for I_best
        
        bars = ax.barh(cleaned_phases[::-1], fraction_values[::-1], left=bottom_bar[::-1], 
                   color=highlight_color, label=interp if interp_idx < 5 else "", hatch=hatch_pattern)
        
        # Update total width tracker
        for bar, phase in zip(bars, cleaned_phases):
            total_bar_widths[phase] += bar.get_width()

        bottom_bar += fraction_values
    

    # Only this loop should be placing the text
    for phase in cleaned_phases:
        total_width = total_bar_widths[phase]  # Get total width of stacked bar
        phase_idx = cleaned_phases.index(phase)  # Correctly map phase to index
        ax.text(total_width + 1,  # Move text slightly beyond bar
                phase_idx,  # Ensure text is on the correct row
                f"{total_width:.1f}%", 
                va="center", ha="left", fontsize=12)

    
    ax.set_xlabel("Phase Probability (%)", fontsize=14)
    ax.set_xlim(0, max(probabilities) + 30)
    ax.set_yticks(range(len(cleaned_phases)))
    ax.set_yticklabels(
    [
        f"*{phase}" if phase.split("_")[0] in precursors else phase 
        for phase in cleaned_phases[::-1]
    ],
    fontsize=14, 
    fontweight='bold'
)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Apply y-tick colors
    for ticklabel, phase in zip(ax.get_yticklabels(), cleaned_phases[::-1]):
        ticklabel.set_color(phase_colors.get(phase, "black"))

    legend_handles = []

    for interp in interpretation_names:
        color = "darkcyan" if interp == I_best else interpretation_colors.get(interp, default_interpretation_color)
        
        # Use a rectangle (Patch) for ALL interpretations for uniform size
        if interp == I_best:
            patch = mpatches.Patch(facecolor=color, edgecolor="black", hatch="xx", label=interp, linewidth=0)
        else:
            patch = mpatches.Patch(facecolor=color, edgecolor="white", label=interp, linewidth=1.5)

        legend_handles.append(patch)

    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        label.set_fontweight("bold")  # Make bold
        label.set_color("black")  # Force black color

    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=True, edgecolor="black", facecolor="white", framealpha=1, fancybox=True)

    plt.tight_layout()
    output_dir = get_output_dir(target, project_number)
    output_path = os.path.join(output_dir,"phase_probabilities_polychromatic.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()



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

def add_flag(background, observed):
    """
    Computes the excess background flag:
    sum of max(0, background - observed) over the spectrum.

    Parameters:
        background (array-like): Background intensities.
        observed (array-like): Observed intensities.

    Returns:
        float: Flag value (total excess background), or None if inputs are invalid.
    """
    background = np.asarray(background)
    observed = np.asarray(observed)

    # if background.shape != observed.shape:
    #     return None
    excess_bkg = np.sum(np.maximum(0, background - observed))
    total_signal = np.sum(observed)

    normalized_excess = excess_bkg*100 / total_signal

    return excess_bkg, normalized_excess

def flag_interpretation_trustworthiness(interpretations: dict) -> dict:
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

def compute_trust_score(interpretations: dict) -> dict:
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


def net_signal_score(plot_data, angle_window=(10, 70)):
    observed = np.asarray(plot_data.y_obs)
    background = np.asarray(plot_data.y_bkg)
    angles = np.asarray(plot_data.x)

    region_mask = (angles >= angle_window[0]) & (angles <= angle_window[1])
    obs, bkg = observed[region_mask], background[region_mask]

    signal_above = np.sum(np.maximum(0, obs - bkg))
    signal_below = np.sum(np.maximum(0, bkg - obs))
    
    signal_score = signal_above / (signal_above + signal_below + 1e-8)
    return signal_score * 100

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

def abs_diff_score(plot_data, angle_window=(10, 70)):
    observed = np.asarray(plot_data.y_obs)
    background = np.asarray(plot_data.y_bkg)
    angles = np.asarray(plot_data.x)

    region_mask = (angles >= angle_window[0]) & (angles <= angle_window[1])
    obs, bkg, theta = observed[region_mask], background[region_mask], angles[region_mask]

    delta_theta = theta[-1] - theta[0]
    score = np.sum(np.abs(obs - bkg)) / (delta_theta + 1e-8)
    return score


def estimate_baseline(y, window_size=50):
    """A non-parametric baseline: the minimum in each sliding window."""
    return minimum_filter1d(y, size=window_size, mode='reflect')
def bkg_baseline_distance_score(plot_data,
                                angle_window=(10,70),
                                window_size=50):
    obs   = np.asarray(plot_data.y_obs)
    bkg   = np.asarray(plot_data.y_bkg)
    θ     = np.asarray(plot_data.x)
    mask  = (θ>=angle_window[0]) & (θ<=angle_window[1])
    obs, bkg = obs[mask], bkg[mask]

    # 1) compute data-driven baseline
    baseline = estimate_baseline(obs, window_size=window_size)

    # 2) mean absolute deviation between fitted background and baseline
    return np.mean(np.abs(bkg - baseline))


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


def plot_contribution_decomposition(interpretations, project_number, target):
    """
    Plot contribution decomposition for a given sample and save the figure.

    Parameters:
        interpretations (dict): Dictionary with I_1, I_2... as keys and their metrics.
        sample_id (str): Sample name (e.g., "TRI_106") for the title and filename.
        project_number (str): Project ID for folder structure.
        target (str): Project target name.
    """
    # Weights
    w_b, w_llm = 0.7, 0.5
    w_score, w_rwp = 0.5, 1.0

    components = ["Normalized RWP", "Normalized Score", "Balance", "LLM"]
    colors = ["#a6d854", "lightblue", "mediumseagreen", "teal"]
    bar_height = 0.6
    spacing = 0.4

    sorted_items = sorted(interpretations.items(), key=lambda x: x[1]["unnormalized_posterior"], reverse=True)[::-1]
    y_positions = [i * (bar_height + spacing) for i in range(len(sorted_items))]

    fig, ax = plt.subplots(figsize=(12, 7))

    max_posterior = max(d["unnormalized_posterior"] for _, d in sorted_items) * 100

    for idx, (interp_key, data) in enumerate(sorted_items):
        label = f"Interpretation_{interp_key.split('_')[1]}"
        bal = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]
        score = data["normalized_score"]
        rwp = data["normalized_rwp"]
        posterior = data["unnormalized_posterior"]

        # Compute contributions
        prior = (bal * w_b + llm * w_llm) / (w_b + w_llm)
        fit_quality = (score * w_score + rwp * w_rwp) / (w_score + w_rwp)
        C_balance = (bal * w_b / (w_b + w_llm)) * prior
        C_llm = (llm * w_llm / (w_b + w_llm)) * prior
        C_score = (score * w_score / (w_score + w_rwp)) * fit_quality
        C_rwp = (rwp * w_rwp / (w_score + w_rwp)) * fit_quality

        # Reorder for plotting
        total_contribution = C_llm + C_balance + C_score + C_rwp
        contributions = [C_rwp, C_score, C_balance, C_llm]
        percentages = [(C / total_contribution) * 100 for C in contributions]
        values = [(p / 100) * posterior * 100 for p in percentages]  # scale

        # Draw stacked bar
        left = 0
        for val, color, pct in zip(values, colors, percentages):
            ax.barh([y_positions[idx]], [val], left=left, height=bar_height,
                    color=color, edgecolor='white')
            label_text = f"{pct:.1f}%"
            if val > 5:
                ax.text(left + val / 2, y_positions[idx], label_text, ha='center', va='center',
                        fontsize=10, color='black', fontweight='bold')
            elif pct > 1:
                ax.text(left + 0.3, y_positions[idx], label_text, ha='left', va='center',
                        fontsize=9, color='black', fontweight='bold')

    
        
        # Show posterior %
        ax.text(max_posterior * 1.05, y_positions[idx], f"{posterior * 100:.1f}%",
                ha='left', va='center', fontsize=11, fontweight='bold', color='black')
        # Label interpretation
        ax.text(-1.5, y_positions[idx], label, va='center', ha='right', fontsize=14)

    # Final formatting
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Posterior (%)", fontsize=14)
    ax.set_title(f"Interpretation Contribution Breakdown", fontsize=16, fontweight='bold')
    ax.set_xlim(0, max_posterior * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(components, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=13)
    plt.tight_layout()

    # Save
    output_dir = get_output_dir(target, project_number)
    save_path = os.path.join(output_dir, f"contribution_breakdown.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved to: {save_path}")



def plot_contribution_decomposition_dual(interpretations, project_number, target):
    """
    Create two side-by-side plots:
    1. Posterior probability per interpretation (left).
    2. Normalized contribution breakdown to 100% (right).

    Parameters:
        interpretations (dict): Dictionary with I_1, I_2... as keys and their metrics.
        project_number (str): Project ID for folder structure.
        target (str): Project target name.
    """
    # Weights
    w_b, w_llm = 0.7, 0.5
    w_score, w_rwp = 0.5, 1.0

    components = ["Normalized RWP", "Normalized Score", "Balance", "LLM"]
    colors = ["#a6d854", "lightblue", "mediumseagreen", "teal"]
    bar_height = 0.6
    spacing = 0.4

    sorted_items = sorted(interpretations.items(), key=lambda x: x[1]["unnormalized_posterior"], reverse=True)[::-1]
    y_positions = [i * (bar_height + spacing) for i in range(len(sorted_items))]
    labels = [f"Interpretation_{key.split('_')[1]}" for key, _ in sorted_items]

    fig, (ax_post, ax_breakdown) = plt.subplots(
        ncols=2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]}
    )

    max_posterior = max(d["unnormalized_posterior"] for _, d in sorted_items) * 100

    for idx, (interp_key, data) in enumerate(sorted_items):
        label = f"Interpretation_{interp_key.split('_')[1]}"
        bal = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]
        score = data["normalized_score"]
        rwp = data["normalized_rwp"]
        posterior = data["unnormalized_posterior"]

        # Compute contributions
        prior = (bal * w_b + llm * w_llm) / (w_b + w_llm)
        fit_quality = (score * w_score + rwp * w_rwp) / (w_score + w_rwp)
        C_balance = (bal * w_b / (w_b + w_llm)) * prior
        C_llm = (llm * w_llm / (w_b + w_llm)) * prior
        C_score = (score * w_score / (w_score + w_rwp)) * fit_quality
        C_rwp = (rwp * w_rwp / (w_score + w_rwp)) * fit_quality

        # Plot 1: Posterior
        ax_post.barh([y_positions[idx]], [posterior * 100], height=bar_height,
                     color="lightgray", edgecolor='black')
        ax_post.text(posterior * 100 + 1, y_positions[idx], f"{posterior * 100:.1f}%",
                     ha='left', va='center', fontsize=11, fontweight='bold')
        ax_post.text(-1.5, y_positions[idx], label, va='center', ha='right', fontsize=13)

        # Plot 2: Normalized breakdown (sum = 100%)
        total = C_rwp + C_score + C_balance + C_llm
        normalized_contributions = [100 * x / total for x in [C_rwp, C_score, C_balance, C_llm]]

        left = 0
        for val, color in zip(normalized_contributions, colors):
            ax_breakdown.barh([y_positions[idx]], [val], left=left, height=bar_height,
                              color=color, edgecolor='white')
            ax_breakdown.text(
                left + val / 2, y_positions[idx], f"{val:.1f}%",
                ha='center', va='center', fontsize=10, fontweight='bold', color='black'
            )
            left += val

    # Posterior axis
    ax_post.set_yticks([])
    ax_post.set_xlim(0, max_posterior * 1.25)
    ax_post.set_title("Posterior Probability per Interpretation", fontsize=14, fontweight='bold')
    ax_post.set_xlabel("Posterior (%)", fontsize=12)
    ax_post.spines['top'].set_visible(False)
    ax_post.spines['right'].set_visible(False)

    # Breakdown axis
    ax_breakdown.set_yticks([])
    ax_breakdown.set_xlim(0, 100)
    ax_breakdown.set_title("Normalized Contribution Breakdown", fontsize=14, fontweight='bold')
    ax_breakdown.set_xlabel("Contribution to Final Probability (%)", fontsize=12)
    ax_breakdown.spines['top'].set_visible(False)
    ax_breakdown.spines['right'].set_visible(False)
    ax_breakdown.legend(components, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

    plt.tight_layout()

    # Save
    output_dir = get_output_dir(target, project_number)  # assumes this is defined elsewhere
    save_path = os.path.join(output_dir, f"contribution_breakdown_dual_normalized.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Normalized dual contribution plot saved to: {save_path}")



def plot_contribution_decomposition_dual_normalized_right_v2(interpretations, project_number, target):
    """
    Create two side-by-side plots:
    1. Posterior probability per interpretation (left).
    2. Normalized contribution breakdown to 100% (right).

    - Bars are slightly smaller and closer.
    - Posterior bars are light blue (no grey, no black border).
    - Numbers flip outside the bar if they do not fit.

    Parameters:
        interpretations (dict): Dictionary with I_1, I_2... as keys and their metrics.
        project_number (str): Project ID for folder structure.
        target (str): Project target name.
    """
    # Weights
    w_b, w_llm = 0.7, 0.5
    w_score, w_rwp = 0.5, 1.0

    components = ["Normalized Rwp", "Normalized score", "Balance score", "LLM"]
    colors = ["#a6d854", "lightblue", "mediumseagreen", "teal"]
    bar_height = 0.4
    spacing = 0.25

    sorted_items = sorted(interpretations.items(), key=lambda x: x[1]["unnormalized_posterior"], reverse=True)[::-1]
    y_positions = [i * (bar_height + spacing) for i in range(len(sorted_items))]
    labels = [f"Interpretation_{key.split('_')[1]}" for key, _ in sorted_items]

    fig, (ax_post, ax_breakdown) = plt.subplots(
        ncols=2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]}
    )

    max_posterior = max(d["unnormalized_posterior"] for _, d in sorted_items) * 100

    for idx, (interp_key, data) in enumerate(sorted_items):
        label = f"Interpretation_{interp_key.split('_')[1]}"
        bal = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]
        score = data["normalized_score"]
        rwp = data["normalized_rwp"]
        posterior = data["unnormalized_posterior"]

        # Compute contributions
        prior = (bal * w_b + llm * w_llm) / (w_b + w_llm)
        fit_quality = (score * w_score + rwp * w_rwp) / (w_score + w_rwp)
        C_balance = (bal * w_b / (w_b + w_llm)) * prior
        C_llm = (llm * w_llm / (w_b + w_llm)) * prior
        C_score = (score * w_score / (w_score + w_rwp)) * fit_quality
        C_rwp = (rwp * w_rwp / (w_score + w_rwp)) * fit_quality

        # Plot 1: Posterior bar (light blue)
        posterior_val = posterior * 100
        ax_post.barh([y_positions[idx]], [posterior_val], height=bar_height,
                     color="lightblue", edgecolor='none')

        if posterior_val > 6:
            ax_post.text(posterior_val - 1, y_positions[idx], f"{posterior_val:.1f}%",
                         ha='right', va='center', fontsize=10, fontweight='bold', color='black')
        else:
            ax_post.text(posterior_val + 1, y_positions[idx], f"{posterior_val:.1f}%",
                         ha='left', va='center', fontsize=10, fontweight='bold', color='black')

        ax_post.text(-1.5, y_positions[idx], label, va='center', ha='right', fontsize=12)

        # Plot 2: Contribution breakdown (normalized to 100%)
        total = C_rwp + C_score + C_balance + C_llm
        normalized_contributions = [100 * x / total for x in [C_rwp, C_score, C_balance, C_llm]]

        left = 0
        for val, color in zip(normalized_contributions, colors):
            ax_breakdown.barh([y_positions[idx]], [val], left=left, height=bar_height,
                              color=color, edgecolor='white')
            if val > 5:
                ax_breakdown.text(left + val / 2, y_positions[idx], f"{val:.1f}%",
                                  ha='center', va='center', fontsize=10, fontweight='bold', color='black')
            else:
                ax_breakdown.text(left + val + 0.5, y_positions[idx], f"{val:.1f}%",
                                  ha='left', va='center', fontsize=10, fontweight='bold', color='black')
            left += val

    # Posterior axis formatting
    ax_post.set_yticks([])
    ax_post.set_xlim(0, max_posterior * 1.25)
    ax_post.set_title("Posterior Probability per Interpretation", fontsize=14, fontweight='bold')
    ax_post.set_xlabel("Posterior (%)", fontsize=12)
    ax_post.spines['top'].set_visible(False)
    ax_post.spines['right'].set_visible(False)

    # Breakdown axis formatting
    ax_breakdown.set_yticks([])
    ax_breakdown.set_xlim(0, 100)
    ax_breakdown.set_title("Normalized Contribution Breakdown", fontsize=14, fontweight='bold')
    ax_breakdown.set_xlabel("Contribution to Final Probability (%)", fontsize=12)
    ax_breakdown.spines['top'].set_visible(False)
    ax_breakdown.spines['right'].set_visible(False)
    ax_breakdown.legend(components, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

    plt.tight_layout()

    # Save
    output_dir = get_output_dir(target, project_number)  # assumes you have this function defined
    save_path = os.path.join(output_dir, f"contribution_breakdown_dual_normalized_v2.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Enhanced dual contribution plot saved to: {save_path}")


def plot_contribution_pie_scaled(interpretations, project_number, target):
    """
    - Pie chart size proportional to posterior probability.
    - Max 4 interpretations per row (6 -> 3 x 2).
    - Legend under the pies.
    - Larger titles and percentage labels.
    - Label placement:
        * if posterior < 0.39:
              - labels for slices with pct >= 40% go inside the slice
              - all other labels outside with straight radial lines, radially staggered
        * otherwise:
              - labels inside unless the slice is too narrow, then outside.
    """
    import os
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    components = ["Rwp", "Score", "Balance", "LLM"]
    colors = ["#a6d854", "lightblue", "mediumseagreen", "teal"]

    # Weights
    w_b, w_llm = 0.7, 0.5
    w_score, w_rwp = 0.5, 1.0

    # Sort interpretations by posterior
    sorted_items = sorted(
        interpretations.items(),
        key=lambda x: x[1]["unnormalized_posterior"],
        reverse=True,
    )
    n = len(sorted_items)
    max_posterior = max(d["unnormalized_posterior"] for _, d in sorted_items)
    min_rwp_key = max(sorted_items, key=lambda x: x[1]["normalized_rwp"])[0]
    max_post_key = sorted_items[0][0]

    # Grid layout: max 4 per row, but 6 -> 3 x 2
    if n <= 4:
        ncols = n
        nrows = 1
    elif n == 6:
        ncols = 3
        nrows = 2
    else:
        ncols = 4
        nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 4.0 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55)

    axes_flat = axes.flatten()
    small_posterior_threshold = 0.39  # < 39% = "tiny" pies

    for i, (label, data) in enumerate(sorted_items):
        ax = axes_flat[i]

        posterior = data["unnormalized_posterior"]
        rwp = data["normalized_rwp"]
        score = data["normalized_score"]
        balance = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]

        # Contributions
        prior = (balance * w_b + llm * w_llm) / (w_b + w_llm)
        fit_quality = (score * w_score + rwp * w_rwp) / (w_score + w_rwp)
        C_balance = (balance * w_b / (w_b + w_llm)) * prior
        C_llm = (llm * w_llm / (w_b + w_llm)) * prior
        C_score = (score * w_score / (w_score + w_rwp)) * fit_quality
        C_rwp = (rwp * w_rwp / (w_score + w_rwp)) * fit_quality

        values = [C_rwp, C_score, C_balance, C_llm]
        total = sum(values)
        norm_values = [v / total for v in values]

        # Pie radius scaled by posterior
        radius = np.sqrt(posterior / max_posterior)

        wedges, _ = ax.pie(
            norm_values,
            startangle=90,
            colors=colors,
            radius=radius,
        )

        for j, wedge in enumerate(wedges):
            pct = norm_values[j] * 100
            label_text = f"{pct:.1f}%"

            theta1, theta2 = wedge.theta1, wedge.theta2
            angle = (theta1 + theta2) / 2.0
            x = np.cos(np.radians(angle))
            y = np.sin(np.radians(angle))

            if posterior < small_posterior_threshold:
                if pct >= 40.0:
                    # Big segment: label fully inside the slice, slightly smaller font
                    ax.text(
                        0.5 * x * radius,   # was 0.6 -> deeper inside
                        0.5 * y * radius,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=11,        # a bit smaller to stay well inside
                        fontweight="bold",
                    )
                else:
                    # Smaller segments: labels outside with straight-ish radial lines
                    # Radial + perpendicular staggering to avoid overlap
                    base_r = 1.38 * radius

                    # a bit further out for the tiniest slices
                    if pct < 10:
                        base_r += 0.18 * radius
                    elif pct < 25:
                        base_r += 0.10 * radius
                    else:   # 25–40%
                        base_r += 0.05 * radius
                    if 15.4 <= pct <= 18:
                        base_r += 0.30 * radius 

                    # Perpendicular offset based on slice index (spread labels sideways)
                    # Order [-1.5, -0.5, 0.5, 1.5] for j = 0..3
                    j_centered = j - 1.5

                    perp_scale = 0.25 * radius   # was 0.18 -> stronger separation
                    dx = j_centered * perp_scale * (-y)   # (-y, x) is ⟂ to (x, y)
                    dy = j_centered * perp_scale * (x)

                    text_r = base_r

                    ax.annotate(
                        label_text,
                        xy=(x * radius, y * radius),
                        xytext=(x * text_r + dx, y * text_r + dy),
                        arrowprops=dict(
                            arrowstyle="-",
                            color="black",
                            lw=1.4,
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                    )
            else:
                # ---- LARGER PIES ----
                angle_span_rad = np.radians(theta2 - theta1)
                chord_length = 2 * radius * np.sin(angle_span_rad / 2.0)

                approx_char_width = 0.06  # rough text width per char
                text_width = len(label_text) * approx_char_width

                place_outside = chord_length < text_width

                if place_outside:
                    text_r = 1.25 * radius
                    ax.annotate(
                        label_text,
                        xy=(x * radius, y * radius),
                        xytext=(x * text_r, y * text_r),
                        arrowprops=dict(
                            arrowstyle="-",
                            color="black",
                            lw=1.2,
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        0.6 * x * radius,
                        0.6 * y * radius,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                    )

        # Titles
        interp_id = label.split("_")[1]
        title_color = (
            "darkcyan"
            if label == max_post_key
            else "cyan"
            if label == min_rwp_key
            else "black"
        )
        ax.set_title(
            f"Interpretation {interp_id}\nPosterior: {posterior * 100:.1f}%",
            fontsize=14,
            fontweight="bold",
            color=title_color,
        )
        ax.set_aspect("equal")

    # Hide unused axes
    for k in range(i + 1, len(axes_flat)):
        axes_flat[k].axis("off")

    # Different margins for 1 vs 2 rows
    if nrows == 1:
        legend_y = 0.10
        rect_bottom = 0.20
        rect_top = 0.88   # extra headroom for labels
    else:
        legend_y = 0.20
        rect_bottom = 0.16
        rect_top = 0.94

    # Legend under pies
    fig.legend(
        components,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        fontsize=11,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, rect_bottom, 0.98, rect_top])

    # Save figure
    output_dir = get_output_dir(target, project_number)
    save_path = os.path.join(output_dir, "contribution_pies_line_highlighted.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved to: {save_path}")


def plot_contribution_decomposition_dual_normalized_right_v3(interpretations, project_number, target):
    # Weights
    w_b, w_llm = 0.7, 0.5
    w_score, w_rwp = 0.5, 1.0

    components = ["Normalized Rwp", "Normalized score", "Balance score", "LLM"]
    colors = ["#a6d854", "lightblue", "mediumseagreen", "teal"]
    bar_height = 0.4
    spacing = 0.25

    sorted_items = sorted(interpretations.items(), key=lambda x: x[1]["unnormalized_posterior"], reverse=True)[::-1]
    y_positions = [i * (bar_height + spacing) for i in range(len(sorted_items))]

    fig, (ax_post, ax_breakdown) = plt.subplots(ncols=2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})

    max_posterior = max(d["unnormalized_posterior"] for _, d in sorted_items) * 100

    # Identify special interpretations
    lowest_rwp_key = max(sorted_items, key=lambda x: x[1]["normalized_rwp"])[0]
    highest_post_key = max(sorted_items, key=lambda x: x[1]["unnormalized_posterior"])[0]

    for idx, (interp_key, data) in enumerate(sorted_items):
        label = f"Interpretation_{interp_key.split('_')[1]}"
        bal = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]
        score = data["normalized_score"]
        rwp = data["normalized_rwp"]
        posterior = data["unnormalized_posterior"]

        # Compute contributions
        prior = (bal * w_b + llm * w_llm) / (w_b + w_llm)
        fit_quality = (score * w_score + rwp * w_rwp) / (w_score + w_rwp)
        C_balance = (bal * w_b / (w_b + w_llm)) * prior
        C_llm = (llm * w_llm / (w_b + w_llm)) * prior
        C_score = (score * w_score / (w_score + w_rwp)) * fit_quality
        C_rwp = (rwp * w_rwp / (w_score + w_rwp)) * fit_quality

        # Left plot: All bars same color
        posterior_val = posterior * 100
        ax_post.barh([y_positions[idx]], [posterior_val], height=bar_height,
                     color="lightblue", edgecolor='none')

        # Posterior value text
        if posterior_val > 6:
            ax_post.text(posterior_val - 1, y_positions[idx], f"{posterior_val:.1f}%",
                         ha='right', va='center', fontsize=10, fontweight='bold', color='black')
        else:
            ax_post.text(posterior_val + 1, y_positions[idx], f"{posterior_val:.1f}%",
                         ha='left', va='center', fontsize=10, fontweight='bold', color='black')

        # Label color logic
        if interp_key == highest_post_key:
            label_color = 'darkcyan'
        elif interp_key == lowest_rwp_key:
            label_color = 'cyan'
        else:
            label_color = 'black'

        ax_post.text(-1.5, y_positions[idx], label, va='center', ha='right',
                     fontsize=12, fontweight='bold', color=label_color)

        # Right plot: Contribution breakdown
        total = C_rwp + C_score + C_balance + C_llm
        normalized_contributions = [100 * x / total for x in [C_rwp, C_score, C_balance, C_llm]]
        left = 0
        for val, color in zip(normalized_contributions, colors):
            ax_breakdown.barh([y_positions[idx]], [val], left=left, height=bar_height,
                              color=color, edgecolor='white')
            if val > 5:
                ax_breakdown.text(left + val / 2, y_positions[idx], f"{val:.1f}%",
                                  ha='center', va='center', fontsize=10, fontweight='bold', color='black')
            else:
                ax_breakdown.text(left + val + 0.5, y_positions[idx], f"{val:.1f}%",
                                  ha='left', va='center', fontsize=10, fontweight='bold', color='black')
            left += val

    # Format left plot
    ax_post.set_yticks([])
    ax_post.set_xlim(0, max_posterior * 1.25)
    ax_post.set_title("Posterior Probability per Interpretation", fontsize=14, fontweight='bold')
    ax_post.set_xlabel("Posterior (%)", fontsize=12)
    ax_post.spines['top'].set_visible(False)
    ax_post.spines['right'].set_visible(False)

    # Format right plot
    ax_breakdown.set_yticks([])
    ax_breakdown.set_xlim(0, 100)
    ax_breakdown.set_title("Normalized Contribution Breakdown", fontsize=14, fontweight='bold')
    ax_breakdown.set_xlabel("Contribution to Final Probability (%)", fontsize=12)
    ax_breakdown.spines['top'].set_visible(False)
    ax_breakdown.spines['right'].set_visible(False)
    ax_breakdown.legend(components, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

    plt.tight_layout()

    # Save figure
    output_dir = get_output_dir(target, project_number)  # assumes you have this function defined
    save_path = os.path.join(output_dir, f"contribution_breakdown_dual_normalized_v3.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved to: {save_path}")


def plot_contribution_decomposition_dual_normalized_right_v4(
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
        key=lambda x: x[1]["unnormalized_posterior"],
        reverse=True
    )[::-1]

    y_positions = [i * (bar_height + spacing) for i in range(len(sorted_items))]

    fig, ax = plt.subplots(figsize=(12, 6))

    max_posterior = max(d["unnormalized_posterior"] for _, d in sorted_items) * 100

    # Identify special interpretations for coloring labels
    lowest_rwp_key = max(sorted_items, key=lambda x: x[1]["normalized_rwp"])[0]
    highest_post_key = max(sorted_items, key=lambda x: x[1]["unnormalized_posterior"])[0]

    for idx, (interp_key, data) in enumerate(sorted_items):
        label = f"Interpretation_{interp_key.split('_')[1]}"

        bal = data["balance_score"]
        llm = data["LLM_interpretation_likelihood"]
        score = data["normalized_score"]
        rwp = data["normalized_rwp"]
        posterior = data["unnormalized_posterior"]

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
        output_dir, "contribution_breakdown_dual_normalized_v4.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved to: {save_path}")