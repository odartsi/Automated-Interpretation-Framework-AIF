from rich.console import Console
from rich.table import Table
import json
from pathlib import Path
import argparse
import pandas as pd
import ast
import re

console = Console(record=True)


def split_phase_name(phase):
    """Split into base and space group: 'V6013_69' → ('V6013', '_69')"""
    match = re.match(r"^(.+?)(_\d+|_[A-Za-z]+)?$", phase)
    if match:
        base = match.group(1)
        suffix = match.group(2) or ""
        return base, suffix
    return phase, ""

def format_phase(base, suffix, color=None):
    if color:
        return f"{base}[{color}]{suffix}[/]"
    else:
        return f"{base}{suffix}"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_trust_score(interp):
    """
    Computes a smoother numeric trust score (0 to 1) based on deviation from ideal values.
    Penalizes:
      - LLM interpretation likelihood below 1.0
      - Balance score below 1.0
      - Signal above background below 85%
    
    Parameters:
        interp (dict): An interpretation dictionary with keys:
                       'LLM_interpretation_likelihood', 'balance_score', 'signal_above_bkg'
    
    Returns:
        float: trust_score ∈ [0, 1]
    """
    # Safe float conversions
    llm = float(interp.get("LLM_interpretation_likelihood", 1.0))
    balance_score = float(interp.get("balance_score", 1.0))
    signal_above_bkg = float(interp.get("signal_above_bkg", 100.0))

    # Compute individual penalties (clipped between 0 and 1)
    penalty_llm = max(0.0, min(1.0, 1.0 - llm))
    penalty_balance = max(0.0, min(1.0, 1.0 - balance_score))
    penalty_signal = max(0.0, min(1.0, (85.0 - signal_above_bkg) / 85.0))

    # Average penalty
    total_penalty = (penalty_llm + penalty_balance + penalty_signal) / 3.0

    trust_score = max(0.0, 1.0 - total_penalty)
    return round(trust_score, 3)

def color(val1, val2, is_lower_better=False):
    """Return colored strings for values based on comparison."""
    if val1 is None or val2 is None:
        return str(val1), str(val2)

    # Detect integer values
    is_integer = isinstance(val1, int) and isinstance(val2, int)

    val1_rounded = round(val1, 4)
    val2_rounded = round(val2, 4)

    if val1_rounded == val2_rounded:
        return f"{val1}" if is_integer else f"{val1_rounded:.4f}", f"{val2}" if is_integer else f"{val2_rounded:.4f}"

    better = val1_rounded < val2_rounded if is_lower_better else val1_rounded > val2_rounded
    val1_fmt = f"{val1}" if is_integer else f"{val1_rounded:.4f}"
    val2_fmt = f"{val2}" if is_integer else f"{val2_rounded:.4f}"

    return (
        f"[bold green]{val1_fmt}[/]" if better else f"[red1]{val1_fmt}[/]",
        f"[bold red1]{val2_fmt}[/]" if better else f"[bold green]{val2_fmt}[/]"
    )
def summarize(interp):
    trust_score = interp.get("trust_score")
    if trust_score is None:
        trust_score = compute_trust_score(interp) 
    return {
        "phases": ", ".join(interp.get("phases", [])),
        "rwp": interp.get("rwp"),
        "search_result_rwp" : interp.get("search_result_rwp"),
        "score": interp.get("score"),
        "search_result_score" : interp.get("search_result_score"),
        # "dara_score": interp.get("dara_score"),
        # "normalized_rwp": interp.get("normalized_rwp"),
        # "normalized_score": interp.get("normalized_score"),
        "llm_likelihood": interp.get("LLM_interpretation_likelihood"),
        "balance_score": interp.get("balance_score"),
        "missing_peaks": interp.get("missing_peaks"),
        "extra_peaks": interp.get("extra_peaks"),
        "excess_bkg": interp.get("normalized_flag"),
        "signal_above_bkg": interp.get("signal_above_bkg"),
        "trust_score": round(trust_score, 3) if trust_score is not None else None,
        "trustworthy": interp.get("trustworthy"),
    }

def compare_dara_vs_aif(json_data, sample_names,synthesis_df):
    html_all = ""
    html_diff = ""

    for sample_name in sample_names:
        if sample_name not in json_data:
            console.print(f"[red]⚠️ Sample '{sample_name}' not found.[/]")
            continue

        interpretations = json_data[sample_name]

        # Find interpretation with highest posterior_probability
        aif_interp_id, aif_interp = max(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0)
        )

        dara_interp_id, dara_interp = min(
            interpretations.items(),
            key=lambda x: x[1].get("rwp", 0)
        )

        dara_summary = summarize(dara_interp)
        aif_summary = summarize(aif_interp)

        console.print(f"\n📌 Sample: [bold cyan]{sample_name}[/]")

        # Load synthesis CSV once
        # synthesis_csv = "../data/alab_synthesis_data/synthesis_and_predictions_new_test.csv"
        synthesis_df = pd.read_csv(synthesis_csv)


        filtered_df = synthesis_df[synthesis_df["Name"].str.contains(rf'^{sample_name}$', na=False)]
        if filtered_df.empty:
            swapped_name = sample_name.replace("_", "-") if "_" in sample_name else sample_name.replace("-", "_")
            filtered_df = synthesis_df[synthesis_df["Name"].str.contains(rf'^{swapped_name}$', na=False)]

        if not filtered_df.empty:
            synthesis_row = filtered_df.iloc[0].copy()
            try:
                precursors_list = ast.literal_eval(synthesis_row['Precursors'])
            except:
                precursors_list = [synthesis_row['Precursors']]

            synthesis_text = f"""
        [bold]Synthesis Description:[/bold]
        • Target: [cyan]{synthesis_row['Target']}[/cyan]
        • Precursors: {", ".join(precursors_list)}
        • Temperature: {synthesis_row['Temperature (C)']}°C
        • Duration: {synthesis_row['Dwell Duration (h)']} h
        • Furnace: {synthesis_row['Furnace']}
            """.strip()
            console.print(synthesis_text)

        table = Table(show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column(f"DARA ({dara_interp_id})", style="cyan")
        table.add_column(f"AIF ({aif_interp_id})", style="cyan")

        # Color-coded phase comparison
        def dedup(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        dara_phases_raw = dedup(dara_interp.get("phases", []))
        aif_phases_raw = dedup(aif_interp.get("phases", []))
        dara_set = set(dara_phases_raw)
        aif_set = set(aif_phases_raw)

        from collections import defaultdict
        dara_bases = defaultdict(set)
        aif_bases = defaultdict(set)

        for p in dara_phases_raw:
            base, suffix = split_phase_name(p)
            dara_bases[base].add(suffix)

        for p in aif_phases_raw:
            base, suffix = split_phase_name(p)
            aif_bases[base].add(suffix)

        highlighted_dara_phases = []
        highlighted_aif_phases = []

        for p in dara_phases_raw:
            base, suffix = split_phase_name(p)
            if p not in aif_set:
                highlighted_dara_phases.append(f"[bold orange1]{p}[/]")
            elif suffix not in aif_bases.get(base, set()):
                highlighted_dara_phases.append(format_phase(base, suffix, "bold orange1"))
            else:
                highlighted_dara_phases.append(p)

        for p in aif_phases_raw:
            base, suffix = split_phase_name(p)
            if p not in dara_set:
                highlighted_aif_phases.append(f"[bold deep_pink1]{p}[/]")
            elif suffix not in dara_bases.get(base, set()):
                highlighted_aif_phases.append(format_phase(base, suffix, "bold deep_pink1"))
            else:
                highlighted_aif_phases.append(p)

        dara_phases_str = ", ".join(highlighted_dara_phases)
        aif_phases_str = ", ".join(highlighted_aif_phases)
        table.add_row("Phases", dara_phases_str, aif_phases_str)

        metrics = [
            ("rwp", True),
            ("search_result_rwp", True),
            ("score", False),
            ("search_result_score", False),
            ("llm_likelihood", False),
            ("balance_score", False),
            ("missing_peaks", True),
            ("extra_peaks", True),
            ("excess_bkg", True),
            ("signal_above_bkg", False),
            ("trust_score", False),
        ]

        for key, is_lower_better in metrics:
            dval, aval = dara_summary[key], aif_summary[key]
            dval_str, aval_str = color(dval, aval, is_lower_better)
            table.add_row(key, dval_str, aval_str)

        d_trust_str = "[bold green]True[/]" if dara_summary.get("trustworthy", False) else "[bold red1]False[/]"
        a_trust_str = "[bold green]True[/]" if aif_summary.get("trustworthy", False) else "[bold red1]False[/]"
        table.add_row("Trustworthy", d_trust_str, a_trust_str)

        console.print(table)

        # Export HTML for this sample
        html_sample = console.export_html(inline_styles=True, clear=True)

        # Save to all report
        html_all += html_sample

        # Save to diff-only report if AIF selected interpretation is not dara_interp_id
        if aif_interp_id != dara_interp_id:
            html_diff += html_sample

    # Save combined reports
    # project_suffix = f"_{args.project}" if args.project else ""
    project_suffix = f"_{project_key}"

    # Save combined reports
    with open(f"aif_dara_report_all{project_suffix}.html", "w") as f:
        f.write(
            "<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
            + html_all +
            "</body></html>"
        )

    with open(f"aif_dara_report_diff_selections{project_suffix}.html", "w") as f:
        f.write(
            "<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
            + html_diff +
            "</body></html>"
        )
    # with open("aif_dara_report_all.html", "w") as f:
    #     f.write(
    #         "<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
    #         + html_all +
    #         "</body></html>"
    #     )

    # with open("aif_dara_report_diff_selections.html", "w") as f:
    #     f.write(
    #         "<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
    #         + html_diff +
    #         "</body></html>"
    #     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compare DARA and AIF interpretations for selected sample(s).")
#     parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
#     parser.add_argument("samples", nargs="+", help="Sample names to analyze (e.g., TRI_91 TRI_82 or 'all')")
#     parser.add_argument("--project", help="Project prefix to filter samples (e.g., TRI, ARR).", default=None)
#     args = parser.parse_args()

#     json_path = Path(args.json_file)
#     if not json_path.exists():
#         print(f"❌ File not found: {json_path}")
#     else:
#         json_data = load_json(json_path)

#         if args.samples == ["all"]:
#             sample_list = list(json_data.keys())
#         else:
#             sample_list = args.samples

#         if args.project:
#             sample_list = [s for s in sample_list if s.startswith(args.project)]

#         project_csv_map = {
#             "TRI": "../data/alab_synthesis_data/synthesis_TRI.csv",
#             "ARR": "../data/alab_synthesis_data/synthesis_ARR.csv",
#             "MINES": "../data/alab_synthesis_data/synthesis_MINES.csv",
#             "PG": "../data/alab_synthesis_data/synthesis_PG_genome.csv",
#         }

#         # Default to TRI if not specified
#         project_key = args.project.upper() if args.project else "TRI"
#         synthesis_csv = project_csv_map.get(project_key, project_csv_map["TRI"])

#         try:
#             synthesis_df = pd.read_csv(synthesis_csv)
#         except Exception as e:
#             print(f"❌ Failed to load synthesis CSV for project {project_key}: {e}")
#             exit(1)

#         compare_dara_vs_aif(json_data, sample_list, synthesis_df)

#         project_suffix = f"_{args.project}" if args.project else ""

#         console.save_text(f"aif_dara_report{project_suffix}.txt")
from collections import defaultdict

def group_samples_by_project(samples):
    project_map = defaultdict(list)
    for sample in samples:
        # Extract the project prefix (e.g., 'TRI', 'ARR', etc.)
        project_prefix = sample.split("_")[0]
        project_map[project_prefix].append(sample)
    return project_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DARA and AIF interpretations for selected sample(s).")
    parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
    parser.add_argument("samples", nargs="+", help="Sample names to analyze (e.g., TRI_91 TRI_82 or 'all')")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        exit(1)

    json_data = load_json(json_path)
    sample_list = list(json_data.keys()) if args.samples == ["all"] else args.samples

    # Group by project prefix
    sample_groups = group_samples_by_project(sample_list)

    project_csv_map = {
        "TRI": "../data/alab_synthesis_data/synthesis_TRI.csv",
        "ARR": "../data/alab_synthesis_data/synthesis_ARR.csv",
        "MINES": "../data/alab_synthesis_data/synthesis_MINES.csv",
        "PG": "../data/alab_synthesis_data/synthesis_PG_genome.csv",
        "R" : "../../data/xrd_data/RRUFF/metadata_rruff.csv"
    }

    for project_key, samples in sample_groups.items():
        synthesis_csv = project_csv_map.get(project_key.upper(), project_csv_map["R"])
        try:
            synthesis_df = pd.read_csv(synthesis_csv)
        except Exception as e:
            print(f"❌ Failed to load synthesis CSV for project {project_key}: {e}")
            continue

        print(f"\n📂 Processing project: {project_key} ({len(samples)} samples)")
        compare_dara_vs_aif(json_data, samples, synthesis_df)

        console.save_text(f"aif_dara_report_{project_key}.txt")

        

# python tests/compare_AIF_Dara_selections.py interpretations_LLM_newapproach_and_dara_score.json TRI_112

# python tests/compare_AIF_Dara_selections.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json TRI_182
#python tests/compare_AIF_Dara_selections.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json all 
# python tests/compare_AIF_Dara_selections.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json all  --project TRI
# python tests/compare_AIF_Dara_selections.py ../data/xrd_data/interpretations/interpretations_second_evaluation.json all  --project ARR