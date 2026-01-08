#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rich.console import Console
from rich.table import Table
import json
from pathlib import Path
import argparse
import pandas as pd
import ast
import re

console = Console(record=True)

# ----------------------
# Helpers
# ----------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def split_phase_name(phase: str):
    """Split into base and space group-like suffix: 'V6013_69' → ('V6013', '_69')"""
    match = re.match(r"^(.+?)(_\d+|_[A-Za-z]+)?$", phase)
    if match:
        base = match.group(1)
        suffix = match.group(2) or ""
        return base, suffix
    return phase, ""

def format_phase(base: str, suffix: str, color: str | None = None):
    if color:
        return f"{base}[{color}]{suffix}[/]"
    else:
        return f"{base}{suffix}"

def compute_trust_score(interp: dict) -> float:
    """
    Computes a smoother numeric trust score (0 to 1) based on deviation from ideal values.
    Penalizes:
      - LLM interpretation likelihood below 1.0
      - Balance score below 1.0
      - Signal above background below 85%
    """
    llm = float(interp.get("LLM_interpretation_likelihood", 1.0))
    balance_score = float(interp.get("balance_score", 1.0))
    signal_above_bkg = float(interp.get("signal_above_bkg", 100.0))

    penalty_llm = max(0.0, min(1.0, 1.0 - llm))
    penalty_balance = max(0.0, min(1.0, 1.0 - balance_score))
    penalty_signal = max(0.0, min(1.0, (85.0 - signal_above_bkg) / 85.0))

    total_penalty = (penalty_llm + penalty_balance + penalty_signal) / 3.0
    trust_score = max(0.0, 1.0 - total_penalty)
    return round(trust_score, 3)

def color(val1, val2, is_lower_better=False):
    """Return colored strings for values based on comparison."""
    if val1 is None or val2 is None:
        return str(val1), str(val2)

    # Detect integer values
    is_integer = isinstance(val1, int) and isinstance(val2, int)

    try:
        val1_rounded = round(val1, 4)
        val2_rounded = round(val2, 4)
    except TypeError:
        # Fallback if one value isn't numeric
        return str(val1), str(val2)

    if val1_rounded == val2_rounded:
        return (
            f"{val1}" if is_integer else f"{val1_rounded:.4f}",
            f"{val2}" if is_integer else f"{val2_rounded:.4f}",
        )

    better = val1_rounded < val2_rounded if is_lower_better else val1_rounded > val2_rounded
    val1_fmt = f"{val1}" if is_integer else f"{val1_rounded:.4f}"
    val2_fmt = f"{val2}" if is_integer else f"{val2_rounded:.4f}"

    return (
        f"[bold green]{val1_fmt}[/]" if better else f"[red1]{val1_fmt}[/]",
        f"[bold red1]{val2_fmt}[/]" if better else f"[bold green]{val2_fmt}[/]"
    )

def summarize(interp: dict) -> dict:
    trust_score = interp.get("trust_score")
    if trust_score is None:
        trust_score = compute_trust_score(interp)
    return {
        "phases": ", ".join(interp.get("phases", [])),
        "rwp": interp.get("rwp"),
        "search_result_rwp": interp.get("search_result_rwp"),
        "score": interp.get("score"),
        "search_result_score": interp.get("search_result_score"),
        "llm_likelihood": interp.get("LLM_interpretation_likelihood"),
        "balance_score": interp.get("balance_score"),
        "missing_peaks": interp.get("missing_peaks"),
        "extra_peaks": interp.get("extra_peaks"),
        "excess_bkg": interp.get("normalized_flag"),
        "signal_above_bkg": interp.get("signal_above_bkg"),
        "trust_score": round(trust_score, 3) if trust_score is not None else None,
        "trustworthy": interp.get("trustworthy"),
    }

def try_get_synthesis_row(synthesis_df: pd.DataFrame | None, sample_name: str):
    """
    Given a synthesis_df and a sample name, returns the first matching row or None.
    Tries exact match, then a single '_' <-> '-' swap variant.
    """
    if synthesis_df is None or "Name" not in synthesis_df.columns:
        return None

    # Exact full match
    filtered = synthesis_df[synthesis_df["Name"].astype(str).str.fullmatch(sample_name, na=False)]
    if filtered.empty:
        swapped = sample_name.replace("_", "-") if "_" in sample_name else sample_name.replace("-", "_")
        filtered = synthesis_df[synthesis_df["Name"].astype(str).str.fullmatch(swapped, na=False)]

    if filtered.empty:
        return None
    return filtered.iloc[0].copy()

def dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ----------------------
# Core comparison
# ----------------------
def compare_dara_vs_aif(json_data: dict, sample_names: list[str], synthesis_dfs: dict[str, pd.DataFrame], report_suffix: str = "mixed"):
    """
    synthesis_dfs: dict like {"TRI": df, "ARR": df, "PG": df, "MINES": df}
    report_suffix: used in output filenames
    """
    html_all = ""
    html_diff = ""

    for sample_name in sample_names:
        if sample_name not in json_data:
            console.print(f"[red]⚠️ Sample '{sample_name}' not found in JSON.[/]")
            continue

        interpretations = json_data[sample_name]
        if not isinstance(interpretations, dict) or not interpretations:
            console.print(f"[yellow]No interpretations for sample '{sample_name}'.[/]")
            continue

        # AIF pick: highest posterior; DARA pick: lowest RWP
        aif_interp_id, aif_interp = max(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0)
        )
        dara_interp_id, dara_interp = min(
            interpretations.items(),
            key=lambda x: x[1].get("rwp", float("inf"))
        )

        dara_summary = summarize(dara_interp)
        aif_summary = summarize(aif_interp)

        console.print(f"\n📌 Sample: [bold cyan]{sample_name}[/]")

        # Synthesis metadata (based on project prefix)
        project_prefix = sample_name.split("_")[0].upper() if "_" in sample_name else sample_name[:3].upper()
        synthesis_df = synthesis_dfs.get(project_prefix)
        row = try_get_synthesis_row(synthesis_df, sample_name)

        if row is not None:
            try:
                precursors_list = ast.literal_eval(row.get('Precursors', '[]'))
                if not isinstance(precursors_list, list):
                    precursors_list = [str(precursors_list)]
            except Exception:
                precursors_list = [str(row.get('Precursors', ''))]

            synthesis_text = f"""
[bold]Synthesis Description:[/bold]
• Project: [magenta]{project_prefix}[/]
• Target: [cyan]{row.get('Target','')}[/]
• Precursors: {", ".join(map(str, precursors_list))}
• Temperature: {row.get('Temperature (C)','')}
• Duration (h): {row.get('Dwell Duration (h)','')}
• Furnace: {row.get('Furnace','')}
""".strip()
            console.print(synthesis_text)
        else:
            console.print(f"[yellow]No synthesis row found for '{sample_name}' in project '{project_prefix}'.[/]")

        # Table of metrics
        table = Table(show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column(f"DARA ({dara_interp_id})", style="cyan")
        table.add_column(f"AIF ({aif_interp_id})", style="cyan")

        # Phase highlighting with base vs suffix (e.g., SG) differences
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
        for p in dara_phases_raw:
            base, suffix = split_phase_name(p)
            if p not in aif_set:
                highlighted_dara_phases.append(f"[bold orange1]{p}[/]")
            elif suffix not in aif_bases.get(base, set()):
                highlighted_dara_phases.append(format_phase(base, suffix, "bold orange1"))
            else:
                highlighted_dara_phases.append(p)

        highlighted_aif_phases = []
        for p in aif_phases_raw:
            base, suffix = split_phase_name(p)
            if p not in dara_set:
                highlighted_aif_phases.append(f"[bold deep_pink1]{p}[/]")
            elif suffix not in dara_bases.get(base, set()):
                highlighted_aif_phases.append(format_phase(base, suffix, "bold deep_pink1"))
            else:
                highlighted_aif_phases.append(p)

        table.add_row("Phases", ", ".join(highlighted_dara_phases), ", ".join(highlighted_aif_phases))

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
            dval, aval = dara_summary.get(key), aif_summary.get(key)
            dval_str, aval_str = color(dval, aval, is_lower_better)
            table.add_row(key, dval_str, aval_str)

        d_trust_str = "[bold green]True[/]" if dara_summary.get("trustworthy", False) else "[bold red1]False[/]"
        a_trust_str = "[bold green]True[/]" if aif_summary.get("trustworthy", False) else "[bold red1]False[/]"
        table.add_row("Trustworthy", d_trust_str, a_trust_str)

        console.print(table)

        # Export HTML for this sample (append to combined)
        html_sample = console.export_html(inline_styles=True, clear=True)
        html_all += html_sample
        if aif_interp_id != dara_interp_id:
            html_diff += html_sample

    # Write single combined reports
    all_path = f"aif_dara_report_all_{report_suffix}.html"
    diff_path = f"aif_dara_report_diff_selections_{report_suffix}.html"

    with open(all_path, "w", encoding="utf-8") as f:
        f.write(
            "<html><head><meta charset='UTF-8'>"
            "<style>span { font-weight: bold !important; }</style></head><body>"
            + html_all +
            "</body></html>"
        )
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(
            "<html><head><meta charset='UTF-8'>"
            "<style>span { font-weight: bold !important; }</style></head><body>"
            + html_diff +
            "</body></html>"
        )

    console.print(f"\n[bold green]✓ Wrote[/] {all_path}")
    console.print(f"[bold green]✓ Wrote[/] {diff_path}")

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare DARA and AIF interpretations for selected sample(s), across mixed projects."
    )
    parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
    parser.add_argument(
        "samples",
        nargs="+",
        help="Sample names to analyze (e.g., TRI_91 ARR_12 PG_03) or 'all' to process all in the JSON."
    )
    # Optional: override default CSV paths with flags if needed later

    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        raise SystemExit(1)

    json_data = load_json(json_path)
    sample_list = list(json_data.keys()) if args.samples == ["all"] else args.samples

    # Load all project CSVs once (OK if some are missing)
    project_csv_map = {
        "TRI": "../../data/alab_synthesis_data/synthesis_TRI.csv",
        "ARR": "../../data/alab_synthesis_data/synthesis_ARR.csv",
        "MINES": "../../data/alab_synthesis_data/synthesis_MINES.csv",
        "PG": "../../data/alab_synthesis_data/synthesis_PG_genome.csv",
    }

    synthesis_dfs: dict[str, pd.DataFrame] = {}
    for key, csv_path in project_csv_map.items():
        try:
            df = pd.read_csv(csv_path)
            synthesis_dfs[key] = df
        except Exception as e:
            console.print(f"[yellow][WARN][/yellow] Could not load {key} CSV at {csv_path}: {e}")

    report_suffix = Path(args.json_file).stem

    console.print(f"\n📂 Processing [bold]{len(sample_list)}[/] samples (mixed projects allowed)")
    compare_dara_vs_aif(json_data, sample_list, synthesis_dfs, report_suffix=report_suffix)

    # Optional: a single text transcript of the Rich output
    txt_path = f"aif_dara_report_{report_suffix}.txt"
    console.save_text(txt_path)
    console.print(f"[bold green]✓ Wrote[/] {txt_path}")

# # All samples in the JSON (mixed ARR + PG + TRI + MINES are fine)
# python compare_AIF_Dara_selections_16extraevaluations.py ../../data/xrd_data/interpretations/interpretations_16_more_forevaluation.json all
# # Only certain mixed samples
# python compare_AIF_Dara_selections.py_16extraevaluations.py ../../data/xrd_data/interpretations/interpretations_16_more_forevaluation.json ARR_12 PG_07 PG_19