import json
import pandas as pd
import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path
from rich.console import Console
from rich.table import Table
from itertools import cycle
console = Console(record=True)

project_csv_map = {
    "TRI": "../data/alab_synthesis_data/synthesis_TRI.csv",
    "ARR": "../data/alab_synthesis_data/synthesis_ARR.csv",
    "MINES": "../data/alab_synthesis_data/synthesis_MINES.csv",
    "PG": "../data/alab_synthesis_data/synthesis_PG_genome.csv",
}
def assign_base_and_suffix_colors(all_interps):
    """Assigns consistent base color and distinct suffix color if needed."""
    from itertools import cycle

    BASE_COLORS = cycle([
        "cyan", "magenta", "green", "red", "orange1", "blue", "chartreuse1", "turquoise2",
        "medium_purple", "gold1", "orchid", "light_pink1", "light_steel_blue", "deep_sky_blue1"
    ])
    SUFFIX_COLORS = cycle([
        "yellow", "orange1", "plum1", "light_salmon1", "tomato3", "light_green", "spring_green1"
    ])

    # First collect all base → suffix mappings
    base_to_suffixes = defaultdict(set)
    interp_phase_map = {}

    for interp_id, interp in all_interps:
        phases = interp.get("phases", [])
        interp_phase_map[interp_id] = phases
        for phase in phases:
            base, suffix = split_phase_name(phase)
            base_to_suffixes[base].add(suffix)

    # Assign consistent base colors
    base_color_map = {base: next(BASE_COLORS) for base in sorted(base_to_suffixes)}
    # Assign suffix colors only for bases with multiple suffixes
    suffix_color_map = {
        base: {suffix: next(SUFFIX_COLORS) for suffix in sorted(suffixes)}
        for base, suffixes in base_to_suffixes.items()
        if len(suffixes) > 1
    }

    # Build formatted output
    formatted_phases = {}
    for interp_id, phases in interp_phase_map.items():
        formatted = []
        for phase in phases:
            base, suffix = split_phase_name(phase)
            base_color = base_color_map.get(base, None)
            suffix_color = suffix_color_map.get(base, {}).get(suffix, None)

            if base_color and suffix_color:
                formatted.append(f"[{base_color}]{base}[/{base_color}][{suffix_color}]{suffix}[/{suffix_color}]")
            elif base_color:
                formatted.append(f"[{base_color}]{base}{suffix}[/{base_color}]")
            else:
                formatted.append(phase)

        formatted_phases[interp_id] = ", ".join(formatted)

    return formatted_phases
def assign_unique_phase_colors(all_interps):
    """Assigns a unique color per full phase (including space group) across all interpretations."""
    from itertools import cycle

    COLOR_PALETTE = [
        "cyan", "magenta", "green", "red", "yellow", "orange1", "blue", "chartreuse1",
        "turquoise2", "medium_purple", "light_salmon1", "deep_sky_blue1", "aquamarine1",
        "gold1", "orchid", "plum1", "light_pink1", "light_steel_blue", "medium_orchid", "tomato3",
        "spring_green1", "slate_blue1", "light_coral", "indian_red", "sea_green1", "dark_orange"
    ]
    color_cycle = cycle(COLOR_PALETTE)

    # Collect all unique phases (including suffixes)
    all_phases = set()
    interp_phase_map = {}

    for interp_id, interp in all_interps:
        phases = interp.get("phases", [])
        interp_phase_map[interp_id] = phases
        all_phases.update(phases)

    # Assign each unique full phase its own color
    phase_color_map = {phase: next(color_cycle) for phase in sorted(all_phases)}

    # Format each interpretation's phase list with assigned color
    formatted_phases = {}
    for interp_id, phases in interp_phase_map.items():
        formatted = []
        for phase in phases:
            color = phase_color_map.get(phase, None)
            formatted.append(f"[{color}]{phase}[/]" if color else phase)
        formatted_phases[interp_id] = ", ".join(formatted)

    return formatted_phases

PHASE_COLORS = cycle([
    "bright_blue", "cyan", "magenta", "green", "red", "orange1", "yellow", "purple", 
    "chartreuse1", "turquoise2", "medium_purple", "light_salmon1", "deep_sky_blue1", "aquamarine1",
    "gold1", "orchid", "plum1", "light_pink1", "light_steel_blue", "medium_orchid", "tomato3"
])
def assign_phase_colors(all_interps):
    base_color_map = {}
    base_suffixes = defaultdict(set)
    all_phases = set()

    # Collect all base→suffix and all full phases
    for _, interp in all_interps:
        for phase in interp.get("phases", []):
            base, suffix = split_phase_name(phase)
            base_suffixes[base].add(suffix)
            all_phases.add(phase)

    # Assign a unique color per base
    for base in base_suffixes:
        base_color_map[base] = next(PHASE_COLORS)

    # Format each interpretation's phase list
    phase_display_map = {}
    for interp_id, interp in all_interps:
        formatted = []
        for phase in interp.get("phases", []):
            base, suffix = split_phase_name(phase)
            color = base_color_map.get(base, None)
            if color:
                # If multiple suffixes exist for same base, highlight the suffix
                if len(base_suffixes[base]) > 1:
                    formatted.append(format_phase(base, suffix, color))
                else:
                    formatted.append(f"[{color}]{base}{suffix}[/]")
            else:
                formatted.append(phase)  # fallback
        phase_display_map[interp_id] = ", ".join(formatted)

    return phase_display_map

def split_phase_name(phase):
    match = re.match(r"^(.+?)(_\d+|_[A-Za-z]+)?$", phase)
    if match:
        base = match.group(1)
        suffix = match.group(2) or ""
        return base, suffix
    return phase, ""


def format_phase(base, suffix, color=None):
    return f"{base}[{color}]{suffix}[/]" if color else f"{base}{suffix}"


def compute_trust_score(interp):
    llm = float(interp.get("LLM_interpretation_likelihood", 1.0))
    balance = float(interp.get("balance_score", 1.0))
    signal = float(interp.get("signal_above_bkg", 100.0))

    penalty_llm = max(0, min(1, 1 - llm))
    penalty_balance = max(0, min(1, 1 - balance))
    penalty_signal = max(0, min(1, (85.0 - signal) / 85.0))

    total_penalty = (penalty_llm + penalty_balance + penalty_signal) / 3.0
    return round(1 - total_penalty, 3)


def color(val1, val2, is_lower_better=False):
    if val1 is None or val2 is None:
        return str(val1), str(val2)
    is_int = isinstance(val1, int) and isinstance(val2, int)
    v1, v2 = round(val1, 4), round(val2, 4)

    if v1 == v2:
        return f"{v1}" if is_int else f"{v1:.4f}", f"{v2}" if is_int else f"{v2:.4f}"

    better = v1 < v2 if is_lower_better else v1 > v2
    fmt1 = f"{v1}" if is_int else f"{v1:.4f}"
    fmt2 = f"{v2}" if is_int else f"{v2:.4f}"
    return (
        f"[bold green]{fmt1}[/]" if better else f"[red1]{fmt1}[/]",
        f"[bold red1]{fmt2}[/]" if better else f"[bold green]{fmt2}[/]"
    )


def summarize(interp):
    trust = interp.get("trust_score", compute_trust_score(interp))
    return {
        "phases": ", ".join(interp.get("phases", [])),
        "weight_fraction": interp.get("weight_fraction", []),
        "rwp": interp.get("rwp"),
        "score": interp.get("score"),
        "normalized_rwp": interp.get("normalized_rwp"),
        "normalized_score": interp.get("normalized_score"),
        "llm_likelihood": interp.get("LLM_interpretation_likelihood"),
        "balance_score": interp.get("balance_score"),
        "missing_peaks": interp.get("missing_peaks"),
        "extra_peaks": interp.get("extra_peaks"),
        "excess_bkg": interp.get("normalized_flag"),
        "signal_above_bkg": interp.get("signal_above_bkg"),
        "trust_score": round(trust, 3),
        "trustworthy": interp.get("trustworthy"),
    }


def compare_dara_vs_aif(json_data, sample_names, synthesis_df, project_key="combined"):
    html_all, html_diff = "", ""

    for sample_name in sample_names:
        if sample_name not in json_data:
            console.print(f"[red]⚠️ Sample '{sample_name}' not found.[/]")
            continue

        interps = json_data[sample_name]
        aif_id, aif_interp = max(interps.items(), key=lambda x: x[1].get("posterior_probability", 0))
        dara_id, dara_interp = min(interps.items(), key=lambda x: x[1].get("rwp", float("inf")))

        aif_summary = summarize(aif_interp)
        dara_summary = summarize(dara_interp)

        console.print(f"\n📌 Sample: [bold cyan]{sample_name}[/]")

        filtered = synthesis_df[synthesis_df["Name"].str.contains(rf'^{sample_name}$', na=False)]
        if filtered.empty:
            alt = sample_name.replace("_", "-") if "_" in sample_name else sample_name.replace("-", "_")
            filtered = synthesis_df[synthesis_df["Name"].str.contains(rf'^{alt}$', na=False)]

        if not filtered.empty:
            row = filtered.iloc[0]
            try:
                precursors = ast.literal_eval(row['Precursors'])
            except:
                precursors = [row['Precursors']]
            console.print(f"""
    [bold]Synthesis Description:[/bold]
    • Target: [cyan]{row['Target']}[/cyan]
    • Precursors: {", ".join(precursors)}
    • Temperature: {row['Temperature (C)']}°C
    • Duration: {row['Dwell Duration (h)']} h
    • Furnace: {row['Furnace']}
            """.strip())
            # === Show ranked interpretations ===
            console.print(f"\n[bold]All Interpretations Ranked by Unnormalized Posterior:[/bold]")

            sorted_interps = sorted(interps.items(), key=lambda x: x[1].get("posterior_probability", 0), reverse=True)

            table_all = Table(show_lines=True)
            table_all.add_column("Interp ID", style="bold")
            table_all.add_column("Posterior", style="bold")
            table_all.add_column("Phases")
            table_all.add_column("Weight Fraction")
            table_all.add_column("RWP")
            table_all.add_column("Norm. RWP")
            table_all.add_column("Norm. Score")
            table_all.add_column("LLM Likelihood")
            table_all.add_column("Balance")
            table_all.add_column("Trust Score")

            # Collect metric values to find best ones
            metric_keys = ["rwp", "normalized_rwp", "normalized_score", "LLM_interpretation_likelihood", "balance_score", "trust_score"]
            metric_values = {key: [] for key in metric_keys}

            # First pass to collect values
            interp_summaries = {}
            for interp_id, interp in sorted_interps:
                summary = summarize(interp)
                interp_summaries[interp_id] = summary
                for k in metric_keys:
                    val = summary.get(k if k != "LLM_interpretation_likelihood" else "llm_likelihood")
                    if val is not None:
                        metric_values[k].append(val)

            # Determine best per metric
            best_values = {
                "rwp" :min(metric_values["rwp"]),
                "normalized_rwp": max(metric_values["normalized_rwp"]),
                "normalized_score": max(metric_values["normalized_score"]),
                "LLM_interpretation_likelihood": max(metric_values["LLM_interpretation_likelihood"]),
                "balance_score": max(metric_values["balance_score"]),
                "trust_score": max(metric_values["trust_score"]),
            }

            # Now add rows
            # highlighted_phases = highlight_phase_differences(sorted_interps)
            highlighted_phases = assign_base_and_suffix_colors(sorted_interps)
            for interp_id, interp in sorted_interps:
                summary = interp_summaries[interp_id]
                phases = interp.get("phases", [])
                weights = interp.get("weight_fraction", [])
                weight_str = str([round(w, 2) for w in weights]) if weights else ""
                
                def highlight(k, val):
                    best = best_values[k]
                    if val == best:
                        return f"[bold green]{val:.4f}[/]"
                    return f"{val:.4f}"

                row = [
                    f"[bold green]{interp_id}[/]" if interp_id in [dara_id, aif_id] else interp_id,
                    f"{interp.get('posterior_probability', 0):.3f}",
                    highlighted_phases.get(interp_id, ", ".join(phases)),
                    # ", ".join(phases),
                    # highlighted_phases.get(interp_id, ", ".join(phases)),
                    weight_str,
                    highlight("rwp", summary.get("rwp", 0)),
                    highlight("normalized_rwp", summary.get("normalized_rwp", 0)),
                    highlight("normalized_score", summary.get("normalized_score", 0)),
                    highlight("LLM_interpretation_likelihood", summary.get("llm_likelihood", 0)),
                    highlight("balance_score", summary.get("balance_score", 0)),
                    highlight("trust_score", summary.get("trust_score", 0)),
                ]
                table_all.add_row(*row)

            console.print(table_all)

        def dedup(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        d_phases = dedup(dara_interp.get("phases", []))
        a_phases = dedup(aif_interp.get("phases", []))
        d_set, a_set = set(d_phases), set(a_phases)

        d_bases, a_bases = defaultdict(set), defaultdict(set)
        for p in d_phases: d_bases[split_phase_name(p)[0]].add(split_phase_name(p)[1])
        for p in a_phases: a_bases[split_phase_name(p)[0]].add(split_phase_name(p)[1])

        d_fmt, a_fmt = [], []
        for p in d_phases:
            base, suffix = split_phase_name(p)
            d_fmt.append(format_phase(base, suffix, "bold orange1") if p not in a_set or suffix not in a_bases.get(base, {}) else p)

        for p in a_phases:
            base, suffix = split_phase_name(p)
            a_fmt.append(format_phase(base, suffix, "bold deep_pink1") if p not in d_set or suffix not in d_bases.get(base, {}) else p)

        table = Table(show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column(f"DARA ({dara_id})", style="cyan")
        table.add_column(f"AIF ({aif_id})", style="cyan")

        table.add_row("Phases", ", ".join(d_fmt), ", ".join(a_fmt))
        def format_phase_weights(phases, weights):
            if not weights or len(phases) != len(weights):
                return ", ".join(phases)
            return str([round(w, 2) for w in weights]) if weights else ""#", ".join(f"{p} ({w:.2f})" for p, w in zip(phases, weights))

        d_weights = dara_summary.get("weight_fraction", [])
        a_weights = aif_summary.get("weight_fraction", [])

        d_weight_str = format_phase_weights(d_phases, d_weights)
        a_weight_str = format_phase_weights(a_phases, a_weights)

        table.add_row("Weight fraction", d_weight_str, a_weight_str)
        metrics = [
            ("rwp", True), ("score", False),
            ("normalized_rwp", False), ("normalized_score", False),
            ("llm_likelihood", False), ("balance_score", False),
            ("missing_peaks", True), ("extra_peaks", True),
            ("excess_bkg", True), ("signal_above_bkg", False),
            ("trust_score", False)
        ]

        for key, is_lower in metrics:
            dval, aval = dara_summary[key], aif_summary[key]
            d_str, a_str = color(dval, aval, is_lower)
            table.add_row(key, d_str, a_str)

        d_trust = "[bold green]True[/]" if dara_summary["trustworthy"] else "[bold red1]False[/]"
        a_trust = "[bold green]True[/]" if aif_summary["trustworthy"] else "[bold red1]False[/]"
        table.add_row("Trustworthy", d_trust, a_trust)

        console.print(table)
        html_sample = console.export_html(inline_styles=True, clear=True)
        html_all += html_sample

        if aif_id != dara_id:
            html_diff += html_sample

    with open(f"aif_dara_report_all_{project_key}.html", "w") as f:
        f.write(f"<html><head><meta charset='UTF-8'><style>span {{ font-weight: bold; }}</style></head><body>{html_all}</body></html>")

    with open(f"aif_dara_report_diff_selections_{project_key}.html", "w") as f:
        f.write(f"<html><head><meta charset='UTF-8'><style>span {{ font-weight: bold; }}</style></head><body>{html_diff}</body></html>")


def group_samples_by_project(samples):
    project_map = defaultdict(list)
    for sample in samples:
        prefix = sample.split("_")[0]
        project_map[prefix].append(sample)
    return project_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DARA vs AIF comparison on train/test sets.")
    parser.add_argument("split_json", help="Path to train.json or test.json")
    parser.add_argument("--out_prefix", default="combined", help="Prefix for report file names")
    args = parser.parse_args()

    json_data = json.load(open(args.split_json))
    sample_names = list(json_data.keys())
    sample_groups = group_samples_by_project(sample_names)

    merged_df = pd.DataFrame()
    merged_samples = []

    for proj, samples in sample_groups.items():
        csv = project_csv_map.get(proj.upper())
        if not csv:
            print(f"⚠️ No synthesis CSV for {proj}")
            continue
        try:
            df = pd.read_csv(csv)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            merged_samples.extend(samples)
        except Exception as e:
            print(f"❌ Failed to load {proj}: {e}")

    compare_dara_vs_aif(json_data, merged_samples, merged_df, project_key=args.out_prefix)
    console.save_text(f"aif_dara_report_{args.out_prefix}.txt")
    print(f"\n✅ Reports saved with prefix '{args.out_prefix}'")

#python tests/run_comparison_from_split.py test.json --out_prefix test
#python tests/run_comparison_from_split.py train.json --out_prefix train