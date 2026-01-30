import json
from pathlib import Path
import argparse
from rich.table import Table
from rich.console import Console


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def highlight_best(value, best_value, is_lower_better=False, fmt="{:.2f}", decimals=2):
    try:
        value = float(value)
        best_value = float(best_value)
    except (ValueError, TypeError):
        return str(value)

    value_rounded = round(value, decimals)
    best_rounded = round(best_value, decimals)

    if value_rounded == best_rounded:
        return f"[bold green]{fmt.format(value)}[/]"
    return fmt.format(value)


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


def rank_interpretations_by_aif(json_data, sample_names):
    console = Console()

    for sample_name in sample_names:
        if sample_name not in json_data:
            console.print(f"[red]⚠️ Sample '{sample_name}' not found.[/]")
            continue

        interpretations = json_data[sample_name]
        sorted_interps = sorted(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0),
            reverse=True
        )
    
        # Collect column-wise best values
        best_values = {
            "posterior_probability": max(x[1].get("posterior_probability", 0) for x in sorted_interps),
            "posterior_probability": max(x[1].get("posterior_probability", 0) for x in sorted_interps),
            "fit_quality": max(x[1].get("fit_quality", 0) for x in sorted_interps),
            "prior_probability": max(x[1].get("prior_probability", 0) for x in sorted_interps),
            "rwp": min(x[1].get("rwp", float("inf")) for x in sorted_interps),
            "score": max(x[1].get("score", 0) for x in sorted_interps),
            "search_result_rwp" : min(x[1].get("search_result_rwp", float("inf")) for x in sorted_interps),
            "search_result_score" :  max(x[1].get("search_result_score", 0) for x in sorted_interps),
            # "dara_score": max(x[1].get("dara_score", 0) for x in sorted_interps),
            "LLM_interpretation_likelihood": max(x[1].get("LLM_interpretation_likelihood", 0) for x in sorted_interps),
            "balance_score": max(x[1].get("balance_score", 0) for x in sorted_interps),
            "flag": min(x[1].get("flag", float("inf")) for x in sorted_interps),
            "normalized_flag": min(x[1].get("normalized_flag", float("inf")) for x in sorted_interps),
            "missing_peaks": min(x[1].get("missing_peaks", float("inf")) for x in sorted_interps),
            "extra_peaks": min(x[1].get("extra_peaks", float("inf")) for x in sorted_interps),
            "signal_above_bkg": max(x[1].get("signal_above_bkg", 0) for x in sorted_interps)
            
        }
        best_values["trust_score"] = max(compute_trust_score(x[1]) for x in sorted_interps)
        # best_values["trust_score"] = max(x[1].get("trust_score",0) for x in sorted_interps)
        
        top_id, top_interp = sorted_interps[0]
        console.print(f"\n📌 [bold cyan]Sample: {sample_name}[/bold cyan] — Top: [green]{top_id}[/green] with posterior={top_interp.get('posterior_probability', 0):.4f}")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="right")
        table.add_column("Interp ID", justify="left")
        table.add_column("Posterior", justify="right")
        table.add_column("Unorm Posterior", justify = "right")
        table.add_column("Phases", justify="left")
        table.add_column("Fit quality", justify="center")
        table.add_column("Prior", justify="right")
        table.add_column("RWP", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Search rwp", justify ="right")
        table.add_column("Search score", justify ="right")
        # table.add_column("Dara Score", justify="right")
        table.add_column("LLM Likelihood", justify="right")
        table.add_column("Balance Score", justify="right")
        # table.add_column("Flag", justify="right")
        table.add_column("Excess bkg (%)", justify="right")
        table.add_column("Missing peaks", justify="right")
        table.add_column("Extra peaks", justify="right")
        table.add_column("Signal above bkg (%)", justify="right")
        table.add_column("Trustworty", justify="right")
        table.add_column("Trust Score", justify="right")
        

        for i, (interp_id, data) in enumerate(sorted_interps, 1):
            trust_score = compute_trust_score(data)
            data["trust_score"] = trust_score
            table.add_row(
                str(i),
                interp_id,
                highlight_best(data.get("posterior_probability", 0), best_values["posterior_probability"]),
                highlight_best(data.get("posterior_probability", 0), best_values["posterior_probability"]),
                ", ".join(data.get("phases", [])),
                highlight_best(data.get("fit_quality", 0), best_values["fit_quality"], is_lower_better=False, fmt="{:.2f}"),
                highlight_best(data.get("prior_probability", 0), best_values["prior_probability"], is_lower_better=False, fmt="{:.2f}"),
                highlight_best(data.get("rwp", 0), best_values["rwp"], is_lower_better=True, fmt="{:.2f}"),
                highlight_best(data.get("score", 0), best_values["score"]),
                highlight_best(data.get("search_result_rwp", 0), best_values["search_result_rwp"], is_lower_better=True, fmt="{:.2f}"),
                highlight_best(data.get("search_result_score", 0), best_values["search_result_score"]),
                # highlight_best(data.get("dara_score", 0), best_values["dara_score"]),
                highlight_best(data.get("LLM_interpretation_likelihood", 0), best_values["LLM_interpretation_likelihood"]),
                highlight_best(data.get("balance_score", 0), best_values["balance_score"]),
                # highlight_best(data.get("flag", 0), best_values["flag"], is_lower_better=True, fmt="{:.2f}"),s
                highlight_best(data.get("normalized_flag", 0),best_values.get("normalized_flag", 0),is_lower_better=True,fmt="{:.2f}"),
                highlight_best(data.get("missing_peaks", 0),best_values.get("missing_peaks", 0),is_lower_better=True,fmt="{:.0f}"),
                highlight_best(data.get("extra_peaks", 0),best_values.get("extra_peaks", 0),is_lower_better=True,fmt="{:.0f}"),
                highlight_best(data.get("signal_above_bkg", 0), best_values.get("signal_above_bkg", 0), is_lower_better=False, fmt="{:.1f}"),
                "[bold green]True[/]" if data.get("trustworthy", False) else "[bold red]False[/]",
                highlight_best(data.get("trust_score", 0), best_values.get("trust_score", 0), is_lower_better=False, fmt="{:.2f}"),
                
            )
        console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank interpretations by posterior probability (AIF logic).")
    parser.add_argument("json_file", help="Path to JSON file with interpretation data.")
    parser.add_argument("samples", nargs="+", help="Sample names to analyze (e.g., TRI_91 TRI_82)")

    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
    else:
        json_data = load_json(json_path)
        rank_interpretations_by_aif(json_data, args.samples)
