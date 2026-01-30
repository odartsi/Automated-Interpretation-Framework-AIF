"""
Updated DARA vs AIF comparison script
• Adds two raw metrics (`signal_above_bkg_score`, `bkg_overshoot_score`).
• Adds a derived ratio metric (`signal_to_bkg_ratio`).
• Inlines fit‑snapshot PNGs for DARA and/or AIF (self‑contained HTML).
"""

import argparse
import ast
import base64
import json
import re
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
# def trust_flag(sig_score, bkg_score, sig_thresh=9000, bkg_thresh=1200):
#     if sig_score is None or bkg_score is None:
#         return "-"
#     if sig_score < sig_thresh:
#         return "[red1]low_signal[/]"
#     elif bkg_score > bkg_thresh:
#         return "[red1]high_bkg[/]"
#     else:
#         return "[bold green]✔[/]"
def trust_flag(sig_score, bkg_score, sig_thresh=9000, bkg_thresh=1200, ratio_thresh=15):
    if sig_score is None or bkg_score is None or bkg_score == 0:
        return "-"
    
    ratio = sig_score / bkg_score

    if sig_score < sig_thresh:
        return "[red1]low_signal[/]"
    elif bkg_score > bkg_thresh:
        return "[red1]high_bkg[/]"
    elif ratio <= ratio_thresh:
        return "[red1]low_S/B[/]"
    else:
        return "[bold green]✔[/]"
# ---------------------------------------------------------------------------
# Console (records prints so we can export them as HTML later)
# ---------------------------------------------------------------------------
console = Console(record=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PNG_ROOT = (Path(__file__).resolve().parent / ".." / "data" / "xrd_data" / "xrd_analysis").resolve()
PROJECT_CSV_MAP = {
    "TRI":   "../data/alab_synthesis_data/synthesis_TRI.csv",
    "ARR":   "../data/alab_synthesis_data/synthesis_ARR.csv",
    "MINES": "../data/alab_synthesis_data/synthesis_MINES.csv",
    "PG":    "../data/alab_synthesis_data/synthesis_PG_genome.csv",
}
# ──────────────────────────────────────────────────────────────
# Map project prefixes → synthesis-CSV paths
# Adjust paths if your directory layout differs
# ──────────────────────────────────────────────────────────────
project_csv_map: dict[str, str] = {
    "TRI":   "../data/alab_synthesis_data/synthesis_TRI.csv",
    "ARR":   "../data/alab_synthesis_data/synthesis_ARR.csv",
    "MINES": "../data/alab_synthesis_data/synthesis_MINES.csv",
    "PG":    "../data/alab_synthesis_data/synthesis_PG_genome.csv",
}

# ---------- value-comparison helper (green best / red worst) ------------------
def color(val1, val2, is_lower_better=False):
    """
    Return two Rich-formatted strings so the better value appears bold-green
    and the worse one red.  If the numbers are equal, both are plain text.
    Set is_lower_better=True for metrics where smaller is better (e.g. RWP).
    """
    if val1 is None or val2 is None:
        return str(val1), str(val2)

    is_int = isinstance(val1, int) and isinstance(val2, int)
    v1, v2 = round(float(val1), 4), round(float(val2), 4)

    # identical → no highlight
    if v1 == v2:
        return (
            f"{v1}" if is_int else f"{v1:.4f}",
            f"{v2}" if is_int else f"{v2:.4f}",
        )

    better = v1 < v2 if is_lower_better else v1 > v2
    f1 = f"{v1}" if is_int else f"{v1:.4f}"
    f2 = f"{v2}" if is_int else f"{v2:.4f}"

    return (
        f"[bold green]{f1}[/]" if better else f"[red1]{f1}[/]",
        f"[bold red1]{f2}[/]"  if better else f"[bold green]{f2}[/]",
    )
# ---------------------------------------------------------------------------
# PNG helpers
# ---------------------------------------------------------------------------

def find_png(project: str, target: str, sample: str, interp_id: str) -> str | None:
    """Return first PNG that matches our naming scheme, or **None**."""
    folder = PNG_ROOT / project / target / sample
    if not folder.is_dir():
        return None

    candidates = [
        folder / f"{sample}_{interp_id}.png",  # preferred naming
        folder / f"{interp_id}.png",            # fallback 1
    ] + list(folder.glob(f"*{interp_id}*.png"))  # wildcard fallback

    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def png_to_img_tag(png_path: str, width_px: int = 640) -> str:
    """Return an `<img>` tag with the PNG inlined as base‑64."""
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"<img src=\"data:image/png;base64,{b64}\" width=\"{width_px}px\">"

# ──────────────────────────────────────────────────────────────
#  HTML snapshot locator  (PNG/SVG ignored)
# ──────────────────────────────────────────────────────────────
def find_html(project: str, target: str, sample: str, interp_id: str) -> Path | None:
    folder = (PNG_ROOT / project / target / sample).resolve()
    if not folder.is_dir():
        return None
    for pat in (f"{sample}_{interp_id}.html",
                f"{interp_id}.html",
                f"*{interp_id}*.html"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None

from pathlib import Path

import os
from pathlib import Path

def html_to_iframe(path: Path, width_px: int = 900, height_px: int | None = None) -> str:
    """
    Embed an HTML snapshot via <iframe src='relative/path.html'>.
    Uses os.path.relpath so it never raises ValueError.
    """
    if height_px is None:
        height_px = int(width_px * 0.75)

    rel_path = os.path.relpath(path, start=Path.cwd())   # robust relative path
    rel_path = rel_path.replace("\\", "/")               # Windows → web slashes

    return (
        f'<iframe src="{rel_path}" '
        f'width="{width_px}px" height="{height_px}px" '
        f'style="border:1px solid #ccc;"></iframe>'
    )
# ---------------------------------------------------------------------------
# Phase helpers (colour assignment etc.)
# ---------------------------------------------------------------------------

PHASE_COLORS = cycle([
    "bright_blue", "cyan", "magenta", "green", "red", "orange1", "yellow", "purple",
    "chartreuse1", "turquoise2", "medium_purple", "light_salmon1", "deep_sky_blue1",
    "aquamarine1", "gold1", "orchid", "plum1", "light_pink1", "light_steel_blue",
    "medium_orchid", "tomato3",
])


def split_phase_name(phase: str):
    m = re.match(r"^(.+?)(_[\dA-Za-z]+)?$", phase)
    return (m.group(1), m.group(2) or "") if m else (phase, "")


def format_phase(base: str, suffix: str, color: str | None = None):
    return f"[{color}]{base}{suffix}[/{color}]" if color else f"{base}{suffix}"


def assign_base_and_suffix_colors(all_interps):
    base_palette = cycle([
        "cyan", "magenta", "green", "red", "orange1", "blue", "chartreuse1", "turquoise2",
        "medium_purple", "gold1", "orchid", "light_pink1", "light_steel_blue", "deep_sky_blue1",
    ])
    suffix_palette = cycle([
        "yellow", "orange1", "plum1", "light_salmon1", "tomato3", "light_green", "spring_green1",
    ])

    base2suffix = defaultdict(set)
    interp2phases = {}
    for iid, interp in all_interps:
        phases = interp.get("phases", [])
        interp2phases[iid] = phases
        for ph in phases:
            b, s = split_phase_name(ph)
            base2suffix[b].add(s)

    base2color = {b: next(base_palette) for b in sorted(base2suffix)}
    suffix2color = {
        b: {s: next(suffix_palette) for s in sorted(s_set)}
        for b, s_set in base2suffix.items() if len(s_set) > 1
    }

    out = {}
    for iid, phases in interp2phases.items():
        coloured = []
        for ph in phases:
            b, s = split_phase_name(ph)
            bc = base2color.get(b)
            sc = suffix2color.get(b, {}).get(s)
            if bc and sc:
                coloured.append(f"[{bc}]{b}[/{bc}][{sc}]{s}[/{sc}]")
            elif bc:
                coloured.append(f"[{bc}]{b}{s}[/{bc}]")
            else:
                coloured.append(ph)
        out[iid] = ", ".join(coloured)
    return out

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_trust_score(interp):
    llm = float(interp.get("LLM_interpretation_likelihood", 1.0))
    bal = float(interp.get("balance_score", 1.0))
    sig = float(interp.get("signal_above_bkg", 100.0))
    penalties = [max(0, min(1, 1 - llm)), max(0, min(1, 1 - bal)), max(0, min(1, (85 - sig) / 85))]
    return round(1 - sum(penalties) / 3, 3)


def compare_val(v1, v2, lower_better=False):
    if v1 is None or v2 is None:
        return str(v1), str(v2)
    v1r, v2r = round(float(v1), 4), round(float(v2), 4)
    if v1r == v2r:
        return f"{v1r}", f"{v2r}"
    better = v1r < v2r if lower_better else v1r > v2r
    fmt = lambda x, good: f"[bold green]{x}[/]" if good else f"[red1]{x}[/]"
    return fmt(v1r, better), fmt(v2r, not better)


def summarize(interp):
    sig_s = interp.get("signal_above_bkg_score")
    bkg_s = interp.get("bkg_overshoot_score")
    ratio = round(sig_s / bkg_s, 3) if sig_s and bkg_s else None
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
        "signal_above_bkg_score": sig_s,
        "bkg_overshoot_score": bkg_s,
        "signal_to_bkg_ratio": ratio,
    }

# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare_dara_vs_aif(json_data, sample_names, synthesis_df, project_key="combined"):
    html_all = html_diff = ""

    for sample in sample_names:
        if sample not in json_data:
            console.print(f"[red]⚠️ {sample} not in JSON[/]")
            continue

        interps = json_data[sample]
        aif_id, aif_int = max(interps.items(), key=lambda x: x[1].get("posterior_probability", 0))
        dara_id, dara_int = min(interps.items(), key=lambda x: x[1].get("rwp", float("inf")))
        aif_sum, dara_sum = summarize(aif_int), summarize(dara_int)

        console.print(f"\n📌 Sample [bold cyan]{sample}[/]")

        # synthesis row → may be absent
        row = synthesis_df.loc[synthesis_df["Name"].eq(sample)].copy()
        if row.empty:
            alt = sample.replace("_", "-") if "_" in sample else sample.replace("-", "_")
            row = synthesis_df.loc[synthesis_df["Name"].eq(alt)].copy()
        if not row.empty:
            row = row.iloc[0]
            try:
                prec = ast.literal_eval(row["Precursors"])
            except Exception:
                prec = [row["Precursors"]]
            console.print(f"""
[bold]Synthesis Description[/]:
• Target: [cyan]{row['Target']}[/]
• Precursors: {', '.join(prec)}
• Temperature: {row['Temperature (C)']}°C
• Duration: {row['Dwell Duration (h)']} h
• Furnace: {row['Furnace']}
""".rstrip())
        else:
            row = None

        # ---------------- images ----------------
        img_block = ""
        if row is not None:
            proj   = sample.split("_")[0]
            target = row["Target"]
            png_tags = []
            # DARA snapshot
            for interp in (dara_id, aif_id):            # always include both
                if (p := find_html(proj, target, sample, interp)):
                    png_tags.append(html_to_iframe(p, width_px=900))   # ← bigger width

            img_block = (
                "<div style='margin:10px 0;'>" + "&nbsp;".join(png_tags) + "</div>"
            ) if png_tags else ""
                    
            
            # if (p := find_png(proj, target, sample, dara_id)):
            #     png_tags.append(png_to_img_tag(p))
            # if aif_id != dara_id and (p := find_png(proj, target, sample, aif_id)):
            #     png_tags.append(png_to_img_tag(p))
            # if png_tags:
            #     img_block = "<div style='margin:8px 0;'>" + "&nbsp;".join(png_tags) + "</div>"
                # print a placeholder so section separation is clear in Rich
                # console.print("[bold]Fit snapshot(s) embedded in HTML[/]")

        # ---------------- ranking table ----------------
        ranked = sorted(interps.items(), key=lambda x: x[1].get("posterior_probability", 0), reverse=True)
        coloured_ph = assign_base_and_suffix_colors(ranked)

        best_rwp = min(i[1].get("rwp", float("inf")) for i in ranked)
        best_vals = {
            "normalized_rwp": max(i[1].get("normalized_rwp", 0) for i in ranked),
            "normalized_score": max(i[1].get("normalized_score", 0) for i in ranked),
            "ll": max(i[1].get("LLM_interpretation_likelihood", 0) for i in ranked),
            "bal": max(i[1].get("balance_score", 0) for i in ranked),
            "sig": max(i[1].get("signal_above_bkg_score", 0) for i in ranked),
            "bkg": min(i[1].get("bkg_overshoot_score", float("inf")) for i in ranked),
            "ratio": max((i[1].get("signal_above_bkg_score") or 0) / (i[1].get("bkg_overshoot_score") or 1) for i in ranked),
        }

        tbl_rank = Table(show_lines=True)
        tbl_rank.add_column("Interp")
        tbl_rank.add_column("Posterior")
        tbl_rank.add_column("Phases")
        tbl_rank.add_column("RWP", justify="right")
        tbl_rank.add_column("Norm RWP", justify="right")
        tbl_rank.add_column("Norm Score", justify="right")
        tbl_rank.add_column("LLM", justify="right")
        tbl_rank.add_column("Bal", justify="right")
        tbl_rank.add_column("Sig", justify="right")
        tbl_rank.add_column("Bkg", justify="right")
        tbl_rank.add_column("S/B", justify="right")
        tbl_rank.add_column("Trust fit", justify="right")

        def green_if_best(val, best, lower=False):
            if val is None:
                return "-"
            good = val == best if not lower else val == best
            return f"[bold green]{val:.2f}[/]" if good else f"{val:.2f}"

        for iid, intp in ranked:
            s = summarize(intp)
            tbl_rank.add_row(
                f"[bold green]{iid}[/]" if iid in (aif_id, dara_id) else iid,
                f"{intp.get('posterior_probability', 0):.3f}",
                coloured_ph.get(iid, s["phases"]),
                green_if_best(s["rwp"], best_rwp, lower=True),
                green_if_best(s["normalized_rwp"], best_vals["normalized_rwp"]),
                green_if_best(s["normalized_score"], best_vals["normalized_score"]),
                green_if_best(s["llm_likelihood"], best_vals["ll"]),
                green_if_best(s["balance_score"], best_vals["bal"]),
                green_if_best(s["signal_above_bkg_score"], best_vals["sig"]),
                green_if_best(s["bkg_overshoot_score"], best_vals["bkg"], lower=True),
                green_if_best(s["signal_to_bkg_ratio"], best_vals["ratio"]),
                trust_flag(s["signal_above_bkg_score"], s["bkg_overshoot_score"]),
            )
        console.print(tbl_rank)

        # ---------------- side‑by‑side DARA vs AIF ----------------
        tbl_cmp = Table(show_lines=True)
        tbl_cmp.add_column("Metric", style="bold")
        tbl_cmp.add_column(f"DARA ({dara_id})", style="cyan")
        tbl_cmp.add_column(f"AIF ({aif_id})", style="cyan")

        def dedup(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        def colour_list(lst, other_set, other_bases, miss_col):
            out = []
            for p in lst:
                b, sfx = split_phase_name(p)
                if p not in other_set or sfx not in other_bases.get(b, {}):
                    out.append(format_phase(b, sfx, miss_col))
                else:
                    out.append(p)
            return ", ".join(out)

        d_ph = dedup(dara_int.get("phases", []))
        a_ph = dedup(aif_int.get("phases", []))
        d_set, a_set = set(d_ph), set(a_ph)
        d_bases, a_bases = defaultdict(set), defaultdict(set)
        for p in d_ph: d_bases[split_phase_name(p)[0]].add(split_phase_name(p)[1])
        for p in a_ph: a_bases[split_phase_name(p)[0]].add(split_phase_name(p)[1])

        def colour_list(ph_list, other_set, other_bases, miss_col):
            out = []
            for p in ph_list:
                base, suf = split_phase_name(p)
                if p not in other_set or suf not in other_bases.get(base, {}):
                    out.append(format_phase(base, suf, miss_col))
                else:
                    out.append(p)
            return ", ".join(out)

        tbl_cmp = Table(show_lines=True)
        tbl_cmp.add_column("Metric", style="bold")
        tbl_cmp.add_column(f"DARA ({dara_id})", style="cyan")
        tbl_cmp.add_column(f"AIF ({aif_id})",  style="cyan")

        # phase row
        tbl_cmp.add_row(
            "Phases",
            colour_list(d_ph, a_set, a_bases, "bold orange1"),
            colour_list(a_ph, d_set, d_bases, "bold deep_pink1"),
        )


        # weight fraction row
        def wtxt(ph, wt):
            return str([round(w, 2) for w in wt]) if wt and len(ph) == len(wt) else ""
        tbl_cmp.add_row(
            "Weight fraction",
            wtxt(d_ph, dara_sum.get("weight_fraction")),
            wtxt(a_ph, aif_sum.get("weight_fraction")),
        )

        # metric comparison rows (includes the new S/B metrics)
        metrics = [
            ("rwp", True), ("score", False),
            ("normalized_rwp", False), ("normalized_score", False),
            ("llm_likelihood", False), ("balance_score", False),
            ("missing_peaks", True), ("extra_peaks", True),
            ("excess_bkg", True), ("signal_above_bkg", False),
            ("signal_above_bkg_score", False), ("bkg_overshoot_score", True),
            ("signal_to_bkg_ratio", False),
        ]
        for k, lower in metrics:
            d_str, a_str = color(dara_sum.get(k), aif_sum.get(k), is_lower_better=lower)
            tbl_cmp.add_row(k, d_str, a_str)

        console.print(tbl_cmp)

        # export html chunks
        # ── export html chunk ────────────────────────────────────────────────
        rich_chunk = console.export_html(inline_styles=True, clear=True)
        html_sample = img_block + rich_chunk
        html_all += html_sample
        if aif_id != dara_id:
            html_diff += html_sample

    # write combined reports
    with open(f"aif_dara_report_all_{project_key}.html", "w") as f:
        f.write(f"<html><head><meta charset='UTF-8'><style>span {{ font-weight:bold; }}</style></head><body>{html_all}</body></html>")
    with open(f"aif_dara_report_diff_selections_{project_key}.html", "w") as f:
        f.write(f"<html><head><meta charset='UTF-8'><style>span {{ font-weight:bold; }}</style></head><body>{html_diff}</body></html>")

# -----------------------------------------------------------------------------
#  Utility: group samples → project prefix
# -----------------------------------------------------------------------------

def group_samples_by_project(samples):
    groups: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        groups[s.split("_")[0]].append(s)
    return groups

# -----------------------------------------------------------------------------
#  CLI entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run DARA vs AIF comparison on train/test sets, including new S/B metrics.")
    p.add_argument("split_json", help="Path to train.json or test.json")
    p.add_argument("--out_prefix", default="combined", help="Prefix for report file names")
    args = p.parse_args()

    data = json.load(open(args.split_json))
    samp_names = list(data.keys())
    groups = group_samples_by_project(samp_names)

    merged_df = pd.DataFrame(); merged_samples = []
    for proj, samps in groups.items():
        csv = project_csv_map.get(proj.upper())
        if not csv:
            print(f"⚠️ No synthesis CSV for {proj}"); continue
        try:
            merged_df = pd.concat([merged_df, pd.read_csv(csv)], ignore_index=True)
            merged_samples.extend(samps)
        except Exception as e:
            print(f"❌ Failed to load {proj}: {e}")
    compare_dara_vs_aif(data, merged_samples, merged_df, project_key=args.out_prefix)
    # console.save_text(f"aif_dara_report_{args.out_prefix}.txt")
    print(f"\n✅ Reports saved with prefix '{args.out_prefix}'")

# python excess_bkg_and_signal_flag.py ../data/xrd_data/interpretations/interpretations_3scores.json --out_prefix bkg_signal_flag 
