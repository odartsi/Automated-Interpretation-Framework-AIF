# from rich.console import Console
# from rich.table import Table
# import json
# from pathlib import Path
# import argparse
# import pandas as pd
# import ast
# import re
# import numpy as np

# # --- pymatgen imports (per your preference) ---
# from pymatgen.core import Structure, Composition, Element
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# console = Console(record=True)

# # one-time flag for nm→Å console note
# _nm_note_printed = False

# # Root folder that contains the 'cifs/' directory; set via --cif-root
# _CIF_ROOT: Path | None = None
# MORPH_DEBUG = False
# # --- RRUFF formula normalization & pretty-printing (safe with decimals) ---
# _RR_CHARGE_RE = re.compile(r"\^[^^]*\^")                   # remove ^...^ charge markup
# _RR_SUB_RE    = re.compile(r"_(\d+(?:\.\d+)?)_")           # _3_ / _2.00_ -> sentinel with captured number
# _SENTINEL_FMT = "<<NUM:{num}>>"
# _SENTINEL_RE  = re.compile(r"<<NUM:(\d+(?:\.\d+)?)>>")
# # Optional cosmetic: Ca1 or Ca1.00 -> Ca (applied after rendering numbers)
# _ELEM_1_RE    = re.compile(r"([A-Z][a-z]?)1(?:\.0+)?\b")

# def read_measured_phases_clean(row) -> list[str]:
#     """
#     Read 'Measured_Phases_clean' from the CSV and split on comma/semicolon.
#     Returns a list of already-clean tokens (keep any SG suffixes as-is).
#     """
#     s = safe_get(row, "Measured_Phases_clean")
#     if not s:
#         return []
#     return [t.strip() for t in re.split(r"[;,]", str(s)) if t.strip()]

# def _norm_base(token: str) -> str:
#     base, _ = split_phase_name(token)
#     return normalize_rruff_formula_text(base)
# def resolve_cif_path(cif_rel: str | Path) -> Path | None:
#     """
#     Resolve a phase_cifs entry against _CIF_ROOT (which may be a 'pool/' dir or the parent of 'cifs/').
#     Tries several options:
#       1) _CIF_ROOT / cif_rel
#       2) strip leading 'cifs/' -> _CIF_ROOT / <rest>
#       3) basename only -> _CIF_ROOT / basename(cif_rel)
#       4) absolute path if cif_rel is already absolute
#     Returns Path or None if not found.
#     """
#     if not _CIF_ROOT or not cif_rel:
#         return None
#     rel = str(cif_rel).strip()
#     root = Path(_CIF_ROOT)

#     c1 = (root / rel)
#     if c1.exists():
#         return c1

#     if rel.lower().startswith("cifs/"):
#         tail = rel.split("/", 1)[1]
#         c2 = (root / tail)
#         if c2.exists():
#             return c2
#         c3 = (root / Path(rel).name)
#         if c3.exists():
#             return c3

#     # If they accidentally stored an absolute path in JSON
#     p = Path(rel)
#     if p.is_absolute() and p.exists():
#         return p

#     return None


# def _norm_phase_token(phase: str) -> str:
#     """Normalize an interpretation phase token like 'Cu3H4SO8_62' to 'Cu3H4SO8_62' with cleaned formula."""
#     base, suffix = split_phase_name(phase)
#     return normalize_rruff_formula_text(base) + (suffix or "")
# def color_morphologies(morph_str: str, csv_cs: str | None) -> str:
#     """
#     Color each morphology token that equals the CSV crystal system (case-insensitive).
#     Keeps comma-separated formatting. Leaves '-' and '—' untouched.
#     """
#     if not morph_str or morph_str == "—" or not csv_cs:
#         return morph_str or "—"
#     target = csv_cs.strip().lower()
#     parts = [t.strip() for t in morph_str.split(",")]
#     colored = []
#     for t in parts:
#         if not t or t in {"-", "—"}:
#             colored.append(t or "—")
#         elif t.lower() == target:
#             colored.append(f"[bold orange1]{t}[/]")
#         else:
#             colored.append(t)
#     return ", ".join(colored)

# def color_crystal_system(cs_val: str | None, morph_str: str | None) -> str:
#     """
#     Highlight the Crystal_System cell if it appears in the morphology list (case-insensitive).
#     """
#     if not cs_val or not morph_str or cs_val in {"-", "—"}:
#         return cs_val or "—"
#     morphs = [t.strip().lower() for t in morph_str.split(",")]
#     if cs_val.strip().lower() in morphs:
#         return f"[bold orange1]{cs_val}[/]"
#     return cs_val

# def _normalize_num_str(numstr: str) -> str:
#     """Render 2.00 -> 2, 2.50 -> 2.5, keep minimal clean form."""
#     try:
#         x = float(numstr)
#     except Exception:
#         return numstr
#     if abs(x - round(x)) < 1e-6:
#         return str(int(round(x)))
#     s = f"{x}".rstrip("0").rstrip(".")
#     return s if s else "0"


# def normalize_rruff_formula_text(s: str) -> str:
#     """
#     Convert RRUFF-styled 'Ca_1.00_CO_3_' -> 'CaCO3', 'As_2.00_O_3_' -> 'As2O3'.
#     Steps:
#       - drop ^charges^
#       - turn _n_ / _n.m_ into sentinels, then remove underscores between symbols
#       - restore numbers with smart formatting (2.00->2)
#       - drop explicit '1' (Ca1 -> Ca)
#       - strip stray spaces/boxes; keep only letters/digits after restoration
#     """
#     if not s:
#         return ""
#     t = str(s)
#     t = _RR_CHARGE_RE.sub("", t)
#     # replace _num_ with sentinels so the dot survives through cleanup
#     t = _RR_SUB_RE.sub(lambda m: _SENTINEL_FMT.format(num=m.group(1)), t)
#     # remove leftover underscores and [box]
#     t = t.replace("_", "").replace("[box]", "").replace(" ", "")
#     # restore numbers from sentinels
#     t = _SENTINEL_RE.sub(lambda m: _normalize_num_str(m.group(1)), t)
#     # Ca1 / Ca1.00 -> Ca
#     _ELEM_1_RE = re.compile(r"([A-Z][a-z]?)(?:1(?:\.0+)?)((?=[A-Z]|$)|(?!\d))")
#     t = _ELEM_1_RE.sub(r"\1", t)
#     # final cleanup: keep only letters/digits
#     t = re.sub(r"[^A-Za-z0-9]", "", t)
#     return t


# def normalize_phase_list_field(field) -> list[str]:
#     """Split a CSV/semicolon list of phases and normalize each token."""
#     if not field:
#         return []
#     items = re.split(r"[;,]", str(field))
#     out = []
#     for t in items:
#         t = t.strip()
#         if not t:
#             continue
#         out.append(normalize_rruff_formula_text(t))
#     return out


# def desc_cell_from_row(row) -> dict:
#     """Read a,b,c,alpha,beta,gamma from RRUFF CSV (floats)."""
#     keys = ("a", "b", "c", "alpha", "beta", "gamma")
#     out = {}
#     for k in keys:
#         v = safe_get(row, k)
#         try:
#             out[k] = float(v) if (v not in ("", None)) else None
#         except Exception:
#             out[k] = None
#     return out


# def round_eq(x, y, nd=2) -> bool:
#     if x is None or y is None:
#         return False
#     try:
#         return round(float(x), nd) == round(float(y), nd)
#     except Exception:
#         return False


# def fmt_match(value, desc_value, nd_show=4, nd_match=2, color="bold orange1"):
#     s = fmt_num(value, nd_show)
#     return f"[{color}]{s}[/]" if round_eq(value, desc_value, nd_match) else s


# def parse_sgn_set(sgn_field) -> set[int]:
#     return {int(tok) for tok in re.findall(r"\d+", str(sgn_field) if sgn_field is not None else "")}


# def parse_measured_phase_bases(measured_phases_field) -> set[str]:
#     """Use normalized measured phases for matching bases."""
#     bases = set()
#     for ph in normalize_phase_list_field(measured_phases_field):
#         base, _ = split_phase_name(ph)
#         bases.add(base)
#     return bases


# # --- matching + formatting helpers (no unit logic here) ---
# def _to_float_or_none(x):
#     try:
#         return float(x)
#     except Exception:
#         return None


# def _coerce_number(x):
#     """Return a float from value/tuple/list/string; else None."""
#     if x is None:
#         return None
#     # value with uncertainty like (0.5779, 0.00011)
#     if isinstance(x, (list, tuple)) and x:
#         x = x[0]
#     # strings like "0.5779" or "(0.5779, 0.00011)"
#     if isinstance(x, str):
#         x = x.strip()
#         if x.startswith("(") and x.endswith(")"):
#             try:
#                 t = ast.literal_eval(x)
#                 if isinstance(t, (list, tuple)) and t:
#                     x = t[0]
#             except Exception:
#                 pass
#         # fall-through: try float
#     try:
#         return float(x)
#     except Exception:
#         return None


# def _normalize_unit(u: str | None) -> str | None:
#     if not u:
#         return None
#     u = str(u).strip().lower()
#     if u in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
#         return "nm"
#     if u in {"å", "ang", "angstrom", "angstroms", "a", "angström"}:
#         return "ang"
#     return u  # unknown token; handled by heuristic later


# def _nm_to_ang_if_needed(x: float, explicit_unit: str | None) -> tuple[float | None, bool]:
#     """
#     Returns (value_in_angstrom, did_convert_nm_to_angstrom).
#     If explicit_unit=='nm' → convert; if 'ang' → no convert; else apply heuristic.
#     """
#     if x is None:
#         return (None, False)
#     unit = _normalize_unit(explicit_unit)
#     if unit == "nm":
#         return (x * 10.0, True)
#     if unit == "ang":
#         return (x, False)
#     # Heuristic if no/unknown unit: 0.05–3.0 likely nm (→ 0.5–30 Å)
#     if 0.05 <= x <= 3.0:
#         return (x * 10.0, True)
#     return (x, False)


# def get_cell_params_from_interp(interp: dict) -> dict:
#     """
#     Extracts a,b,c,alpha,beta,gamma, normalizes to Å for a,b,c.
#     Accepts either:
#       - interp["cell_parameters"] = {a,b,c,alpha,beta,gamma, unit?}
#       - flat keys on interp: a,b,c,alpha,beta,gamma, unit
#       - values may be floats, strings, or (value, sigma) tuples
#     """
#     global _nm_note_printed

#     # prefer nested dict if present
#     cp = interp.get("cell_parameters")
#     if isinstance(cp, dict) and cp:
#         src = cp
#         unit_field = cp.get("unit") or interp.get("unit")
#     else:
#         # fall back to flat keys on the interpretation
#         src = interp
#         unit_field = interp.get("unit")

#     # coerce raw values
#     raw = {k: _coerce_number(src.get(k)) for k in ("a", "b", "c", "alpha", "beta", "gamma")}

#     # convert a,b,c if needed
#     converted_any = False
#     out = {}
#     for k in ("a", "b", "c"):
#         v_ang, did = _nm_to_ang_if_needed(raw.get(k), unit_field)
#         out[k] = v_ang
#         converted_any = converted_any or did

#     # angles: keep as-is (no unit conversion)
#     for k in ("alpha", "beta", "gamma"):
#         out[k] = raw.get(k)

#     # one-time console note if we converted
#     if converted_any and not _nm_note_printed:
#         try:
#             console.print("[dim]Note: interpretation cell a,b,c were in nm; converted to Å for display.[/dim]")
#         except Exception:
#             pass
#         _nm_note_printed = True

#     return out


# def fmt_num(val, nd=4):
#     """Format a float or int for display; '—' if None."""
#     if val is None:
#         return "—"
#     if isinstance(val, int) and not isinstance(val, bool):
#         return f"{val}"
#     try:
#         f = float(val)
#         # don't show trailing zeros for integers-in-float form, but keep 4 dp otherwise
#         return f"{f:.{nd}f}".rstrip("0").rstrip(".")
#     except Exception:
#         return str(val)


# # ---- Trust decision with explicit reasons (like your compare_disagreements.py) ----
# TRUST_THRESHOLDS = {
#     "llm_max_bad": 0.4,     # False if LLM ≤ 0.4
#     "signal_min": 9000,     # False if signal_above_bkg_score < 9000  (when present)
#     "bkg_max": 1200,        # False if bkg_overshoot_score > 1200     (when present)
#     "ratio_min": 15,        # False if (signal/bkg) < 15              (when both present)
#     "balance_min": 0.6,     # False if balance_score < 0.6
# }


# def target_with_cs_sg(row, target_str: str, highlight: bool = False) -> str:
#     """Append (crystal system; SG …) after Target if available; optionally highlight Target."""
#     cs = safe_get(row, "Crystal_System")
#     sg = safe_get(row, "Space_Group_Number")
#     parts = []
#     if cs:
#         parts.append(cs)
#     if sg:
#         parts.append(f"SG {sg}")
#     suffix = f" [dim]({'; '.join(parts)})[/]" if parts else ""
#     core = f"[bold orange1]{target_str}[/]" if highlight else f"[cyan]{target_str}[/cyan]"
#     return f"• Target: {core}{suffix}"


# def trust_reasons(interp: dict) -> tuple[bool, list[str]]:
#     # pull values with safe defaults
#     llm  = float(interp.get("LLM_interpretation_likelihood", 1.0) or 1.0)
#     bal  = float(interp.get("balance_score", 1.0) or 1.0)
#     sabs = float(interp.get("signal_above_bkg_score", 0.0) or 0.0)
#     bkg  = float(interp.get("bkg_overshoot_score", 0.0) or 0.0)

#     reasons = []
#     if llm <= TRUST_THRESHOLDS["llm_max_bad"]:
#         reasons.append(f"low LLM (≤{TRUST_THRESHOLDS['llm_max_bad']})")
#     if sabs and sabs < TRUST_THRESHOLDS["signal_min"]:
#         reasons.append(f"low signal_above_bkg_score (<{TRUST_THRESHOLDS['signal_min']})")
#     if bkg and bkg > TRUST_THRESHOLDS["bkg_max"]:
#         reasons.append(f"high bkg_overshoot_score (>{TRUST_THRESHOLDS['bkg_max']})")
#     if sabs and bkg and (sabs / bkg) < TRUST_THRESHOLDS["ratio_min"]:
#         reasons.append(f"low signal/bkg ratio (<{TRUST_THRESHOLDS['ratio_min']})")
#     if bal < TRUST_THRESHOLDS["balance_min"]:
#         reasons.append(f"low balance score (<{TRUST_THRESHOLDS['balance_min']})")

#     return (len(reasons) == 0), reasons


# def format_across(values, is_lower_better=False):
#     """
#     Color best/worst across multiple columns.
#     values: list of numbers or None
#     Returns: list[str] with bold green for best and bold red1 for worst.
#     Ties: multiple best/worst get colored.
#     """
#     # keep track of which are numbers
#     numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
#     if not numeric:
#         return ["—" if v is None else str(v) for v in values]

#     # round for comparison/printing (but keep int formatting for ints)
#     rounded = {}
#     as_int = {}
#     for i, v in numeric:
#         as_int[i] = isinstance(v, int) and not isinstance(v, bool)
#         rounded[i] = round(float(v), 4)

#     # find best/worst indices
#     comp_vals = {i: rounded[i] for i, _ in numeric}
#     if is_lower_better:
#         best_val = min(comp_vals.values())
#         worst_val = max(comp_vals.values())
#     else:
#         best_val = max(comp_vals.values())
#         worst_val = min(comp_vals.values())

#     best_idxs = {i for i, rv in comp_vals.items() if rv == best_val}
#     worst_idxs = {i for i, rv in comp_vals.items() if rv == worst_val}

#     # build strings
#     out = []
#     for idx, v in enumerate(values):
#         if v is None:
#             out.append("—")
#             continue
#         if idx in rounded:
#             s = f"{int(v)}" if as_int[idx] else f"{rounded[idx]:.4f}"
#             if idx in best_idxs and idx in worst_idxs:
#                 # all equal: no color
#                 out.append(s)
#             elif idx in best_idxs:
#                 out.append(f"[bold green]{s}[/]")
#             elif idx in worst_idxs:
#                 out.append(f"[bold red1]{s}[/]")
#             else:
#                 out.append(s)
#         else:
#             out.append(str(v))
#     return out


# # ----------------- helpers -----------------
# def split_phase_name(phase):
#     """Split into base and space group: 'V6013_69' → ('V6013', '_69')"""
#     match = re.match(r"^(.+?)(_\d+|_[A-Za-z]+)?$", phase)
#     if match:
#         base = match.group(1)
#         suffix = match.group(2) or ""
#         return base, suffix
#     return phase, ""


# def format_phase(base, suffix, color=None):
#     if color:
#         return f"{base}[{color}]{suffix}[/]"
#     else:
#         return f"{base}{suffix}"


# def load_json(path):
#     with open(path, "r") as f:
#         return json.load(f)


# def compare_param_color(val, desc_val, tol_orange=0.1, tol_blue=0.3):
#     """
#     Compare interpretation vs description numeric value:
#       - |Δ| <= tol_orange → orange
#       - tol_orange < |Δ| <= tol_blue → blue
#       - otherwise plain
#     Returns formatted string with color tag.
#     """
#     if val is None:
#         return "—"
#     s = fmt_num(val)
#     if desc_val is None:
#         return s
#     try:
#         diff = abs(float(val) - float(desc_val))
#     except Exception:
#         return s
#     if diff <= tol_orange:
#         return f"[bold orange1]{s}[/]"
#     elif diff <= tol_blue:
#         return f"[bold blue]{s}[/]"
#     return s


# def format_sg_list_with_highlight(sg_field, highlight_set: set[int], color="bold orange1") -> str:
#     """
#     Render 'Space Group #:' numbers, coloring any that appear in highlight_set.
#     Accepts CSV forms like '20;59;62'.
#     """
#     if sg_field in (None, ""):
#         return ""
#     tokens = re.findall(r"\d+", str(sg_field))
#     if not tokens:
#         return str(sg_field)
#     parts = []
#     for t in tokens:
#         try:
#             n = int(t)
#             parts.append(f"[{color}]{n}[/]" if n in highlight_set else str(n))
#         except Exception:
#             parts.append(str(t))
#     return ";".join(parts)


# def compute_trust_score(interp: dict) -> float:
#     """
#     Smooth trust score (0..1) using six soft criteria:
#       - LLM_interpretation_likelihood  (≥ 0.4)
#       - signal_above_background score  (≥ 9000)
#       - background overshoot score     (≤ 1200)
#       - signal/overshoot ratio         (≥ 15)
#       - balance_score                  (≥ 0.6)
#       - normalized_score (peak match)  (≥ 0.6)

#     Trust score = 1 - average of per-criterion penalties (each clipped to [0,1]).
#     """
#     try:
#         llm       = float(interp.get("LLM_interpretation_likelihood", 1.0))
#         signal    = float(interp.get("signal_above_bkg_score", 10000.0))
#         overshoot = float(interp.get("bkg_overshoot_score", 0.0))
#         balance   = float(interp.get("balance_score", 1.0))
#         pmatch    = float(interp.get("normalized_score", 1.0))
#     except (TypeError, ValueError):
#         return 0.0

#     # helpers
#     clip01 = lambda x: max(0.0, min(1.0, x))

#     # 1) LLM (ideal ≥ 0.4)
#     penalty_llm = clip01((0.4 - llm) / 0.4) if llm < 0.4 else 0.0

#     # 2) Signal above background score (ideal ≥ 9000)
#     penalty_signal = clip01((9000.0 - signal) / 9000.0) if signal < 9000.0 else 0.0

#     # 3) Background overshoot score (ideal ≤ 1200)
#     penalty_overshoot = clip01((overshoot - 1200.0) / 1200.0) if overshoot > 1200.0 else 0.0

#     # 4) Signal / overshoot ratio (ideal ≥ 15)
#     if overshoot > 0.0:
#         ratio = signal / overshoot
#         penalty_ratio = clip01((15.0 - ratio) / 15.0) if ratio < 15.0 else 0.0
#     else:
#         penalty_ratio = 0.0  # no penalty if no overshoot

#     # 5) Balance score (ideal ≥ 0.6)
#     penalty_balance = clip01((0.6 - balance) / 0.6) if balance < 0.6 else 0.0

#     # 6) Peak match score (ideal ≥ 0.6)
#     penalty_match = clip01((0.6 - pmatch) / 0.6) if pmatch < 0.6 else 0.0

#     total_penalty = (
#         penalty_llm
#         + penalty_signal
#         + penalty_overshoot
#         + penalty_ratio
#         + penalty_balance
#         + penalty_match
#     ) / 6.0

#     return round(max(0.0, 1.0 - total_penalty), 3)


# def color(val1, val2, is_lower_better=False):
#     """Return colored strings for values based on comparison."""
#     if val1 is None or val2 is None:
#         return str(val1), str(val2)

#     is_integer = isinstance(val1, int) and isinstance(val2, int)
#     v1 = round(val1, 4)
#     v2 = round(val2, 4)

#     if v1 == v2:
#         return (f"{val1}" if is_integer else f"{v1:.4f}",
#                 f"{val2}" if is_integer else f"{v2:.4f}")

#     better = v1 < v2 if is_lower_better else v1 > v2
#     v1s = f"{val1}" if is_integer else f"{v1:.4f}"
#     v2s = f"{val2}" if is_integer else f"{v2:.4f}"
#     return (
#         f"[bold green]{v1s}[/]" if better else f"[red1]{v1s}[/]",
#         f"[bold red1]{v2s}[/]" if better else f"[bold green]{v2s}[/]"
#     )


# # =======================
# # CIF → morphology helpers
# # =======================
# # =======================
# # CIF → morphology helpers (with SG# fallback + diagnostics)
# # =======================

# # Map space group number to crystal system
# def _sg_to_crystal_system(n: int) -> str | None:
#     if 1 <= n <= 2:
#         return "triclinic"
#     if 3 <= n <= 15:
#         return "monoclinic"
#     if 16 <= n <= 74:
#         return "orthorhombic"
#     if 75 <= n <= 142:
#         return "tetragonal"
#     if 143 <= n <= 167:
#         return "trigonal"
#     if 168 <= n <= 194:
#         return "hexagonal"
#     if 195 <= n <= 230:
#         return "cubic"
#     return None

# def _log_morph_issue(msg: str):
#     try:
#         if MORPH_DEBUG:
#             console.print(f"[dim]{msg}[/dim]")
#     except Exception:
#         pass

# def get_crystal_system_from_cif(cif_path: Path) -> str | None:
#     """
#     Return the crystal system (e.g., 'cubic', 'tetragonal') from a CIF file.
#     Returns None if the file can't be read or spglib is unavailable.
#     """
#     try:
#         # Quick check for file existence
#         if not cif_path.exists():
#             _log_morph_issue(f"morphology: CIF not found: {cif_path}")
#             return None

#         structure = Structure.from_file(str(cif_path))
#         try:
#             analyzer = SpacegroupAnalyzer(structure)
#             cs = analyzer.get_crystal_system()
#             return str(cs).lower() if cs else None
#         except Exception as e:
#             _log_morph_issue(f"morphology: SpacegroupAnalyzer failed (spglib missing?): {e}")
#             return None
#     except Exception as e:
#         _log_morph_issue(f"morphology: failed to read CIF ({cif_path}): {e}")
#         return None

# def _infer_cs_from_phase_name(phase: str) -> str | None:
#     # Expect suffix like _62
#     m = re.search(r"_(\d+)\b", phase)
#     if not m:
#         return None
#     try:
#         n = int(m.group(1))
#         return _sg_to_crystal_system(n)
#     except Exception:
#         return None

# # def get_phase_morphologies(interp: dict) -> list[str]:
# #     """
# #     For each phase in interpretation, try CIF → crystal system;
# #     if that fails, fall back to space-group number parsed from phase name;
# #     if both fail → '-'.
# #     Order follows interp['phase_cifs'] / interp['phases'].
# #     """
# #     out = []
# #     # Align CIFs to phases: if #cifs != #phases, we still try best-effort
# #     phases = list(interp.get("phases") or [])
# #     phase_cifs = interp.get("phase_cifs") or []
# #     if isinstance(phase_cifs, str):
# #         phase_cifs = [phase_cifs]

# #     # pad/truncate CIF list to length of phases
# #     cifs_aligned = list(phase_cifs) + [None] * max(0, len(phases) - len(phase_cifs))
# #     cifs_aligned = cifs_aligned[:len(phases)] if phases else phase_cifs

# #     if not phases:
# #         # no phases; fallback to just CIFs list
# #         for cif_rel in phase_cifs:
# #             if cif_rel and _CIF_ROOT:
# #                 cs = get_crystal_system_from_cif((Path(_CIF_ROOT) / cif_rel).resolve())
# #                 out.append(cs if cs else "-")
# #             else:
# #                 out.append("-")
# #         return out

# #     for phase, cif_rel in zip(phases, cifs_aligned):
# #         cs = None
# #         if cif_rel and _CIF_ROOT:
# #             cif_path = (Path(_CIF_ROOT) / cif_rel).resolve()
# #             cs = get_crystal_system_from_cif(cif_path)

# #         # Fallback: infer from SG number in the phase token
# #         if cs is None:
# #             cs = _infer_cs_from_phase_name(phase)
# #             if cs is None:
# #                 _log_morph_issue(f"morphology: no CIF/SG fallback for '{phase}' (cif={cif_rel})")
# #                 out.append("-")
# #             else:
# #                 out.append(cs)
# #         else:
# #             out.append(cs)
# #     return out

# def get_phase_morphologies(interp: dict) -> list[str]:
#     """
#     For each phase in interpretation, return crystal system ONLY if a CIF can be resolved.
#     Otherwise '-'. (No SG/system fallback.)
#     """
#     out = []
#     phases = list(interp.get("phases") or [])
#     phase_cifs = interp.get("phase_cifs") or []
#     if isinstance(phase_cifs, str):
#         phase_cifs = [phase_cifs]

#     # align CIFs to phases (best-effort)
#     cifs_aligned = list(phase_cifs) + [None] * max(0, len(phases) - len(phase_cifs))
#     cifs_aligned = cifs_aligned[:len(phases)] if phases else []

#     for cif_rel in cifs_aligned:
#         path = resolve_cif_path(cif_rel) if cif_rel else None
#         if path:
#             cs = get_crystal_system_from_cif(path)
#             out.append(cs if cs else "-")
#         else:
#             out.append("-")
#     return out
# # =======================
# # Composition / L1 helpers
# # =======================

# # Build elements list in increasing Z (compatible across pymatgen versions)
# try:
#     _ELEMENTS_BY_Z = sorted(list(Element), key=lambda e: e.Z)
# except Exception:
#     _ELEMENTS_BY_Z = [Element.from_Z(z) for z in range(1, 119)]

# ALL_ELEMENTS = [e.symbol for e in _ELEMENTS_BY_Z]
# SYM_TO_IDX = {sym: i for i, sym in enumerate(ALL_ELEMENTS)}


# def comp_to_vec(comp: str | Composition) -> np.ndarray:
#     """
#     Map a composition to a length-118 vector of atomic fractions (sums to 1 over present elements).
#     Any element not present gets 0. Uses atomic (not weight) fractions.
#     """
#     if not isinstance(comp, Composition):
#         comp = Composition(comp)
#     frac = comp.fractional_composition.get_el_amt_dict()
#     v = np.zeros(len(ALL_ELEMENTS), dtype=float)
#     for el_sym, amt in frac.items():
#         idx = SYM_TO_IDX.get(el_sym)
#         if idx is not None:
#             v[idx] = float(amt)
#     # Normalize to sum 1 over present elements (pymatgen should already enforce)
#     s = v.sum()
#     if s > 0:
#         v /= s
#     return v


# def l1_distance(comp_a: str | Composition, comp_b: str | Composition) -> float:
#     """
#     L1 (Manhattan) distance between two composition vectors.
#     """
#     va = comp_to_vec(comp_a)
#     vb = comp_to_vec(comp_b)
#     return float(np.abs(va - vb).sum())


# def mixture_vector_from_interpretation(interp: dict) -> np.ndarray | None:
#     """
#     Combine phases by weight_fraction to a single composition vector (sums to 1).
#     We use *atomic* fractions, not mass fractions.
#     If missing weights, assume equal weights.
#     If no phases, return None.
#     """
#     phases = interp.get("phases") or []
#     if not phases:
#         return None

#     # weights
#     wf = interp.get("weight_fraction")
#     if isinstance(wf, (int, float)):
#         wf = [wf]
#     if not (isinstance(wf, list) and len(wf) == len(phases)):
#         wf = [1.0] * len(phases)

#     # Normalize weights
#     wsum = float(sum(wf)) if wf else 0.0
#     if wsum <= 0:
#         wf = [1.0] * len(phases)
#         wsum = float(len(phases))
#     wf = [float(x) / wsum for x in wf]

#     # Phase formulas: strip SG suffix: 'Cu3H4SO8_62' → 'Cu3H4SO8'
#     vec = np.zeros(len(ALL_ELEMENTS), dtype=float)
#     for w, p in zip(wf, phases):
#         base, _ = split_phase_name(p)
#         try:
#             pv = comp_to_vec(base)
#             vec += w * pv
#         except Exception:
#             # If the phase can't be parsed, skip it
#             continue

#     # Re-normalize to sum 1 if small drift
#     s = vec.sum()
#     if s > 0:
#         vec /= s
#     return vec


# def per_phase_l1_lists(interp: dict, target_formula: str | None, measured_formulas: list[str]) -> tuple[list[float] | None, list[float] | None]:
#     """
#     Returns (per_phase_l1_to_target, per_phase_l1_to_measured_min).
#     Each is a list aligned with interp['phases'] order. If target/measured not available -> None.
#     """
#     phases = interp.get("phases") or []
#     if not phases:
#         return (None, None)

#     # Base formulas for each phase (strip SG suffix)
#     bases = [split_phase_name(p)[0] for p in phases]

#     # Precompute vectors
#     try:
#         target_vec = comp_to_vec(target_formula) if target_formula else None
#     except Exception:
#         target_vec = None

#     measured_vecs = []
#     for mf in measured_formulas or []:
#         try:
#             measured_vecs.append(comp_to_vec(mf))
#         except Exception:
#             pass

#     # Per-phase to Target
#     l1p_target = None
#     if target_vec is not None:
#         l1p_target = []
#         for b in bases:
#             try:
#                 l1p_target.append(float(np.abs(comp_to_vec(b) - target_vec).sum()))
#             except Exception:
#                 l1p_target.append(None)

#     # Per-phase to closest Measured
#     l1p_meas = None
#     if measured_vecs:
#         l1p_meas = []
#         for b in bases:
#             try:
#                 v = comp_to_vec(b)
#                 dmin = min(float(np.abs(v - mv).sum()) for mv in measured_vecs) if measured_vecs else None
#                 l1p_meas.append(dmin)
#             except Exception:
#                 l1p_meas.append(None)

#     return (l1p_target, l1p_meas)


# # =======================
# # Summarize (extended)
# # =======================

# def summarize(interp):
#     ts = interp.get("trust_score")
#     if ts is None:
#         ts = compute_trust_score(interp)
#     trusty, reasons = trust_reasons(interp)

#     # cell params
#     cp = get_cell_params_from_interp(interp)

#     # morphology per phase from CIFs
#     morph_list = get_phase_morphologies(interp)  # e.g., ['cubic', '-'] aligned with phase_cifs
#     morph_str = ", ".join(morph_list) if morph_list else "—"

#     # L1 distances (mixture vs target, mixture vs measured[min])
#     mix_vec = mixture_vector_from_interpretation(interp)

#     l1_to_target = None
#     l1_to_measured = None

#     # Target & measured formulas provided by caller via injected fields
#     target_formula = interp.get("_target_norm")  # normalized string (e.g., 'Cu3H4SO8')
#     measured_formulas = interp.get("_measured_list") or []  # list of normalized strings

#     try:
#         if mix_vec is not None and target_formula:
#             l1_to_target = float(np.abs(mix_vec - comp_to_vec(target_formula)).sum())
#     except Exception:
#         l1_to_target = None

#     try:
#         if mix_vec is not None and measured_formulas:
#             dists = []
#             for mf in measured_formulas:
#                 try:
#                     dists.append(float(np.abs(mix_vec - comp_to_vec(mf)).sum()))
#                 except Exception:
#                     pass
#             if dists:
#                 l1_to_measured = float(min(dists))
#     except Exception:
#         l1_to_measured = None

#     # Per-phase L1 lists (comma-separated strings)
#     l1p_target_list, l1p_measured_list = per_phase_l1_lists(
#         interp, target_formula=target_formula, measured_formulas=measured_formulas
#     )

#     def _fmt_list(xs):
#         if not xs:
#             return "—"
#         parts = []
#         for x in xs:
#             parts.append("—" if x is None else f"{x:.4f}")
#         return ", ".join(parts)

#     return {
#         "phases": ", ".join(interp.get("phases", [])),
#         "rwp": interp.get("rwp"),
#         "search_result_rwp": interp.get("search_result_rwp"),
#         "score": interp.get("score"),
#         "search_result_score": interp.get("search_result_score"),
#         "llm_likelihood": interp.get("LLM_interpretation_likelihood"),
#         "balance_score": interp.get("balance_score"),
#         "missing_peaks": interp.get("missing_peaks"),
#         "extra_peaks": interp.get("extra_peaks"),
#         "excess_bkg": interp.get("normalized_flag"),
#         "signal_above_bkg": interp.get("signal_above_bkg"),
#         # include the two extra scores so trust logic can read them
#         "signal_above_bkg_score": interp.get("signal_above_bkg_score"),
#         "bkg_overshoot_score": interp.get("bkg_overshoot_score"),
#         "trust_score": round(ts, 3) if ts is not None else None,
#         "trustworthy": trusty,
#         "trust_reasons": reasons,
#         # cell parameters (per-interpretation)
#         "a": cp.get("a"),
#         "b": cp.get("b"),
#         "c": cp.get("c"),
#         "alpha": cp.get("alpha"),
#         "beta": cp.get("beta"),
#         "gamma": cp.get("gamma"),
#         # NEW fields
#         "morphology": morph_str,
#         "l1_to_target": l1_to_target,
#         "l1_to_measured": l1_to_measured,
#         "l1p_to_target_list": l1p_target_list,
#         "l1p_to_measured_list": l1p_measured_list,
#         "l1p_to_target_str": _fmt_list(l1p_target_list),
#         "l1p_to_measured_str": _fmt_list(l1p_measured_list),
#     }


# def dedup(seq):
#     seen, out = set(), []
#     for x in seq:
#         if x not in seen:
#             out.append(x)
#             seen.add(x)
#     return out


# def safe_get(row, col, default=""):
#     return row[col] if (col in row and pd.notna(row[col])) else default


# def pretty_precursors(val):
#     """Handle lists, strings, JSON-like '["A","B"]', and CSV/semicolon."""
#     if val is None or (isinstance(val, float) and pd.isna(val)):
#         return ""
#     if isinstance(val, list):
#         return ", ".join(map(str, val))
#     s = str(val)
#     try:
#         parsed = ast.literal_eval(s)
#         if isinstance(parsed, list):
#             return ", ".join(map(str, parsed))
#     except Exception:
#         pass
#     parts = re.split(r"[;,]", s)
#     parts = [p.strip() for p in parts if p.strip()]
#     return ", ".join(parts)


# # ----------------- core -----------------
# def compare_dara_vs_aif(json_data, sample_names, ruff_df, project="RRUFF"):
#     html_all = ""
#     html_diff = ""

#     # R* only
#     if project.upper() == "RRUFF":
#         sample_names = [s for s in sample_names if str(s).startswith("R")]
#     if not sample_names:
#         console.print("[yellow]No samples starting with 'R' found to process.[/]")
#         return

#     for sample_name in sample_names:
#         if sample_name not in json_data:
#             console.print(f"[red]⚠️ Sample '{sample_name}' not found in interpretations JSON.[/]")
#             continue

#         interpretations = json_data[sample_name]

#         # AIF (max posterior) and DARA (min rwp)
#         aif_interp_id, aif_interp = max(
#             interpretations.items(),
#             key=lambda x: x[1].get("posterior_probability", 0)
#         )
#         dara_interp_id, dara_interp = min(
#             interpretations.items(),
#             key=lambda x: x[1].get("rwp", float("inf"))
#         )

#         # Rank all by posterior (desc) for table column order
#         ranked = sorted(
#             interpretations.items(),
#             key=lambda x: x[1].get("posterior_probability", 0),
#             reverse=True
#         )

#         # Column labels like: AIF(I_4), I_3, Dara(I_2), ...
#         col_labels = []
#         for iid, _ in ranked:
#             if iid == aif_interp_id and iid == dara_interp_id:
#                 col_labels.append(f"AIF=Dara({iid})")
#             elif iid == aif_interp_id:
#                 col_labels.append(f"AIF({iid})")
#             elif iid == dara_interp_id:
#                 col_labels.append(f"Dara({iid})")
#             else:
#                 col_labels.append(f"I_{iid}")

#         console.print(f"\n📌 Sample: [bold cyan]{sample_name}[/]")

#         # ---------- RRUFF metadata ----------
#         filt = ruff_df[ruff_df["Name"].astype(str).str.fullmatch(sample_name, na=False)]
#         if filt.empty:
#             swapped = sample_name.replace("_", "-") if "_" in sample_name else sample_name.replace("-", "_")
#             filt = ruff_df[ruff_df["Name"].astype(str).str.fullmatch(swapped, na=False)]

#         # Defaults if no metadata row
#         desc_cell  = {"a": None, "b": None, "c": None, "alpha": None, "beta": None, "gamma": None}
#         desc_sgns  = set()
#         target_norm = ""
#         target_base = ""
#         meas_list   = []

#         if not filt.empty:
#             row = filt.iloc[0]
#             # --- description context (RRUFF uses Measured_Phases, CNRS falls back to Target) ---
#             desc_cell  = desc_cell_from_row(row)
#             desc_sgns  = parse_sgn_set(safe_get(row, "Space_Group_Number"))

#             target_raw  = safe_get(row, "Target")
#             target_norm = normalize_rruff_formula_text(target_raw) if target_raw else ""
#             target_base = split_phase_name(target_norm)[0] if target_norm else ""
#             target_base_clean = _norm_base(target_norm) if target_norm else ""

#             # Raw (exactly as in CSV, just split on comma/semicolon)
#             meas_list_raw = []
#             if "Measured_Phases" in ruff_df.columns:
#                 s_raw = safe_get(row, "Measured_Phases")
#                 if s_raw:
#                     meas_list_raw = [t.strip() for t in re.split(r"[;,]", str(s_raw)) if t.strip()]

#             # Clean: prefer Measured_Phases_clean column; otherwise normalize raw
#             if "Measured_Phases_clean" in ruff_df.columns:
#                 s_clean = safe_get(row, "Measured_Phases_clean")
#                 meas_list = [t.strip() for t in re.split(r"[;,]", str(s_clean)) if t.strip()] if s_clean else []
#             else:
#                 meas_list = normalize_phase_list_field(s_raw) if "s_raw" in locals() and s_raw else []

#             # reference bases for phase matching in table: measured bases if available, else target
#             if meas_list:
#                 ref_bases = {split_phase_name(p)[0] for p in meas_list}
#             else:
#                 ref_bases = {target_base} if target_base else set()

#             # SGNs present in interpretations for measured bases
#             sgns_in_interps_for_ref = set()
#             for _, interp in interpretations.items():
#                 for p in interp.get("phases", []):
#                     base, suffix = split_phase_name(p)
#                     if ref_bases and base not in ref_bases:
#                         continue
#                     m = re.search(r"_(\d+)", suffix or "")
#                     if m:
#                         try:
#                             sgns_in_interps_for_ref.add(int(m.group(1)))
#                         except Exception:
#                             pass

#             sgns_to_highlight = desc_sgns & sgns_in_interps_for_ref

#             name_rruff = safe_get(row, "Name_RRUFF")
#             precursors = pretty_precursors(safe_get(row, "Precursors"))
#             temp_c = safe_get(row, "Temperature (C)")
#             dwell_h = safe_get(row, "Dwell Duration (h)")
#             furnace = safe_get(row, "Furnace")
#             measured_phases = safe_get(row, "Measured_Phases")  # keep this visible
#             # NEW: symmetry + cell parameters from RRUFF metadata
#             cs   = safe_get(row, "Crystal_System")
#             sg_pretty = format_sg_list_with_highlight(safe_get(row, "Space_Group_Number"), sgns_to_highlight)

#             a_v  = safe_get(row, "a")
#             b_v  = safe_get(row, "b")
#             c_v  = safe_get(row, "c")
#             al_v = safe_get(row, "alpha")
#             be_v = safe_get(row, "beta")
#             ga_v = safe_get(row, "gamma")

#             lines = ["[bold]Synthesis Description:[/bold]"]
#             if name_rruff:       lines.append(f"• Name: [cyan]{name_rruff}[/cyan]")
#             if target_raw:
#                 if meas_list:
#                     highlight_target = (target_norm in set(meas_list))
#                 else:
#                     # CNRS mode: highlight if any interp phase base equals the target base
#                     highlight_target = False
#                     if target_base:
#                         for _, interp in interpretations.items():
#                             bases = {split_phase_name(p)[0] for p in interp.get("phases", [])}
#                             if target_base in bases:
#                                 highlight_target = True
#                                 break
#                 lines.append(target_with_cs_sg(row, target_raw, highlight=highlight_target))
#             sym_parts = []
#             if cs:
#                 # sym_parts.append(f"Crystal System: [cyan]{cs}[/cyan]")
#                 sym_parts.append(f"Crystal System: {cs}")
#             if sg_pretty:
#                 sym_parts.append(f"Space Group #: {sg_pretty}")
#             if sym_parts:
#                 lines.append("• " + " ".join(sym_parts))

#             cell_parts = []
#             if a_v != "":  cell_parts.append(f"a={fmt_num(a_v)} Å")
#             if b_v != "":  cell_parts.append(f"b={fmt_num(b_v)} Å")
#             if c_v != "":  cell_parts.append(f"c={fmt_num(c_v)} Å")
#             if al_v != "": cell_parts.append(f"α={fmt_num(al_v)}°")
#             if be_v != "": cell_parts.append(f"β={fmt_num(be_v)}°")
#             if ga_v != "": cell_parts.append(f"γ={fmt_num(ga_v)}°")

#             if cell_parts:
#                 lines.append("• Cell: " + ", ".join(cell_parts))

#             if precursors:       lines.append(f"• Precursors: {precursors}")
#             if temp_c != "":     lines.append(f"• Temperature: {temp_c}°C")
#             if dwell_h != "":    lines.append(f"• Duration: {dwell_h} h")
#             if furnace:          lines.append(f"• Furnace: {furnace}")
#             # if meas_list:
#             #     tokens = []
#             #     for ph in meas_list:
#             #         token = f"[bold orange1]{ph}[/]" if (target_norm and ph == target_norm) else ph
#             #         tokens.append(token)
#             #     pretty_measured = ", ".join(tokens)
#             #     lines.append(f"• Measured Phases: [magenta]{pretty_measured}[/magenta]")
#             # --- Measured phases: show RAW and CLEAN, both with target highlighting (base-only) ---

#             # RAW (from Measured_Phases)
#             if 'meas_list_raw' in locals() and meas_list_raw:
#                 tokens_raw = []
#                 for ph in meas_list_raw:
#                     base_clean = _norm_base(ph)  # normalize for comparison only
#                     if target_base_clean and base_clean == target_base_clean:
#                         tokens_raw.append(f"[bold orange1]{ph}[/]")
#                     else:
#                         tokens_raw.append(ph)
#                 lines.append(f"• Measured Phases (raw): [magenta]{', '.join(tokens_raw)}[/magenta]")

#             # CLEAN (from Measured_Phases_clean, or normalized fallback)
#             if meas_list:
#                 tokens_clean = []
#                 for ph in meas_list:
#                     base_clean = _norm_base(ph)  # already clean, but keep same comparison path
#                     if target_base_clean and base_clean == target_base_clean:
#                         tokens_clean.append(f"[bold orange1]{ph}[/]")
#                     else:
#                         tokens_clean.append(ph)
#                 lines.append(f"• Measured_Phases_clean: [magenta]{', '.join(tokens_clean)}[/magenta]")

#             # console.print("\n".join(lines))
#             # Defer printing until after we compute morphology so we can color-match it
#             metadata_lines = lines
#             if target_base_clean and base_clean == target_base_clean:
#                 tokens_clean.append(f"[bold orange1]{ph}[/]")

#         # Inject target & measured formulas into each interpretation dict so summarize() can see them
#         for _, interp in interpretations.items():
#             interp["_target_norm"] = target_norm if target_norm else None
#             interp["_measured_list"] = meas_list[:] if meas_list else []

#         # ---------- table with ALL interpretations as columns ----------
#         table = Table(show_lines=True)
#         table.add_column("Metric", style="bold")
#         for label in col_labels:
#             table.add_column(label, style="cyan")

#         # Build summaries per column once (after injection)
#         col_summaries = []
#         for _, interp in ranked:
#             col_summaries.append(summarize(interp))

#         # Phases row (with SG match highlighting)
#         phases_row = []

#         # Clean reference sets
#         meas_bases_clean = { _norm_base(p) for p in (meas_list or []) }
#         target_base_clean = _norm_base(target_norm) if target_norm else ""

#         for _, interp in ranked:
#             names = []
#             for p in interp.get("phases", []):
#                 base_clean = _norm_base(p)

#                 # Color if matches target base OR any measured base (base-only, no SG check)
#                 if (target_base_clean and base_clean == target_base_clean) or \
#                 (meas_bases_clean and base_clean in meas_bases_clean):
#                     names.append(f"[bold orange1]{p}[/]")  # color whole token; do NOT special-case SG digits
#                 else:
#                     names.append(p)

#             phases_row.append(", ".join(names))

#         table.add_row("Phases", *phases_row)

       

#         # a,b,c and angles rows with colored tolerances vs description cell
#         abc_row = []
#         ang_row = []
#         for s in col_summaries:
#             a_s  = compare_param_color(s.get("a"),      desc_cell.get("a"))
#             b_s  = compare_param_color(s.get("b"),      desc_cell.get("b"))
#             c_s  = compare_param_color(s.get("c"),      desc_cell.get("c"))
#             al_s = compare_param_color(s.get("alpha"),  desc_cell.get("alpha"))
#             be_s = compare_param_color(s.get("beta"),   desc_cell.get("beta"))
#             ga_s = compare_param_color(s.get("gamma"),  desc_cell.get("gamma"))

#             abc_row.append(f"a={a_s}, b={b_s}, c={c_s}")
#             ang_row.append(f"α={al_s}, β={be_s}, γ={ga_s}")

#         table.add_row("a,b,c (Å)", *abc_row)
#         table.add_row("α,β,γ (°)", *ang_row)

#         # --- NEW: morphology row (highlight tokens matching CSV Crystal_System) ---
#         morph_row = []
#         csv_cs = cs if 'cs' in locals() else None
#         for s in col_summaries:
#             raw = s.get("morphology") or "—"
#             morph_row.append(color_morphologies(raw, csv_cs))
#         table.add_row("Morphology", *morph_row)
#         # --- Color the "Crystal System:" field in the metadata if any morphology matches ---
#         # --- Reprint metadata with CS highlighted if any morphology matches it ---
#         if 'metadata_lines' in locals() and metadata_lines:
#             try:
#                 if cs:
#                     # does ANY interpretation's morphology contain the same crystal system?
#                     any_match = any(
#                         cs.strip().lower() in [(t.strip().lower()) for t in (s.get("morphology") or "").split(",")]
#                         for s in col_summaries
#                     )
#                     if any_match:
#                         for i, line in enumerate(metadata_lines):
#                             if "Crystal System:" in line:
#                                 # Rebuild just that part of the line robustly
#                                 # Example: "• Crystal System: hexagonal Space Group #: 194"
#                                 before, sep, after = line.partition("Crystal System:")
#                                 if sep:
#                                     # after starts with " hexagonal Space Group #: 194"
#                                     # replace the first occurrence of the raw cs token only
#                                     after = after.replace(f" {cs}", f" [bold orange1]{cs}[/]", 1)
#                                     metadata_lines[i] = before + sep + after
#                                 break
#                 # now print the (possibly recolored) metadata lines
#                 console.print("\n".join(metadata_lines))
#             except Exception:
#                 console.print("\n".join(metadata_lines))
#         # --- Highlight the Crystal_System line in the description if any morphology matches it ---
#         try:
#             # Check if any morphology string contains the crystal system
#             if csv_cs and any(csv_cs.lower() in (s.get("morphology") or "").lower() for s in col_summaries):
#                 colored_cs = f"[bold orange1]{csv_cs}[/]"
#                 console.print(f"[dim](Crystal system '{colored_cs}' appears in morphology.)[/dim]")
#         except Exception:
#             pass

#         # --- NEW: L1 distances (mixture vs target, mixture vs measured[min]) ---
#         vals = [s.get("l1_to_target") for s in col_summaries]
#         table.add_row("L1(Target)", *format_across(vals, is_lower_better=True))

#         vals = [s.get("l1_to_measured") for s in col_summaries]
#         table.add_row("L1(Measured)", *format_across(vals, is_lower_better=True))

#         # --- NEW: per-phase L1 (comma-separated; order matches 'Phases') ---
#         row = [s.get("l1p_to_target_str") for s in col_summaries]
#         table.add_row("L1(Target) per phase", *row)

#         row = [s.get("l1p_to_measured_str") for s in col_summaries]
#         table.add_row("L1(Measured) per phase", *row)

#         # existing metric rows
#         metrics = [
#             ("rwp", True),
#             ("search_result_rwp", True),
#             ("score", False),
#             ("search_result_score", False),
#             ("llm_likelihood", False),
#             ("balance_score", False),
#             ("missing_peaks", True),   # ints
#             ("extra_peaks", True),     # ints
#             ("excess_bkg", True),
#             ("signal_above_bkg", False),
#             ("trust_score", False),
#         ]

#         for key, is_lower_better in metrics:
#             vals = [s.get(key) for s in col_summaries]
#             colored = format_across(vals, is_lower_better=is_lower_better)
#             table.add_row(key, *colored)

#         # Trustworthy boolean with reasons
#         trust_cells = []
#         for s in col_summaries:
#             if s.get("trustworthy", False):
#                 trust_cells.append("[bold green]True[/]")
#             else:
#                 rs = ", ".join(s.get("trust_reasons", [])) or "unqualified"
#                 trust_cells.append(f"[bold red1]False[/] ([italic]{rs}[/])")
#         table.add_row("Trustworthy", *trust_cells)

#         console.print(table)

#         # Export HTML for this sample
#         html_sample = console.export_html(inline_styles=True, clear=True)
#         html_all += html_sample
#         if aif_interp_id != dara_interp_id:
#             html_diff += html_sample

#     # Save combined reports (use function arg 'project' for tag)
#     tag = project.upper()
#     with open(f"aif_dara_report_all_{tag}.html", "w") as f:
#         f.write("<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
#                 + html_all + "</body></html>")
#     with open(f"aif_dara_report_diff_selections_{tag}.html", "w") as f:
#         f.write("<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
#                 + html_diff + "</body></html>")
#     console.save_text(f"aif_dara_report_{tag}.txt")


# # ----------------- CLI -----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compare DARA and AIF interpretations for RRUFF samples (R* only).")
#     parser.add_argument("json_file", help="Path to interpretations JSON.")
#     parser.add_argument("samples", nargs="+", help="Sample names or 'all'")

#     parser.add_argument(
#         "--project",
#         choices=["RRUFF", "CNRS"],
#         default="RRUFF",
#         help="Dataset flavor. Affects metadata CSV path and output filenames.",
#     )
#     parser.add_argument(
#         "--meta-csv",
#         default=None,
#         help="Optional override path to metadata CSV. If set, ignores --project default path.",
#     )
#     parser.add_argument(
#     "--cif-root", "--cif_root", dest="cif_root", metavar="DIR",
#     default=None,
#     help="Folder that contains the 'cifs/' directory (or the 'cifs/' directory itself). Used to resolve phase_cifs for morphology.",
#     )

#     parser.add_argument(
#     "--morph-debug",
#     action="store_true",
#     help="Print reasons when morphology cannot be resolved.",
# )

#     args = parser.parse_args()
#     MORPH_DEBUG = bool(args.morph_debug)

#     json_path = Path(args.json_file)
#     if not json_path.exists():
#         print(f"❌ File not found: {json_path}")
#         raise SystemExit(1)

#     # Set CIF root global (can be None)
#     # Set CIF root global (can be None). Accept either a parent that contains 'cifs/',
#     # or the 'cifs/' directory itself.


#     # or the 'cifs/' directory itself.
#     _CIF_ROOT = None
#     if args.cif_root:
#         p = Path(args.cif_root).resolve()
#         # If they passed .../cifs, use its parent so joins like root/'cifs/...' work
#         if p.is_dir() and p.name.lower() == "cifs":
#             _CIF_ROOT = p.parent
#         else:
#             _CIF_ROOT = p

#     # Fallback auto-detect if not provided: try the JSON's parent
#     if _CIF_ROOT is None:
#         candidate = json_path.parent
#         if (candidate / "cifs").is_dir():
#             _CIF_ROOT = candidate

#     # Final note for visibility
#     try:
#         if _CIF_ROOT:
#             console.print(f"[dim]CIF root resolved to: {_CIF_ROOT}[/dim]")
#         else:
#             console.print("[yellow]No CIF root set; morphology will show '-' (as requested).[/yellow]")
#     except Exception:
#         pass

#     # Load interpretations + sample list
#     json_data = load_json(json_path)
#     sample_list = list(json_data.keys()) if args.samples == ["all"] else args.samples
     

#     # Load RRUFF/CNRS metadata; use your exact path defaults
#     PROJECT_META = {
#         "RRUFF": [
#             "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/RRUFF/metadata_rruff_replaced.csv",
#         ],
#         "CNRS": [
#             "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/opXRD/opxrd/CNRS/summary.csv",
#         ],
#     }
#     if args.meta_csv:
#         ruff_candidates = [args.meta_csv]
#     else:
#         ruff_candidates = PROJECT_META.get(args.project, [])
#     ruff_df = None
#     errors = []
#     for p in ruff_candidates:
#         try:
#             ruff_df = pd.read_csv(p)
#             break
#         except Exception as e:
#             errors.append((p, str(e)))

#     if ruff_df is None:
#         print("❌ Failed to load metadata CSV. Tried:")
#         for p, err in errors:
#             print(f"   - {p}: {err}")
#         raise SystemExit(1)

#     if "Name" not in ruff_df.columns:
#         print("❌ Metadata CSV must include a 'Name' column.")
#         raise SystemExit(1)

#     if "Measured_Phases" not in ruff_df.columns and args.project.upper() == "RRUFF":
#         console.print("[yellow]⚠️ 'Measured_Phases' column not found; field will be blank.[/]")

#     compare_dara_vs_aif(json_data, sample_list, ruff_df, project=args.project)

#     # Example invocations:
#     # python compare_AIF_Dara_selections_RRUFF.py  /Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/interpretations/interpretations_rruff_new.json all  --cif-root /Users/odartsi/Documents/GitHub/XRD_Likelihood_ML/ICSD_2024/DARA_pool_fixed/pool/
#     #
#     # python compare_AIF_Dara_selections_RRUFF.py 
#     #   /Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/interpretations/interpretations_opXRD_CNRS.json all 
#     #   --project CNRS --cif-root /Users/odartsi/Documents/GitHub/Dara_data/notebooks/cifs/

from rich.console import Console
from rich.table import Table
import json
from pathlib import Path
import argparse
import pandas as pd
import ast
import re
import numpy as np

# --- pymatgen imports (per your preference) ---
from pymatgen.core import Structure, Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

console = Console(record=True)

# one-time flag for nm→Å console note
_nm_note_printed = False

# Root folder that contains the 'cifs/' directory; set via --cif-root
_CIF_ROOT: Path | None = None
MORPH_DEBUG = False

# --- RRUFF formula normalization & pretty-printing (safe with decimals) ---
_RR_CHARGE_RE = re.compile(r"\^[^^]*\^")                   # remove ^...^ charge markup
_RR_SUB_RE    = re.compile(r"_(\d+(?:\.\d+)?)_")           # _3_ / _2.00_ -> sentinel with captured number
_SENTINEL_FMT = "<<NUM:{num}>>"
_SENTINEL_RE  = re.compile(r"<<NUM:(\d+(?:\.\d+)?)>>")
# Optional cosmetic: Ca1 or Ca1.00 -> Ca (applied after rendering numbers)
_ELEM_1_RE    = re.compile(r"([A-Z][a-z]?)1(?:\.0+)?\b")

def read_measured_phases_clean(row) -> list[str]:
    """
    Read 'Measured_Phases_clean' from the CSV and split on comma/semicolon.
    Returns a list of already-clean tokens (keep any SG suffixes as-is).
    """
    s = safe_get(row, "Measured_Phases_clean")
    if not s:
        return []
    return [t.strip() for t in re.split(r"[;,]", str(s)) if t.strip()]

def _norm_base(token: str) -> str:
    base, _ = split_phase_name(token)
    return normalize_rruff_formula_text(base)

# def resolve_cif_path(cif_rel: str | Path) -> Path | None:
#     """
#     Resolve a phase_cifs entry against _CIF_ROOT (which may be a 'pool/' dir or the parent of 'cifs/').
#     Tries several options:
#       1) _CIF_ROOT / cif_rel
#       2) strip leading 'cifs/' -> _CIF_ROOT / <rest>
#       3) basename only -> _CIF_ROOT / basename(cif_rel)
#       4) absolute path if cif_rel is already absolute
#     Returns Path or None if not found.
#     """
#     if not _CIF_ROOT or not cif_rel:
#         return None
#     rel = str(cif_rel).strip()
#     root = Path(_CIF_ROOT)

#     c1 = (root / rel)
#     if c1.exists():
#         return c1

#     if rel.lower().startswith("cifs/"):
#         tail = rel.split("/", 1)[1]
#         c2 = (root / tail)
#         if c2.exists():
#             return c2
#         c3 = (root / Path(rel).name)
#         if c3.exists():
#             return c3

#     # If they accidentally stored an absolute path in JSON
#     p = Path(rel)
#     if p.is_absolute() and p.exists():
#         return p

#     return None

# put near your other helpers
_VARIANT_RE = re.compile(r"-(?:None|\d+)$", flags=re.IGNORECASE)

def _strip_variant(stem_no_ext: str) -> str:
    """Remove the trailing '-None' or '-<digits>' from a cif basename (no extension)."""
    return _VARIANT_RE.sub("", stem_no_ext)

def resolve_cif_path(cif_rel: str | Path) -> Path | None:
    """
    Resolve a phase_cifs entry against _CIF_ROOT while IGNORING any trailing variant:
    e.g., treat 'CaCuAsO5_19_(icsd_64694)-69.cif', '-0.cif', '-None.cif' as the same.
    Tries:
      1) Exact path (absolute or _CIF_ROOT/rel)
      2) Strip 'cifs/' prefix
      3) Basename in _CIF_ROOT
      4) Variant swaps (-0 <-> -None)
      5) Prefix match up to '(icsd_... )' (i.e., basename without the variant)
         – prefers exact '<stem>.cif', else the first '<stem>-*.cif' found (stable sort)
    """
    if not _CIF_ROOT or not cif_rel:
        return None

    root = Path(_CIF_ROOT)
    rel = str(cif_rel).strip()
    p_rel = Path(rel)

    # --- 1) Try as given (absolute or under root)
    direct_candidates = []
    direct_candidates.append(p_rel if p_rel.is_absolute() else (root / p_rel))
    if rel.lower().startswith("cifs/"):
        tail = rel.split("/", 1)[1]
        direct_candidates += [(root / tail), (root / Path(rel).name)]
    direct_candidates.append(root / Path(rel).name)

    # Add quick variant swaps for any direct candidates
    def _swap_variants(path: Path) -> list[Path]:
        s = str(path)
        out = []
        if s.endswith("-0.cif"):
            out.append(Path(s[:-6] + "-None.cif"))
        if s.endswith("-None.cif"):
            out.append(Path(s[:-10] + "-0.cif"))
        return out

    all_direct = direct_candidates[:]
    for c in direct_candidates:
        all_direct.extend(_swap_variants(c))

    for c in all_direct:
        try:
            if c.exists():
                return c
        except Exception:
            pass

    # --- 5) Prefix match ignoring trailing variant
    # Work only with the basename (pool appears flat)
    base = Path(rel).name
    stem = _strip_variant(Path(base).stem)  # 'CaCuAsO5_19_(icsd_64694)'
    # Prefer exact '<stem>.cif'
    exact = root / f"{stem}.cif"
    try:
        if exact.exists():
            return exact
    except Exception:
        pass
    # Else any '<stem>-*.cif' (sorted for determinism)
    try:
        matches = sorted(root.glob(f"{stem}-*.cif"))
        if matches:
            return matches[0]
    except Exception:
        pass

    # Last resort: if the original was absolute and exists
    if p_rel.is_absolute() and p_rel.exists():
        return p_rel

    return None

def _norm_phase_token(phase: str) -> str:
    """Normalize an interpretation phase token like 'Cu3H4SO8_62' to 'Cu3H4SO8_62' with cleaned formula."""
    base, suffix = split_phase_name(phase)
    return normalize_rruff_formula_text(base) + (suffix or "")

def color_morphologies(morph_str: str, csv_cs: str | None) -> str:
    """
    Color each morphology token that equals the CSV crystal system (case-insensitive).
    Keeps comma-separated formatting. Leaves '-' and '—' untouched.
    """
    if not morph_str or morph_str == "—" or not csv_cs:
        return morph_str or "—"
    target = csv_cs.strip().lower()
    parts = [t.strip() for t in morph_str.split(",")]
    colored = []
    for t in parts:
        if not t or t in {"-", "—"}:
            colored.append(t or "—")
        elif t.lower() == target:
            colored.append(f"[bold orange1]{t}[/]")
        else:
            colored.append(t)
    return ", ".join(colored)

def color_crystal_system(cs_val: str | None, morph_str: str | None) -> str:
    """
    Highlight the Crystal_System cell if it appears in the morphology list (case-insensitive).
    """
    if not cs_val or not morph_str or cs_val in {"-", "—"}:
        return cs_val or "—"
    morphs = [t.strip().lower() for t in morph_str.split(",")]
    if cs_val.strip().lower() in morphs:
        return f"[bold orange1]{cs_val}[/]"
    return cs_val

def _normalize_num_str(numstr: str) -> str:
    """Render 2.00 -> 2, 2.50 -> 2.5, keep minimal clean form."""
    try:
        x = float(numstr)
    except Exception:
        return numstr
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    s = f"{x}".rstrip("0").rstrip(".")
    return s if s else "0"

def normalize_rruff_formula_text(s: str) -> str:
    """
    Convert RRUFF-styled 'Ca_1.00_CO_3_' -> 'CaCO3', 'As_2.00_O_3_' -> 'As2O3'.
    Steps:
      - drop ^charges^
      - turn _n_ / _n.m_ into sentinels, then remove underscores between symbols
      - restore numbers with smart formatting (2.00->2)
      - drop explicit '1' (Ca1 -> Ca)
      - strip stray spaces/boxes; keep only letters/digits after restoration
    """
    if not s:
        return ""
    t = str(s)
    t = _RR_CHARGE_RE.sub("", t)
    # replace _num_ with sentinels so the dot survives through cleanup
    t = _RR_SUB_RE.sub(lambda m: _SENTINEL_FMT.format(num=m.group(1)), t)
    # remove leftover underscores and [box]
    t = t.replace("_", "").replace("[box]", "").replace(" ", "")
    # restore numbers from sentinels
    t = _SENTINEL_RE.sub(lambda m: _normalize_num_str(m.group(1)), t)
    # Ca1 / Ca1.00 -> Ca
    _ELEM_1_RE = re.compile(r"([A-Z][a-z]?)(?:1(?:\.0+)?)((?=[A-Z]|$)|(?!\d))")
    t = _ELEM_1_RE.sub(r"\1", t)
    # final cleanup: keep only letters/digits
    t = re.sub(r"[^A-Za-z0-9]", "", t)
    return t

def normalize_phase_list_field(field) -> list[str]:
    """Split a CSV/semicolon list of phases and normalize each token."""
    if not field:
        return []
    items = re.split(r"[;,]", str(field))
    out = []
    for t in items:
        t = t.strip()
        if not t:
            continue
        out.append(normalize_rruff_formula_text(t))
    return out

def desc_cell_from_row(row) -> dict:
    """Read a,b,c,alpha,beta,gamma from RRUFF CSV (floats)."""
    keys = ("a", "b", "c", "alpha", "beta", "gamma")
    out = {}
    for k in keys:
        v = safe_get(row, k)
        try:
            out[k] = float(v) if (v not in ("", None)) else None
        except Exception:
            out[k] = None
    return out

def round_eq(x, y, nd=2) -> bool:
    if x is None or y is None:
        return False
    try:
        return round(float(x), nd) == round(float(y), nd)
    except Exception:
        return False

def fmt_match(value, desc_value, nd_show=4, nd_match=2, color="bold orange1"):
    s = fmt_num(value, nd_show)
    return f"[{color}]{s}[/]" if round_eq(value, desc_value, nd_match) else s

def parse_sgn_set(sgn_field) -> set[int]:
    return {int(tok) for tok in re.findall(r"\d+", str(sgn_field) if sgn_field is not None else "")}

def parse_measured_phase_bases(measured_phases_field) -> set[str]:
    """Use normalized measured phases for matching bases."""
    bases = set()
    for ph in normalize_phase_list_field(measured_phases_field):
        base, _ = split_phase_name(ph)
        bases.add(base)
    return bases

# --- matching + formatting helpers (no unit logic here) ---
def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def _coerce_number(x):
    """Return a float from value/tuple/list/string; else None."""
    if x is None:
        return None
    # value with uncertainty like (0.5779, 0.00011)
    if isinstance(x, (list, tuple)) and x:
        x = x[0]
    # strings like "0.5779" or "(0.5779, 0.00011)"
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("(") and x.endswith(")"):
            try:
                t = ast.literal_eval(x)
                if isinstance(t, (list, tuple)) and t:
                    x = t[0]
            except Exception:
                pass
        # fall-through: try float
    try:
        return float(x)
    except Exception:
        return None

def _normalize_unit(u: str | None) -> str | None:
    if not u:
        return None
    u = str(u).strip().lower()
    if u in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return "nm"
    if u in {"å", "ang", "angstrom", "angstroms", "a", "angström"}:
        return "ang"
    return u  # unknown token; handled by heuristic later

def _nm_to_ang_if_needed(x: float, explicit_unit: str | None) -> tuple[float | None, bool]:
    """
    Returns (value_in_angstrom, did_convert_nm_to_angstrom).
    If explicit_unit=='nm' → convert; if 'ang' → no convert; else apply heuristic.
    """
    if x is None:
        return (None, False)
    unit = _normalize_unit(explicit_unit)
    if unit == "nm":
        return (x * 10.0, True)
    if unit == "ang":
        return (x, False)
    # Heuristic if no/unknown unit: 0.05–3.0 likely nm (→ 0.5–30 Å)
    if 0.05 <= x <= 3.0:
        return (x * 10.0, True)
    return (x, False)

def get_cell_params_from_interp(interp: dict) -> dict:
    """
    Extracts a,b,c,alpha,beta,gamma, normalizes to Å for a,b,c.
    Accepts either:
      - interp["cell_parameters"] = {a,b,c,alpha,beta,gamma, unit?}
      - flat keys on interp: a,b,c,alpha,beta,gamma, unit
      - values may be floats, strings, or (value, sigma) tuples
    """
    global _nm_note_printed

    # prefer nested dict if present
    cp = interp.get("cell_parameters")
    if isinstance(cp, dict) and cp:
        src = cp
        unit_field = cp.get("unit") or interp.get("unit")
    else:
        # fall back to flat keys on the interpretation
        src = interp
        unit_field = interp.get("unit")

    # coerce raw values
    raw = {k: _coerce_number(src.get(k)) for k in ("a", "b", "c", "alpha", "beta", "gamma")}

    # convert a,b,c if needed
    converted_any = False
    out = {}
    for k in ("a", "b", "c"):
        v_ang, did = _nm_to_ang_if_needed(raw.get(k), unit_field)
        out[k] = v_ang
        converted_any = converted_any or did

    # angles: keep as-is (no unit conversion)
    for k in ("alpha", "beta", "gamma"):
        out[k] = raw.get(k)

    # one-time console note if we converted
    if converted_any and not _nm_note_printed:
        try:
            console.print("[dim]Note: interpretation cell a,b,c were in nm; converted to Å for display.[/dim]")
        except Exception:
            pass
        _nm_note_printed = True

    return out

def fmt_num(val, nd=4):
    """Format a float or int for display; '—' if None."""
    if val is None:
        return "—"
    if isinstance(val, int) and not isinstance(val, bool):
        return f"{val}"
    try:
        f = float(val)
        # don't show trailing zeros for integers-in-float form, but keep 4 dp otherwise
        return f"{f:.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(val)

# ---- Trust decision with explicit reasons (like your compare_disagreements.py) ----
TRUST_THRESHOLDS = {
    "llm_max_bad": 0.4,     # False if LLM ≤ 0.4
    "signal_min": 9000,     # False if signal_above_bkg_score < 9000  (when present)
    "bkg_max": 1200,        # False if bkg_overshoot_score > 1200     (when present)
    "ratio_min": 15,        # False if (signal/bkg) < 15              (when both present)
    "balance_min": 0.6,     # False if balance_score < 0.6
}

def target_with_cs_sg(row, target_str: str, highlight: bool = False) -> str:
    """Append (crystal system; SG …) after Target if available; optionally highlight Target."""
    cs = safe_get(row, "Crystal_System")
    sg = safe_get(row, "Space_Group_Number")
    parts = []
    if cs:
        parts.append(cs)
    if sg:
        parts.append(f"SG {sg}")
    suffix = f" [dim]({'; '.join(parts)})[/]" if parts else ""
    core = f"[bold orange1]{target_str}[/]" if highlight else f"[cyan]{target_str}[/cyan]"
    return f"• Target: {core}{suffix}"

def trust_reasons(interp: dict) -> tuple[bool, list[str]]:
    # pull values with safe defaults
    llm  = float(interp.get("LLM_interpretation_likelihood", 1.0) or 1.0)
    bal  = float(interp.get("balance_score", 1.0) or 1.0)
    sabs = float(interp.get("signal_above_bkg_score", 0.0) or 0.0)
    bkg  = float(interp.get("bkg_overshoot_score", 0.0) or 0.0)

    reasons = []
    if llm <= TRUST_THRESHOLDS["llm_max_bad"]:
        reasons.append(f"low LLM (≤{TRUST_THRESHOLDS['llm_max_bad']})")
    if sabs and sabs < TRUST_THRESHOLDS["signal_min"]:
        reasons.append(f"low signal_above_bkg_score (<{TRUST_THRESHOLDS['signal_min']})")
    if bkg and bkg > TRUST_THRESHOLDS["bkg_max"]:
        reasons.append(f"high bkg_overshoot_score (>{TRUST_THRESHOLDS['bkg_max']})")
    if sabs and bkg and (sabs / bkg) < TRUST_THRESHOLDS["ratio_min"]:
        reasons.append(f"low signal/bkg ratio (<{TRUST_THRESHOLDS['ratio_min']})")
    if bal < TRUST_THRESHOLDS["balance_min"]:
        reasons.append(f"low balance score (<{TRUST_THRESHOLDS['balance_min']})")

    return (len(reasons) == 0), reasons

def format_across(values, is_lower_better=False):
    """
    Color best/worst across multiple columns.
    values: list of numbers or None
    Returns: list[str] with bold green for best and bold red1 for worst.
    Ties: multiple best/worst get colored.
    """
    # keep track of which are numbers
    numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if not numeric:
        return ["—" if v is None else str(v) for v in values]

    # round for comparison/printing (but keep int formatting for ints)
    rounded = {}
    as_int = {}
    for i, v in numeric:
        as_int[i] = isinstance(v, int) and not isinstance(v, bool)
        rounded[i] = round(float(v), 4)

    # find best/worst indices
    comp_vals = {i: rounded[i] for i, _ in numeric}
    if is_lower_better:
        best_val = min(comp_vals.values())
        worst_val = max(comp_vals.values())
    else:
        best_val = max(comp_vals.values())
        worst_val = min(comp_vals.values())

    best_idxs = {i for i, rv in comp_vals.items() if rv == best_val}
    worst_idxs = {i for i, rv in comp_vals.items() if rv == worst_val}

    # build strings
    out = []
    for idx, v in enumerate(values):
        if v is None:
            out.append("—")
            continue
        if idx in rounded:
            s = f"{int(v)}" if as_int[idx] else f"{rounded[idx]:.4f}"
            if idx in best_idxs and idx in worst_idxs:
                # all equal: no color
                out.append(s)
            elif idx in best_idxs:
                out.append(f"[bold green]{s}[/]")
            elif idx in worst_idxs:
                out.append(f"[bold red1]{s}[/]")
            else:
                out.append(s)
        else:
            out.append(str(v))
    return out

# ----------------- helpers -----------------
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

def compare_param_color(val, desc_val, tol_orange=0.1, tol_blue=0.3):
    """
    Compare interpretation vs description numeric value:
      - |Δ| <= tol_orange → orange
      - tol_orange < |Δ| <= tol_blue → blue
      - otherwise plain
    Returns formatted string with color tag.
    """
    if val is None:
        return "—"
    s = fmt_num(val)
    if desc_val is None:
        return s
    try:
        diff = abs(float(val) - float(desc_val))
    except Exception:
        return s
    if diff <= tol_orange:
        return f"[bold orange1]{s}[/]"
    elif diff <= tol_blue:
        return f"[bold blue]{s}[/]"
    return s

def format_sg_list_with_highlight(sg_field, highlight_set: set[int], color="bold orange1") -> str:
    """
    Render 'Space Group #:' numbers, coloring any that appear in highlight_set.
    Accepts CSV forms like '20;59;62'.
    """
    if sg_field in (None, ""):
        return ""
    tokens = re.findall(r"\d+", str(sg_field))
    if not tokens:
        return str(sg_field)
    parts = []
    for t in tokens:
        try:
            n = int(t)
            parts.append(f"[{color}]{n}[/]" if n in highlight_set else str(n))
        except Exception:
            parts.append(str(t))
    return ";".join(parts)

def compute_trust_score(interp: dict) -> float:
    """
    Smooth trust score (0..1) using six soft criteria:
      - LLM_interpretation_likelihood  (≥ 0.4)
      - signal_above_background score  (≥ 9000)
      - background overshoot score     (≤ 1200)
      - signal/overshoot ratio         (≥ 15)
      - balance_score                  (≥ 0.6)
      - normalized_score (peak match)  (≥ 0.6)

    Trust score = 1 - average of per-criterion penalties (each clipped to [0,1]).
    """
    try:
        llm       = float(interp.get("LLM_interpretation_likelihood", 1.0))
        signal    = float(interp.get("signal_above_bkg_score", 10000.0))
        overshoot = float(interp.get("bkg_overshoot_score", 0.0))
        balance   = float(interp.get("balance_score", 1.0))
        pmatch    = float(interp.get("normalized_score", 1.0))
    except (TypeError, ValueError):
        return 0.0

    # helpers
    clip01 = lambda x: max(0.0, min(1.0, x))

    # 1) LLM (ideal ≥ 0.4)
    penalty_llm = clip01((0.4 - llm) / 0.4) if llm < 0.4 else 0.0

    # 2) Signal above background score (ideal ≥ 9000)
    penalty_signal = clip01((9000.0 - signal) / 9000.0) if signal < 9000.0 else 0.0

    # 3) Background overshoot score (ideal ≤ 1200)
    penalty_overshoot = clip01((overshoot - 1200.0) / 1200.0) if overshoot > 1200.0 else 0.0

    # 4) Signal / overshoot ratio (ideal ≥ 15)
    if overshoot > 0.0:
        ratio = signal / overshoot
        penalty_ratio = clip01((15.0 - ratio) / 15.0) if ratio < 15.0 else 0.0
    else:
        penalty_ratio = 0.0  # no penalty if no overshoot

    # 5) Balance score (ideal ≥ 0.6)
    penalty_balance = clip01((0.6 - balance) / 0.6) if balance < 0.6 else 0.0

    # 6) Peak match score (ideal ≥ 0.6)
    penalty_match = clip01((0.6 - pmatch) / 0.6) if pmatch < 0.6 else 0.0

    total_penalty = (
        penalty_llm
        + penalty_signal
        + penalty_overshoot
        + penalty_ratio
        + penalty_balance
        + penalty_match
    ) / 6.0

    return round(max(0.0, 1.0 - total_penalty), 3)

def color(val1, val2, is_lower_better=False):
    """Return colored strings for values based on comparison."""
    if val1 is None or val2 is None:
        return str(val1), str(val2)

    is_integer = isinstance(val1, int) and isinstance(val2, int)
    v1 = round(val1, 4)
    v2 = round(val2, 4)

    if v1 == v2:
        return (f"{val1}" if is_integer else f"{v1:.4f}",
                f"{val2}" if is_integer else f"{v2:.4f}")

    better = v1 < v2 if is_lower_better else v1 > v2
    v1s = f"{val1}" if is_integer else f"{v1:.4f}"
    v2s = f"{val2}" if is_integer else f"{v2:.4f}"
    return (
        f"[bold green]{v1s}[/]" if better else f"[red1]{v1s}[/]",
        f"[bold red1]{v2s}[/]" if better else f"[bold green]{v2s}[/]"
    )

# =======================
# CIF → morphology helpers
# =======================

# Map space group number to crystal system
def _sg_to_crystal_system(n: int) -> str | None:
    if 1 <= n <= 2:
        return "triclinic"
    if 3 <= n <= 15:
        return "monoclinic"
    if 16 <= n <= 74:
        return "orthorhombic"
    if 75 <= n <= 142:
        return "tetragonal"
    if 143 <= n <= 167:
        return "trigonal"
    if 168 <= n <= 194:
        return "hexagonal"
    if 195 <= n <= 230:
        return "cubic"
    return None

def _log_morph_issue(msg: str):
    try:
        if MORPH_DEBUG:
            console.print(f"[dim]{msg}[/dim]")
    except Exception:
        pass

def get_crystal_system_from_cif(cif_path: Path) -> str | None:
    """
    Return the crystal system (e.g., 'cubic', 'tetragonal') from a CIF file.
    Returns None if the file can't be read or spglib is unavailable.
    """
    try:
        if not cif_path.exists():
            _log_morph_issue(f"morphology: CIF not found: {cif_path}")
            return None

        structure = Structure.from_file(str(cif_path))
        try:
            analyzer = SpacegroupAnalyzer(structure)
            cs = analyzer.get_crystal_system()
            return str(cs).lower() if cs else None
        except Exception as e:
            _log_morph_issue(f"morphology: SpacegroupAnalyzer failed (spglib missing?): {e}")
            return None
    except Exception as e:
        _log_morph_issue(f"morphology: failed to read CIF ({cif_path}): {e}")
        return None

def _infer_cs_from_phase_name(phase: str) -> str | None:
    # Expect suffix like _62
    m = re.search(r"_(\d+)\b", phase)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return _sg_to_crystal_system(n)
    except Exception:
        return None

def get_phase_morphologies(interp: dict) -> list[str]:
    """
    For each phase in interpretation, return crystal system ONLY if a CIF can be resolved.
    Otherwise '-'. (No SG/system fallback.)
    """
    out = []
    phases = list(interp.get("phases") or [])
    phase_cifs = interp.get("phase_cifs") or []
    if isinstance(phase_cifs, str):
        phase_cifs = [phase_cifs]

    # align CIFs to phases (best-effort)
    cifs_aligned = list(phase_cifs) + [None] * max(0, len(phases) - len(phase_cifs))
    cifs_aligned = cifs_aligned[:len(phases)] if phases else []

    for cif_rel in cifs_aligned:
        path = resolve_cif_path(cif_rel) if cif_rel else None
        if path:
            cs = get_crystal_system_from_cif(path)
            out.append(cs if cs else "-")
        else:
            out.append("-")
    return out

# =======================
# Composition / L1 helpers
# =======================

# Build elements list in increasing Z (compatible across pymatgen versions)
try:
    _ELEMENTS_BY_Z = sorted(list(Element), key=lambda e: e.Z)
except Exception:
    _ELEMENTS_BY_Z = [Element.from_Z(z) for z in range(1, 119)]

ALL_ELEMENTS = [e.symbol for e in _ELEMENTS_BY_Z]
SYM_TO_IDX = {sym: i for i, sym in enumerate(ALL_ELEMENTS)}

def comp_to_vec(comp: str | Composition) -> np.ndarray:
    """
    Map a composition to a length-118 vector of atomic fractions (sums to 1 over present elements).
    Any element not present gets 0. Uses atomic (not weight) fractions.
    """
    if not isinstance(comp, Composition):
        comp = Composition(comp)
    frac = comp.fractional_composition.get_el_amt_dict()
    v = np.zeros(len(ALL_ELEMENTS), dtype=float)
    for el_sym, amt in frac.items():
        idx = SYM_TO_IDX.get(el_sym)
        if idx is not None:
            v[idx] = float(amt)
    s = v.sum()
    if s > 0:
        v /= s
    return v

def l1_distance(comp_a: str | Composition, comp_b: str | Composition) -> float:
    """
    L1 (Manhattan) distance between two composition vectors.
    """
    va = comp_to_vec(comp_a)
    vb = comp_to_vec(comp_b)
    return float(np.abs(va - vb).sum())

def mixture_vector_from_interpretation(interp: dict) -> np.ndarray | None:
    """
    Combine phases by weight_fraction to a single composition vector (sums to 1).
    We use *atomic* fractions, not mass fractions.
    If missing weights, assume equal weights.
    If no phases, return None.
    """
    phases = interp.get("phases") or []
    if not phases:
        return None

    # weights
    wf = interp.get("weight_fraction")
    if isinstance(wf, (int, float)):
        wf = [wf]
    if not (isinstance(wf, list) and len(wf) == len(phases)):
        wf = [1.0] * len(phases)

    # Normalize weights
    wsum = float(sum(wf)) if wf else 0.0
    if wsum <= 0:
        wf = [1.0] * len(phases)
        wsum = float(len(phases))
    wf = [float(x) / wsum for x in wf]

    # Phase formulas: strip SG suffix: 'Cu3H4SO8_62' → 'Cu3H4SO8'
    vec = np.zeros(len(ALL_ELEMENTS), dtype=float)
    for w, p in zip(wf, phases):
        base, _ = split_phase_name(p)
        try:
            pv = comp_to_vec(base)
            vec += w * pv
        except Exception:
            continue

    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

def per_phase_l1_lists(interp: dict, target_formula: str | None, measured_formulas: list[str]) -> tuple[list[float] | None, list[float] | None]:
    """
    Returns (per_phase_l1_to_target, per_phase_l1_to_measured_min).
    Each is a list aligned with interp['phases'] order. If target/measured not available -> None.
    """
    phases = interp.get("phases") or []
    if not phases:
        return (None, None)

    # Base formulas for each phase (strip SG suffix)
    bases = [split_phase_name(p)[0] for p in phases]

    # Precompute vectors
    try:
        target_vec = comp_to_vec(target_formula) if target_formula else None
    except Exception:
        target_vec = None

    measured_vecs = []
    for mf in measured_formulas or []:
        try:
            measured_vecs.append(comp_to_vec(mf))
        except Exception:
            pass

    # Per-phase to Target
    l1p_target = None
    if target_vec is not None:
        l1p_target = []
        for b in bases:
            try:
                l1p_target.append(float(np.abs(comp_to_vec(b) - target_vec).sum()))
            except Exception:
                l1p_target.append(None)

    # Per-phase to closest Measured
    l1p_meas = None
    if measured_vecs:
        l1p_meas = []
        for b in bases:
            try:
                v = comp_to_vec(b)
                dmin = min(float(np.abs(v - mv).sum()) for mv in measured_vecs) if measured_vecs else None
                l1p_meas.append(dmin)
            except Exception:
                l1p_meas.append(None)

    return (l1p_target, l1p_meas)

# =======================
# Summarize (extended)
# =======================

def summarize(interp):
    ts = interp.get("trust_score")
    if ts is None:
        ts = compute_trust_score(interp)
    trusty, reasons = trust_reasons(interp)

    # cell params
    cp = get_cell_params_from_interp(interp)

    # morphology per phase from CIFs
    morph_list = get_phase_morphologies(interp)
    morph_str = ", ".join(morph_list) if morph_list else "—"

    # L1 distances (mixture vs target, mixture vs measured[min])
    mix_vec = mixture_vector_from_interpretation(interp)

    l1_to_target = None
    l1_to_measured = None

    # Target & measured formulas provided by caller via injected fields
    target_formula = interp.get("_target_norm")
    measured_formulas = interp.get("_measured_list") or []

    try:
        if mix_vec is not None and target_formula:
            l1_to_target = float(np.abs(mix_vec - comp_to_vec(target_formula)).sum())
    except Exception:
        l1_to_target = None

    try:
        if mix_vec is not None and measured_formulas:
            dists = []
            for mf in measured_formulas:
                try:
                    dists.append(float(np.abs(mix_vec - comp_to_vec(mf)).sum()))
                except Exception:
                    pass
            if dists:
                l1_to_measured = float(min(dists))
    except Exception:
        l1_to_measured = None

    # Per-phase L1 lists (comma-separated strings)
    l1p_target_list, l1p_measured_list = per_phase_l1_lists(
        interp, target_formula=target_formula, measured_formulas=measured_formulas
    )

    def _fmt_list(xs):
        if not xs:
            return "—"
        parts = []
        for x in xs:
            parts.append("—" if x is None else f"{x:.4f}")
        return ", ".join(parts)

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
        # include the two extra scores so trust logic can read them
        "signal_above_bkg_score": interp.get("signal_above_bkg_score"),
        "bkg_overshoot_score": interp.get("bkg_overshoot_score"),
        "trust_score": round(ts, 3) if ts is not None else None,
        "trustworthy": trusty,
        "trust_reasons": reasons,
        # cell parameters (per-interpretation)
        "a": cp.get("a"),
        "b": cp.get("b"),
        "c": cp.get("c"),
        "alpha": cp.get("alpha"),
        "beta": cp.get("beta"),
        "gamma": cp.get("gamma"),
        # NEW fields
        "morphology": morph_str,
        "l1_to_target": l1_to_target,
        "l1_to_measured": l1_to_measured,
        "l1p_to_target_list": l1p_target_list,
        "l1p_to_measured_list": l1p_measured_list,
        "l1p_to_target_str": _fmt_list(l1p_target_list),
        "l1p_to_measured_str": _fmt_list(l1p_measured_list),
    }

def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def safe_get(row, col, default=""):
    return row[col] if (col in row and pd.notna(row[col])) else default

def pretty_precursors(val):
    """Handle lists, strings, JSON-like '["A","B"]', and CSV/semicolon."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, list):
        return ", ".join(map(str, val))
    s = str(val)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return ", ".join(map(str, parsed))
    except Exception:
        pass
    parts = re.split(r"[;,]", s)
    parts = [p.strip() for p in parts if p.strip()]
    return ", ".join(parts)

# ----------------- core -----------------
def compare_dara_vs_aif(json_data, sample_names, ruff_df, project="RRUFF"):
    html_all = ""
    html_diff = ""

    # R* only
    if project.upper() == "RRUFF":
        sample_names = [s for s in sample_names if str(s).startswith("R")]
    if not sample_names:
        console.print("[yellow]No samples starting with 'R' found to process.[/]")
        return

    for sample_name in sample_names:
        if sample_name not in json_data:
            console.print(f"[red]⚠️ Sample '{sample_name}' not found in interpretations JSON.[/]")
            continue

        interpretations = json_data[sample_name]

        # AIF (max posterior) and DARA (min rwp)
        aif_interp_id, aif_interp = max(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0)
        )
        dara_interp_id, dara_interp = min(
            interpretations.items(),
            key=lambda x: x[1].get("rwp", float("inf"))
        )

        # Rank all by posterior (desc) for table column order
        ranked = sorted(
            interpretations.items(),
            key=lambda x: x[1].get("posterior_probability", 0),
            reverse=True
        )

        # Column labels like: AIF(I_4), I_3, Dara(I_2), ...
        col_labels = []
        for iid, _ in ranked:
            if iid == aif_interp_id and iid == dara_interp_id:
                col_labels.append(f"AIF=Dara({iid})")
            elif iid == aif_interp_id:
                col_labels.append(f"AIF({iid})")
            elif iid == dara_interp_id:
                col_labels.append(f"Dara({iid})")
            else:
                col_labels.append(f"I_{iid}")

        console.print(f"\n📌 Sample: [bold cyan]{sample_name}[/]")

        # ---------- RRUFF metadata ----------
        filt = ruff_df[ruff_df["Name"].astype(str).str.fullmatch(sample_name, na=False)]
        if filt.empty:
            swapped = sample_name.replace("_", "-") if "_" in sample_name else sample_name.replace("-", "_")
            filt = ruff_df[ruff_df["Name"].astype(str).str.fullmatch(swapped, na=False)]

        # Defaults if no metadata row
        desc_cell  = {"a": None, "b": None, "c": None, "alpha": None, "beta": None, "gamma": None}
        desc_sgns  = set()
        target_norm = ""
        target_base = ""
        meas_list   = []

        if not filt.empty:
            row = filt.iloc[0]
            # --- description context (RRUFF uses Measured_Phases, CNRS falls back to Target) ---
            desc_cell  = desc_cell_from_row(row)
            desc_sgns  = parse_sgn_set(safe_get(row, "Space_Group_Number"))

            target_raw  = safe_get(row, "Target")
            target_norm = normalize_rruff_formula_text(target_raw) if target_raw else ""
            target_base = split_phase_name(target_norm)[0] if target_norm else ""
            target_base_clean = _norm_base(target_norm) if target_norm else ""

            # Raw (exactly as in CSV, just split on comma/semicolon)
            # meas_list_raw = []
            # if "Measured_Phases" in ruff_df.columns:
            #     s_raw = safe_get(row, "Measured_Phases")
            #     if s_raw:
            #         meas_list_raw = [t.strip() for t in re.split(r"[;,]", str(s_raw)) if t.strip()]
            # --- replace your current RAW/CLEAN block with this ---
            # Raw kept only for display; L1 will use CLEAN exclusively
            meas_list_raw = []
            if "Measured_Phases" in ruff_df.columns:
                s_raw = safe_get(row, "Measured_Phases")
                if s_raw:
                    meas_list_raw = [t.strip() for t in re.split(r"[;,]", str(s_raw)) if t.strip()]

            # CLEAN (authoritative for all L1 calculations and phase-coloring)
            meas_list = read_measured_phases_clean(row)  # returns [] if column missing/empty

            # Clean: prefer Measured_Phases_clean column; otherwise normalize raw
            if "Measured_Phases_clean" in ruff_df.columns:
                s_clean = safe_get(row, "Measured_Phases_clean")
                meas_list = [t.strip() for t in re.split(r"[;,]", str(s_clean)) if t.strip()] if s_clean else []
            else:
                meas_list = normalize_phase_list_field(s_raw) if "s_raw" in locals() and s_raw else []

            # reference bases for phase matching in table: measured bases if available, else target
            if meas_list:
                ref_bases = {split_phase_name(p)[0] for p in meas_list}
            else:
                ref_bases = {target_base} if target_base else set()

            # SGNs present in interpretations for measured bases
            sgns_in_interps_for_ref = set()
            for _, interp in interpretations.items():
                for p in interp.get("phases", []):
                    base, suffix = split_phase_name(p)
                    if ref_bases and base not in ref_bases:
                        continue
                    m = re.search(r"_(\d+)", suffix or "")
                    if m:
                        try:
                            sgns_in_interps_for_ref.add(int(m.group(1)))
                        except Exception:
                            pass

            sgns_to_highlight = desc_sgns & sgns_in_interps_for_ref

            name_rruff = safe_get(row, "Name_RRUFF")
            precursors = pretty_precursors(safe_get(row, "Precursors"))
            temp_c = safe_get(row, "Temperature (C)")
            dwell_h = safe_get(row, "Dwell Duration (h)")
            furnace = safe_get(row, "Furnace")
            measured_phases = safe_get(row, "Measured_Phases")
            cs   = safe_get(row, "Crystal_System")
            sg_pretty = format_sg_list_with_highlight(safe_get(row, "Space_Group_Number"), sgns_to_highlight)

            a_v  = safe_get(row, "a")
            b_v  = safe_get(row, "b")
            c_v  = safe_get(row, "c")
            al_v = safe_get(row, "alpha")
            be_v = safe_get(row, "beta")
            ga_v = safe_get(row, "gamma")

            lines = ["[bold]Synthesis Description:[/bold]"]
            if name_rruff:       lines.append(f"• Name: [cyan]{name_rruff}[/cyan]")
            if target_raw:
                if meas_list:
                    highlight_target = (target_norm in set(meas_list))
                else:
                    # CNRS mode: highlight if any interp phase base equals the target base
                    highlight_target = False
                    if target_base:
                        for _, interp in interpretations.items():
                            bases = {split_phase_name(p)[0] for p in interp.get("phases", [])}
                            if target_base in bases:
                                highlight_target = True
                                break
                lines.append(target_with_cs_sg(row, target_raw, highlight=highlight_target))
            sym_parts = []
            if cs:
                sym_parts.append(f"Crystal System: {cs}")
            if sg_pretty:
                sym_parts.append(f"Space Group #: {sg_pretty}")
            if sym_parts:
                lines.append("• " + " ".join(sym_parts))

            cell_parts = []
            if a_v != "":  cell_parts.append(f"a={fmt_num(a_v)} Å")
            if b_v != "":  cell_parts.append(f"b={fmt_num(b_v)} Å")
            if c_v != "":  cell_parts.append(f"c={fmt_num(c_v)} Å")
            if al_v != "": cell_parts.append(f"α={fmt_num(al_v)}°")
            if be_v != "": cell_parts.append(f"β={fmt_num(be_v)}°")
            if ga_v != "": cell_parts.append(f"γ={fmt_num(ga_v)}°")
            if cell_parts:
                lines.append("• Cell: " + ", ".join(cell_parts))

            if precursors:       lines.append(f"• Precursors: {precursors}")
            if temp_c != "":     lines.append(f"• Temperature: {temp_c}°C")
            if dwell_h != "":    lines.append(f"• Duration: {dwell_h} h")
            if furnace:          lines.append(f"• Furnace: {furnace}")

            # --- Measured phases: show RAW and CLEAN, both with target highlighting (base-only) ---
            # RAW (display only)
            if meas_list_raw:
                tokens_raw = []
                for ph in meas_list_raw:
                    base_clean = _norm_base(ph)
                    if target_norm and base_clean == _norm_base(target_norm):
                        tokens_raw.append(f"[bold orange1]{ph}[/]")
                    else:
                        tokens_raw.append(ph)
                lines.append(f"• Measured Phases (raw): [magenta]{', '.join(tokens_raw)}[/magenta]")

            # CLEAN (drives L1)
            if meas_list:
                tokens_clean = []
                for ph in meas_list:
                    base_clean = _norm_base(ph)
                    if target_norm and base_clean == _norm_base(target_norm):
                        tokens_clean.append(f"[bold orange1]{ph}[/]")
                    else:
                        tokens_clean.append(ph)
                lines.append(f"• Measured_Phases_clean (used for L1): [magenta]{', '.join(tokens_clean)}[/magenta]")

            # Defer printing until after morphology so we can color-match CS
            metadata_lines = lines

        # Inject target & measured formulas into each interpretation dict so summarize() can see them
        for _, interp in interpretations.items():
            interp["_target_norm"] = target_norm if target_norm else None
            interp["_measured_list"] = meas_list[:] #if meas_list else []

        # ---------- table with ALL interpretations as columns ----------
        table = Table(show_lines=True)
        table.add_column("Metric", style="bold")
        for label in col_labels:
            table.add_column(label, style="cyan")

        # Build summaries per column once (after injection)
        col_summaries = []
        for _, interp in ranked:
            col_summaries.append(summarize(interp))

        # Phases row (with base-only highlighting vs target & measured clean)
        phases_row = []

        meas_bases_clean = { _norm_base(p) for p in (meas_list or []) }
        target_base_clean_for_row = _norm_base(target_norm) if target_norm else ""

        for _, interp in ranked:
            names = []
            for p in interp.get("phases", []):
                base_clean = _norm_base(p)
                if (target_base_clean_for_row and base_clean == target_base_clean_for_row) or \
                   (meas_bases_clean and base_clean in meas_bases_clean):
                    names.append(f"[bold orange1]{p}[/]")
                else:
                    names.append(p)
            phases_row.append(", ".join(names))

        table.add_row("Phases", *phases_row)

        # a,b,c and angles rows with colored tolerances vs description cell
        abc_row = []
        ang_row = []
        for s in col_summaries:
            a_s  = compare_param_color(s.get("a"),      desc_cell.get("a"))
            b_s  = compare_param_color(s.get("b"),      desc_cell.get("b"))
            c_s  = compare_param_color(s.get("c"),      desc_cell.get("c"))
            al_s = compare_param_color(s.get("alpha"),  desc_cell.get("alpha"))
            be_s = compare_param_color(s.get("beta"),   desc_cell.get("beta"))
            ga_s = compare_param_color(s.get("gamma"),  desc_cell.get("gamma"))

            abc_row.append(f"a={a_s}, b={b_s}, c={c_s}")
            ang_row.append(f"α={al_s}, β={be_s}, γ={ga_s}")

        table.add_row("a,b,c (Å)", *abc_row)
        table.add_row("α,β,γ (°)", *ang_row)

        # Morphology row (highlight tokens matching CSV Crystal_System)
        morph_row = []
        csv_cs = cs if 'cs' in locals() else None
        for s in col_summaries:
            raw = s.get("morphology") or "—"
            morph_row.append(color_morphologies(raw, csv_cs))
        table.add_row("Morphology", *morph_row)

        # Print the (possibly recolored) metadata now
        if 'metadata_lines' in locals() and metadata_lines:
            try:
                if cs:
                    any_match = any(
                        cs.strip().lower() in [(t.strip().lower()) for t in (s.get("morphology") or "").split(",")]
                        for s in col_summaries
                    )
                    if any_match:
                        for i, line in enumerate(metadata_lines):
                            if "Crystal System:" in line:
                                before, sep, after = line.partition("Crystal System:")
                                if sep:
                                    after = after.replace(f" {cs}", f" [bold orange1]{cs}[/]", 1)
                                    metadata_lines[i] = before + sep + after
                                break
                console.print("\n".join(metadata_lines))
            except Exception:
                console.print("\n".join(metadata_lines))

        try:
            if csv_cs and any(csv_cs.lower() in (s.get("morphology") or "").lower() for s in col_summaries):
                colored_cs = f"[bold orange1]{csv_cs}[/]"
                console.print(f"[dim](Crystal system '{colored_cs}' appears in morphology.)[/dim]")
        except Exception:
            pass

        # L1 distances
        vals = [s.get("l1_to_target") for s in col_summaries]
        table.add_row("L1(Target)", *format_across(vals, is_lower_better=True))

        vals = [s.get("l1_to_measured") for s in col_summaries]
        table.add_row("L1(Measured)", *format_across(vals, is_lower_better=True))

        # Per-phase L1
        row = [s.get("l1p_to_target_str") for s in col_summaries]
        table.add_row("L1(Target) per phase", *row)

        row = [s.get("l1p_to_measured_str") for s in col_summaries]
        table.add_row("L1(Measured) per phase", *row)

        # existing metric rows
        metrics = [
            ("rwp", True),
            ("search_result_rwp", True),
            ("score", False),
            ("search_result_score", False),
            ("llm_likelihood", False),
            ("balance_score", False),
            ("missing_peaks", True),   # ints
            ("extra_peaks", True),     # ints
            ("excess_bkg", True),
            ("signal_above_bkg", False),
            ("trust_score", False),
        ]

        for key, is_lower_better in metrics:
            vals = [s.get(key) for s in col_summaries]
            colored = format_across(vals, is_lower_better=is_lower_better)
            table.add_row(key, *colored)

        # Trustworthy boolean with reasons
        trust_cells = []
        for s in col_summaries:
            if s.get("trustworthy", False):
                trust_cells.append("[bold green]True[/]")
            else:
                rs = ", ".join(s.get("trust_reasons", [])) or "unqualified"
                trust_cells.append(f"[bold red1]False[/] ([italic]{rs}[/])")
        table.add_row("Trustworthy", *trust_cells)

        console.print(table)

        # Export HTML for this sample
        html_sample = console.export_html(inline_styles=True, clear=True)
        html_all += html_sample
        if aif_interp_id != dara_interp_id:
            html_diff += html_sample

    # Save combined reports
    tag = project.upper()
    with open(f"aif_dara_report_all_{tag}.html", "w") as f:
        f.write("<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
                + html_all + "</body></html>")
    with open(f"aif_dara_report_diff_selections_{tag}.html", "w") as f:
        f.write("<html><head><meta charset='UTF-8'><style>span { font-weight: bold !important; }</style></head><body>"
                + html_diff + "</body></html>")
    console.save_text(f"aif_dara_report_{tag}.txt")

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DARA and AIF interpretations for RRUFF samples (R* only).")
    parser.add_argument("json_file", help="Path to interpretations JSON.")
    parser.add_argument("samples", nargs="+", help="Sample names or 'all'")

    parser.add_argument(
        "--project",
        choices=["RRUFF", "CNRS"],
        default="RRUFF",
        help="Dataset flavor. Affects metadata CSV path and output filenames.",
    )
    parser.add_argument(
        "--meta-csv",
        default=None,
        help="Optional override path to metadata CSV. If set, ignores --project default path.",
    )
    parser.add_argument(
        "--cif-root", "--cif_root", dest="cif_root", metavar="DIR",
        default=None,
        help="Folder that contains the 'cifs/' directory (or the 'cifs/' directory itself). Used to resolve phase_cifs for morphology.",
    )
    parser.add_argument(
        "--morph-debug",
        action="store_true",
        help="Print reasons when morphology cannot be resolved.",
    )

    args = parser.parse_args()
    MORPH_DEBUG = bool(args.morph_debug)

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        raise SystemExit(1)

    # Set CIF root global (accept either a parent that contains 'cifs/', or the 'cifs/' directory itself)
    _CIF_ROOT = None
    if args.cif_root:
        p = Path(args.cif_root).resolve()
        if p.is_dir() and p.name.lower() == "cifs":
            _CIF_ROOT = p.parent
        else:
            _CIF_ROOT = p

    # Fallback auto-detect if not provided: try the JSON's parent
    if _CIF_ROOT is None:
        candidate = json_path.parent
        if (candidate / "cifs").is_dir():
            _CIF_ROOT = candidate

    # Visibility
    try:
        if _CIF_ROOT:
            console.print(f"[dim]CIF root resolved to: {_CIF_ROOT}[/dim]")
        else:
            console.print("[yellow]No CIF root set; morphology will show '-' (as requested).[/yellow]")
    except Exception:
        pass

    # Load interpretations + sample list
    json_data = load_json(json_path)
    sample_list = list(json_data.keys()) if args.samples == ["all"] else args.samples

    # Load RRUFF/CNRS metadata; use your exact path defaults
    PROJECT_META = {
        "RRUFF": [
            "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/RRUFF/metadata_rruff_replaced.csv",
        ],
        "CNRS": [
            "/Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/opXRD/opxrd/CNRS/summary.csv",
        ],
    }
    if args.meta_csv:
        ruff_candidates = [args.meta_csv]
    else:
        ruff_candidates = PROJECT_META.get(args.project, [])
    ruff_df = None
    errors = []
    for p in ruff_candidates:
        try:
            ruff_df = pd.read_csv(p)
            break
        except Exception as e:
            errors.append((p, str(e)))

    if ruff_df is None:
        print("❌ Failed to load metadata CSV. Tried:")
        for p, err in errors:
            print(f"   - {p}: {err}")
        raise SystemExit(1)

    if "Name" not in ruff_df.columns:
        print("❌ Metadata CSV must include a 'Name' column.")
        raise SystemExit(1)

    if "Measured_Phases" not in ruff_df.columns and args.project.upper() == "RRUFF":
        console.print("[yellow]⚠️ 'Measured_Phases' column not found; field will be blank.[/]")

    compare_dara_vs_aif(json_data, sample_list, ruff_df, project=args.project)

    # Example invocations:
    # python compare_AIF_Dara_selections_RRUFF.py  /Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/interpretations/interpretations_rruff_new.json all  --cif-root /Users/odartsi/Documents/GitHub/XRD_Likelihood_ML/ICSD_2024/DARA_pool_fixed/pool/
    #
    # python compare_AIF_Dara_selections_RRUFF.py 
    #   /Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/interpretations/interpretations_opXRD_CNRS.json all 
    #   --project CNRS --cif-root /Users/odartsi/Documents/GitHub/Dara_data/notebooks/cifs/

# #     # python compare_AIF_Dara_selections_RRUFF.py /Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/interpretations/interpretations_rruff_new.json all

# # # python compare_AIF_Dara_selections_RRUFF.py /Users/odartsi/Documents/GitHub/Automated_Interpretation_Framework/data/xrd_data/interpretations/interpretations_opXRD_CNRS.json all --project CNRS