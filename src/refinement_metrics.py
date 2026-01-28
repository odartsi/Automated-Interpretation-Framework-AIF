from pathlib import Path
from dara.structure_db import ICSDDatabase
from dara.refine import do_refinement_no_saving
from copy import deepcopy
from dara import search_phases
from dara.search.peak_matcher import PeakMatcher
from dara.peak_detection import detect_peaks
import os
import numpy as np
from itertools import combinations
from utils import (
    save_xrd_plots,
    remove_elements,
    include_elements,
    cleanup_cifs,
    cleanup_phases,
    remove_cifs_prefix,
    remove_cifs_suffix,
    extract_icsd_from_key,
    strip_phase_identifier,
    importance_factor_calculation,
    add_flag,
    net_signal_score,
    signal_above_bkg_score,
    bkg_overshoot_score,
    abs_diff_score,
    bkg_baseline_distance_score
)
import time
TIME_LIMIT_ = 2000
TIME_LIMIT = 1200
MAX_ITER_WITHOUT_NEW_INTERPRETATION = 10


def phase_assets_from_final(final_results, base_dir="cifs"):
    """
    Returns (phase_keys, phase_cifs, phase_icsd) in the SAME ORDER as LST.
    - phase_keys: list[str]
    - phase_cifs: list[str] (paths as strings; chooses existing file if present)
    - phase_icsd: list[str|None]
    """
    if hasattr(final_results, "lst_data") and hasattr(final_results.lst_data, "phases_results"):
        phase_keys = list(final_results.lst_data.phases_results.keys())
    elif hasattr(final_results, "phases_results"):
        phase_keys = list(final_results.phases_results.keys())
    else:
        return [], [], []

    phase_cifs = []
    for k in phase_keys:
        candidates = [Path(base_dir) / f"{k}.cif"]
        if k.endswith("-None"):
            candidates.append(Path(base_dir) / f"{k[:-5]}.cif")
        chosen = next((p for p in candidates if p.exists()), candidates[0])
        phase_cifs.append(str(chosen))

    phase_icsd = [extract_icsd_from_key(k) for k in phase_keys]
    return phase_keys, phase_cifs, phase_icsd

def extract_cell_parameters(result, flatten_if_single=True):
    """
    Extract lattice parameters (a, b, c, alpha, beta, gamma) per phase from a refinement result.

    Args:
        result: Refinement result (or object with .refinement_result and .lst_data.phases_results).
        flatten_if_single: If True and only one phase exists, return that phase's dict instead of {phase_name: dict}.

    Returns:
        dict | None: Per-phase dict with keys "a", "b", "c", "alpha", "beta", "gamma" (in Å and degrees),
        or a single-phase dict when flatten_if_single=True and there is one phase.
        Returns None when: result has no lst_data or phases_results; no phase has valid a, b, c;
        or an exception occurs (e.g. malformed refinement data). Callers should treat None as
        "cell parameters not available" and skip storing or displaying them.
    """
    try:
        res = getattr(result, "refinement_result", result)
        lst = getattr(res, "lst_data", None)
        if lst is None or not hasattr(lst, "phases_results") or not lst.phases_results:
            return None

        out = {}
        for pname, phase in lst.phases_results.items():
            def val_of(x, default=None):
                if x is None:
                    return default
                if isinstance(x, tuple) and x:
                    return x[0]
                return x
                
            a = val_of(getattr(phase, "a", None))
            b = val_of(getattr(phase, "b", None))
            c = val_of(getattr(phase, "c", None))
            alpha = getattr(phase, "alpha", None)
            beta = val_of(getattr(phase, "beta", None))
            gamma = getattr(phase, "gamma", None)

            if alpha is None:
                alpha = 90.0
            if beta is None:
                beta = 90.0
            if gamma is None:
                gamma = 90.0

            if a is not None and b is not None and c is not None:
                out[pname] = {"a": a, "b": b, "c": c,
                              "alpha": alpha, "beta": beta, "gamma": gamma}

        if not out:
            return None
        if flatten_if_single and len(out) == 1:
            return next(iter(out.values()))
        return out

    except Exception as e:
        print(f"[cell-params] skipped due to error: {e}")
        return None

def evaluate_interpretation(i, pattern_path, search_results, final_refinement_params, target):
    """
    Evaluate a single interpretation result using refinement, peak matching, and scoring.
    Returns a tuple: (interpretation_dict, final_results, isolated_missing_peaks, isolated_extra_peaks)
    """
    peak_list= detect_peaks(
    pattern=pattern_path,
    wavelength="Cu",
    instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
    wmin=final_refinement_params.get("wmin", None),
    wmax=None,
    )
    peak_obs=peak_list[["2theta", "intensity"]].values
    refinement_result = search_results[i].refinement_result
    if refinement_result is not None:
        peak_calc = refinement_result.peak_data[["2theta", "intensity"]].values
        matcher = PeakMatcher(peak_calc, peak_obs)
        score_search = matcher.score(
            matched_coeff=1,
            wrong_intensity_coeff=1,
            missing_coeff=-0.1,
            extra_coeff=-0.5,
            normalize=True,
        )
        print(f"Search result {i}: score = {score_search}")
    else:
        print(f"Search result {i} has no refinement result")

   
    final_results = do_refinement_no_saving( 
        pattern_path=pattern_path,
        phases=[phases_[0] for phases_ in search_results[i].phases],
        phase_params=final_refinement_params,
    )
     
    peak_list= detect_peaks(
        pattern=pattern_path,
        wavelength="Cu",
        instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
        wmin=final_refinement_params.get("wmin", None),
        wmax=None,
    )

    peak_obs=peak_list[["2theta", "intensity"]].values
    
    peak_calc = final_results.peak_data[["2theta", "intensity"]].values
    matcher = PeakMatcher(peak_calc,peak_obs)
    
    score = matcher.score(
        matched_coeff=1,
        wrong_intensity_coeff=1,
        missing_coeff=-0.1,
        extra_coeff=-0.5,
        normalize=True,
    )
    normalized_rwp = normalize_rwp(final_results.lst_data.rwp) 
    
    missing_peaks=matcher.missing
    if  missing_peaks is not None:
        missing_peaks  = np.array(missing_peaks).reshape(-1, 2)
    extra_peaks= matcher.extra
    if extra_peaks is not None:
        extra_peaks = np.array(extra_peaks).reshape(-1, 2)
    isolated_missing_peaks = matcher.get_isolated_peaks(
                    peak_type="missing"
                ).tolist()
    isolated_extra_peaks = matcher.get_isolated_peaks(
                    peak_type="extra"
                ).tolist()

    final_results_phases_list = list(final_results.lst_data.phases_results.keys())
    final_results_phases_list_strip = [
        strip_phase_identifier(p.strip()) for p in final_results.lst_data.phases_results.keys()
    ]

    return final_results, matcher, missing_peaks, isolated_missing_peaks, extra_peaks, isolated_extra_peaks, final_results_phases_list, final_results_phases_list_strip, score, score_search, normalized_rwp


def normalize_rwp(rwp):
    """Normalize RWP to [0,1] scale (RWP 0 -> 1, RWP 40 -> 0)."""
    return (rwp - 40) / (-40)


def main(pattern_path, chemical_system, target):
    """
    Run phase search and refinement for a single XRD pattern and chemical system.

    Fetches CIFs for the chemical system, runs phase search and Rietveld refinement,
    and builds multiple interpretations by iteratively excluding phases (via phase_importance).
    Results are stored in an interpretations dict keyed by "I_1", "I_2", etc., with
    phases, RWP, scores, and metadata.
    """
    all_phases_list = ['None']
    all_search_phases_list = []
    all_search_rwp_list = []
    elements_to_remove = []
    interpretations = {}
    interpretation_counter = 1

    start_time = time.time()
    cleanup_cifs()
    cod_database = ICSDDatabase()
    cod_database.get_cifs_by_chemsys(chemical_system, dest_dir="cifs")
    all_cifs = list(Path("cifs").glob("*.cif"))

    search_tree = search_phases(
        pattern_path=pattern_path,
        phases=all_cifs,
        wavelength="Cu",
        instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
        return_search_tree=True,
    )
    search_results = search_tree.get_search_results()
    final_refinement_params = {
        "gewicht": "SPHAR4",
        "lattice_range": 0.02,
        "k1": "0_0^1",
        "k2": "0_0^0.001",
    }
    
    existing_combinations= set()

    for i in range(len(search_results)):
        print(f"Rwp of solution {i} = {search_results[i].refinement_result.lst_data.rwp} %")
   
    for i in range(len(search_results)):
        final_results, matcher, missing_peaks, isolated_missing_peaks, extra_peaks, isolated_extra_peaks, final_results_phases_list, final_results_phases_list_strip, score, score_search, normalized_rwp  = evaluate_interpretation(i, pattern_path, search_results, final_refinement_params, target)
       
        cell_params = extract_cell_parameters(final_results)
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT_:
            print(f"Time limit of {TIME_LIMIT_} seconds exceeded. Exiting interpretation loop after {i} iterations.")
            break

        final_phase_set = frozenset(final_results_phases_list_strip)
       
        if final_phase_set not in existing_combinations:
            existing_combinations.add(final_phase_set)
            save_xrd_plots(
                search_results, final_results, f"I_{interpretation_counter}", pattern_path, isolated_missing_peaks, isolated_extra_peaks, target
            )
            plot_data = final_results.plot_data
            observed = plot_data.y_obs
            background = plot_data.y_bkg
        
            excess_bkg, normalized_excess = add_flag(background, observed)
            signal_above_bkg = net_signal_score(plot_data)
            signal_score = signal_above_bkg_score(plot_data)
            bkg_score = bkg_overshoot_score(plot_data)
            diff_score = abs_diff_score(plot_data)
            bkg_baseline_score = bkg_baseline_distance_score(plot_data)
            

            phase_weights_dict = final_results.get_phase_weights()
            weight_fraction = [phase_weights_dict[phase] for phase in final_results_phases_list]
            phase_keys, phase_cifs, phase_icsd = phase_assets_from_final(final_results, base_dir="cifs")
            interpretations[f"I_{interpretation_counter}"] = {
                **{
                    "phases": final_results_phases_list,
                    "phase_cifs": phase_cifs,  
                    "phase_icsd": phase_icsd,   
                    "weight_fraction": [x * 100 for x in weight_fraction],
                    "rwp": final_results.lst_data.rwp,
                    "search_result_rwp": search_results[i].refinement_result.lst_data.rwp,
                    "score": score,
                    "search_result_score": score_search,
                    "dara_score": score,
                    "normalized_rwp": normalized_rwp,
                    "missing_peaks": len(isolated_missing_peaks),
                    "extra_peaks": len(isolated_extra_peaks),
                    "peaks_calculated": len(matcher.peak_calc),
                    "peaks_observed": len(matcher.peak_obs),
                    "flag": excess_bkg,
                    "normalized_flag": normalized_excess,
                    "signal_above_bkg": signal_above_bkg,
                    "signal_above_bkg_score": signal_score,
                    "bkg_overshoot_score": bkg_score,
                    "abs_diff_score": diff_score,
                    "bkg_baseline_score": bkg_baseline_score,
                },
                **({"cell_parameters": deepcopy(cell_params)} if cell_params is not None else {}),
            }

            interpretation_counter +=1
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT:
                print(f"Time limit of {TIME_LIMIT} seconds exceeded. Exiting interpretation loop after {i} iterations.")
                break
            

    for i in range(len(search_results)):
        initial_phases = []
        for phases_ in list(search_results[i].refinement_result.get_phase_weights().keys()):
            initial_phases.append(phases_)

        elements_to_remove += initial_phases
    
    final_results_rwp_list = [final_results.lst_data.rwp]
    elements_to_remove = list(set(elements_to_remove))
    final_results_rwp_list, all_search_rwp_list, interpretation_counter, interpretations = phase_importance(
        pattern_path,
        existing_combinations,
        elements_to_remove,
        all_search_phases_list,
        all_search_rwp_list,
        [final_results.lst_data.rwp],
        final_results_phases_list,
        all_cifs,
        interpretation_counter,
        interpretations,
        target
    )
    all_weight_fractions = []
    return (
        all_search_phases_list,
        all_search_rwp_list,
        final_results_phases_list,
        final_results_rwp_list,
        all_phases_list,
        search_results,
        final_results,
        elements_to_remove,
        all_weight_fractions,
        interpretations
    )

def phase_importance(
    pattern_path, existing_combinations, elements_to_remove, all_search_phases_list, all_search_rwp_list,
    final_results_rwp_list, final_results_phases_list, all_cifs, interpretation_counter, interpretations,target
):
    """
    Iteratively explores the impact of excluding certain elements (CIFs) on phase identification via XRD refinement.

    Steps:
    1. In the first round, remove each element individually and run:
    - Phase search
    - Refinement
    - Peak matching (score, missing/extra peaks)
    - Background analysis

    2. In subsequent rounds, evaluate all unique combinations of excluded elements (of any length):
    - Skip previously tested combinations
    - Assess new interpretations with the same process

    Each unique set of resulting phases is stored as a new interpretation, with metadata including RWP, peak scores, and background flags.

    The loop stops if:
    - The number of unique interpretations reaches a cap
    - Too many combinations yield no new interpretations
    - Time limit is exceeded
    """
    first_round = True
    start_time = time.time()
    exit_loop = False
    while True:
        print(f"Starting a new interpretation round with {len(elements_to_remove)} elements to remove...")
        iterations_without_new_interpretation = 0
        elements_to_remove_full = elements_to_remove
        elements_to_remove = remove_cifs_prefix(elements_to_remove)
        elements_to_remove = cleanup_phases(elements_to_remove)
        print("Elements to remove after cleanup:", elements_to_remove)

        if first_round:
            print("In first round ")
            new_elements_to_remove = []

            for element_to_remove in elements_to_remove:
                print(f"Creating interpretation by removing: {element_to_remove}")

                all_cifs_new = remove_elements(all_cifs, [element_to_remove])
                
               
                search_tree = search_phases(
                    pattern_path=pattern_path,
                    phases=all_cifs_new,
                    wavelength="Cu",
                    instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
                    return_search_tree=True,
                )
                search_results = search_tree.get_search_results()
                final_refinement_params = {
                    "gewicht": "SPHAR4",
                    "lattice_range": 0.02,
                    "k1": "0_0^1",
                    "k2": "0_0^0.001",
                }
                final_results = None
                for i in range(len(search_results)):
                    final_results, matcher, missing_peaks, isolated_missing_peaks, extra_peaks, isolated_extra_peaks, final_results_phases_list, final_results_phases_list_strip, score, score_search, normalized_rwp  = evaluate_interpretation(i, pattern_path, search_results, final_refinement_params, target)
                    if not final_results:
                        continue
                    cell_params = extract_cell_parameters(final_results)
                    existing_combinations_set = {frozenset(comb) for comb in existing_combinations}
                    final_results_phase_set = frozenset(final_results_phases_list_strip)
                    if final_results_phase_set not in existing_combinations_set:
                        existing_combinations.add(final_results_phase_set)
                        plot_data = final_results.plot_data
                        observed = plot_data.y_obs
                        background = plot_data.y_bkg
                        
                        excess_bkg, normalized_excess = add_flag(background, observed)
                        signal_above_bkg = net_signal_score(plot_data)
                        signal_score = signal_above_bkg_score(plot_data)
                        bkg_score = bkg_overshoot_score(plot_data)
                        diff_score = abs_diff_score(plot_data)
                        bkg_baseline_score = bkg_baseline_distance_score(plot_data)

                        
                        save_xrd_plots(
                            search_results, final_results, f"I_{interpretation_counter}", pattern_path, isolated_missing_peaks, isolated_extra_peaks,target
                        )
                        weight_fraction = list(final_results.get_phase_weights().values())
                        phase_keys, phase_cifs, phase_icsd = phase_assets_from_final(final_results, base_dir="cifs")
                        interpretations[f"I_{interpretation_counter}"] = {
                            **{
                                "phases": final_results_phases_list,
                                "phase_cifs": phase_cifs,
                                "phase_icsd": phase_icsd, 
                                "weight_fraction": [x * 100 for x in weight_fraction],
                                "rwp": final_results.lst_data.rwp,
                                "search_result_rwp": search_results[i].refinement_result.lst_data.rwp,
                                "score": score,
                                "search_result_score": score_search,
                                "dara_score": score,
                                "normalized_rwp": normalized_rwp,
                                "missing_peaks": len(isolated_missing_peaks),
                                "extra_peaks": len(isolated_extra_peaks),
                                "peaks_calculated": len(matcher.peak_calc),
                                "peaks_observed": len(matcher.peak_obs),
                                "flag": excess_bkg,
                                "normalized_flag": normalized_excess,
                                "signal_above_bkg": signal_above_bkg,
                                "signal_above_bkg_score": signal_score,
                                "bkg_overshoot_score": bkg_score,
                                "abs_diff_score": diff_score,
                                "bkg_baseline_score": bkg_baseline_score,
                            },
                            **({"cell_parameters": deepcopy(cell_params)} if cell_params is not None else {}),
                        }
                        new_elements_to_remove += list(final_results.lst_data.phases_results.keys())
                        iterations_without_new_interpretation =0
                        interpretation_counter +=1
                    else:
                        iterations_without_new_interpretation += 1
                       
                for result in search_results:
                    for phases_ in result.phases:
                        if isinstance(phases_, (list, tuple)):
                            new_elements_to_remove += [
                                phase.path.name for phase in phases_ if hasattr(phase, "path")
                            ]
                        elif hasattr(phases_, "path"):
                            new_elements_to_remove.append(phases_.path.name)

                    phase_weights_dict = result.refinement_result.get_phase_weights()
                    initial_phases = list(phase_weights_dict.keys())
                    weight_fraction = [phase_weights_dict[phase] for phase in initial_phases]

                if interpretation_counter >= 10 or iterations_without_new_interpretation >= MAX_ITER_WITHOUT_NEW_INTERPRETATION:
                    print(f"No unique interpretations found HERE after {MAX_ITER_WITHOUT_NEW_INTERPRETATION} iterations. Exiting loop.")
                    break
                first_round = False 
                elapsed_time = time.time() - start_time
                if elapsed_time > TIME_LIMIT:
                    print(f"Time limit of {TIME_LIMIT} seconds exceeded. Exiting loop after {interpretation_counter} iterations.")
                    exit_loop = True
                    break
              
            if exit_loop:
                break

            new_elements_to_remove = remove_cifs_suffix(new_elements_to_remove)
            elements_to_remove_full = list(set(elements_to_remove_full + new_elements_to_remove))
            new_elements_to_remove = cleanup_phases(new_elements_to_remove)
            elements_to_remove = list(set(elements_to_remove + new_elements_to_remove))

            all_combinations = []
            for r in range(1, len(elements_to_remove_full) + 1):
                all_combinations.extend(combinations(elements_to_remove_full, r))
            existing_combinations_sorted = [sorted(combo) for combo in existing_combinations]
            print("Here existing combinations: ", existing_combinations_sorted)
            remaining_combinations = [
                list(combo)
                for combo in all_combinations
                if sorted(combo) not in existing_combinations_sorted
            ]
            print("remaining combinations ", remaining_combinations)
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT or not remaining_combinations:
                    print(f"Time limit of {TIME_LIMIT} seconds exceeded. Exiting loop after {interpretation_counter} iterations.")
                    exit_loop = True
                    break
            
        else:
            iterations_without_new_interpretation = 0 
            for j in range(len(remaining_combinations)):
                all_cifs_new = include_elements(all_cifs, remaining_combinations[j])
                search_tree = search_phases(
                    pattern_path=pattern_path,
                    phases=all_cifs_new,
                    wavelength="Cu",
                    instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
                    return_search_tree=True,
                )
                search_results = search_tree.get_search_results()

                if search_results:
                    best_result = search_results[0]
                    final_refinement_params = {
                        "gewicht": "SPHAR4",
                        "lattice_range": 0.02,
                        "k1": "0_0^1",
                        "k2": "0_0^0.001",
                    }
                    final_results, matcher, missing_peaks, isolated_missing_peaks, extra_peaks, isolated_extra_peaks, final_results_phases_list, final_results_phases_list_strip, score, score_search, normalized_rwp = evaluate_interpretation(0, pattern_path, search_results, final_refinement_params, target)
                    if not final_results:
                        continue
                    cell_params = extract_cell_parameters(final_results)
                    final_results_phases_list_strip = [
                            strip_phase_identifier(p.strip()) for p in final_results.lst_data.phases_results.keys()
                        ]

                    is_unique = frozenset(final_results_phases_list_strip) not in existing_combinations
                    if is_unique:
                        existing_combinations.add(frozenset(final_results_phases_list_strip))
                        plot_data = final_results.plot_data
                        observed = plot_data.y_obs
                        background = plot_data.y_bkg
                        excess_bkg, normalized_excess = add_flag(background, observed)
                        signal_above_bkg = net_signal_score(plot_data)
                        signal_score = signal_above_bkg_score(plot_data)
                        bkg_score = bkg_overshoot_score(plot_data)
                        diff_score = abs_diff_score(plot_data)
                        bkg_baseline_score = bkg_baseline_distance_score(plot_data)
            
                
                        save_xrd_plots(
                            search_results, final_results, f"I_{interpretation_counter}", pattern_path, isolated_missing_peaks, isolated_extra_peaks, target
                        )
                        weight_fraction = list(final_results.get_phase_weights().values())
                        phase_keys, phase_cifs, phase_icsd = phase_assets_from_final(final_results, base_dir="cifs")
                        normalized_rwp = normalize_rwp(final_results.lst_data.rwp)
                        interpretations[f"I_{interpretation_counter}"] = {
                            **{
                                "phases": final_results_phases_list,
                                "phase_cifs": phase_cifs,
                                "phase_icsd": phase_icsd,
                                "weight_fraction": [x * 100 for x in weight_fraction],
                                "rwp": final_results.lst_data.rwp,
                                "search_result_rwp": best_result.refinement_result.lst_data.rwp,
                                "score": score,
                                "search_result_score": score_search,
                                "dara_score": score,
                                "normalized_rwp": normalized_rwp,
                                "missing_peaks": len(isolated_missing_peaks),
                                "extra_peaks": len(isolated_extra_peaks),
                                "peaks_calculated": len(matcher.peak_calc),
                                "peaks_observed": len(matcher.peak_obs),
                                "flag": excess_bkg,
                                "normalized_flag": normalized_excess,
                                "signal_above_bkg": signal_above_bkg,
                                "signal_above_bkg_score": signal_score,
                                "bkg_overshoot_score": bkg_score,
                                "abs_diff_score": diff_score,
                                "bkg_baseline_score": bkg_baseline_score,
                            },
                            **({"cell_parameters": deepcopy(cell_params)} if cell_params is not None else {}),
                        }
                        interpretation_counter += 1
                        iterations_without_new_interpretation = 0
                        if interpretation_counter >=10:
                            break
                    else:
                        iterations_without_new_interpretation += 1
                        if iterations_without_new_interpretation >= MAX_ITER_WITHOUT_NEW_INTERPRETATION:
                            print(f"No unique interpretations found after {MAX_ITER_WITHOUT_NEW_INTERPRETATION} iterations. Exiting loop.")
                            break
                    elements_to_remove = list(set(elements_to_remove + new_elements_to_remove))
                    elapsed_time = time.time() - start_time
                    if elapsed_time > TIME_LIMIT:
                        print(f"Time limit of {TIME_LIMIT} seconds exceeded. Exiting loop after {j} iterations.")
                        exit_loop = True
                        break
                    
                if exit_loop:
                    break
            if exit_loop:
                break
                            
                
            if j == len(remaining_combinations)-1 or interpretation_counter >=10 or iterations_without_new_interpretation >= MAX_ITER_WITHOUT_NEW_INTERPRETATION:
                print("No more combinations to process. Exiting loop.", interpretation_counter)
                return final_results_rwp_list, all_search_rwp_list, interpretation_counter, interpretations
            if elapsed_time > TIME_LIMIT:
                    print(f"Time limit of {TIME_LIMIT} seconds exceeded. Exiting loop after {j} iterations.")
                    return final_results_rwp_list, all_search_rwp_list, interpretation_counter, interpretations
    return final_results_rwp_list, all_search_rwp_list, interpretation_counter, interpretations
   
def calculate_spectrum_likelihood_given_interpretation_wrapper(pattern_path, chemical_system,target, alpha=1):
    """
    Wrapper to calculate P(S | I_n) for a given pattern path and chemical system.

    Parameters:
        pattern_path (str): Path to the XRD pattern file.
        chemical_system (str): Chemical system (e.g., "Ca-Ag-O-C").
        alpha (float): Weighting factor for statistical importance.

    Returns:
        dict: Results including P(S | I_n) and intermediate data for further processing.
    """
    cleanup_cifs()
    (
        all_search_phases_list,
        all_search_rwp_list,
        final_results_phases_list,
        final_results_rwp_list,
        all_phases_list,
        search_results,
        final_results,
        elements_to_remove,
        all_weight_fractions,
        interpretations
    ) = main(pattern_path, chemical_system, target)

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
    interpretations = importance_factor_calculation(interpretations)
    return interpretations, project_number, target
 