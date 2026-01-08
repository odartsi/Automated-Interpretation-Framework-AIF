import json
import itertools
import numpy as np

# Load your interpretations JSON file
with open('../data/xrd_data/interpretations/interpretations_recomputed_test_new_llm6.json', 'r') as f:
# with open("../src/train_new6.json", 'r') as f:
    data = json.load(f)

# Chemist's selections for specific samples
preferences = {
    "TRI_91": "I_1", #✅ 
    "TRI_63": "I_1",
    "TRI_64": "I_1",#✅ 
    "TRI_11": "I_1",#✅ 
    "PG_1651_1": "I_1",
    "TRI_28": "I_2",#✅ 
    "TRI_197": "I_2",#✅ 
    "TRI_104": "I_2",#✅ 
    "TRI_113": "I_9",#✅ 
    "TRI_111": "I_10"#✅ 
}
#   w_llm     = 0.5
#   w_bscore  = 0.7
#   w_rwp     = 0.2
#   w_score   = 0.1
#   Objective = 159
# Functions to compute prior, fit_quality, and unnormalized posterior
def calc_prior(interp, w_llm, w_bscore):
    return (
        interp['LLM_interpretation_likelihood'] * w_llm +
        interp['balance_score']               * w_bscore
    ) / (w_llm + w_bscore)

def calc_fit(interp, w_rwp, w_score):
    return (
        interp['normalized_rwp']   * w_rwp +
        interp['normalized_score'] * w_score
    ) / (w_rwp + w_score)

def unnormalized_posterior(interp, w_llm, w_bscore, w_rwp, w_score):
    prior = calc_prior(interp, w_llm, w_bscore)
    fit   = calc_fit(interp, w_rwp, w_score)
    return prior * fit

# 1) Baseline argmax (desired for non-preference samples)
#all weights =1
# baseline_argmax = {
#     sid: max(
#         {name: unnormalized_posterior(vals, 1, 1, 1, 1)
#          for name, vals in interps.items()},
#         key=lambda k: unnormalized_posterior(interps[k], 1, 1, 1, 1)
#     )
#     for sid, interps in data.items()
# }
# NEW: use the stored JSON field
baseline_argmax = {
    sid: max(
        interps.items(),
        key=lambda kv: kv[1]['unnormalized_posterior']
    )[0]
    for sid, interps in data.items()
}

# 2) Grid search over weights in [0,1] × [0,1] × [0,1] × [0,1] with step 0.1
weights = np.linspace(0.0, 1.0, 11)
best_score = -np.inf
best_weights = None
for w_llm, w_bscore, w_rwp, w_score in itertools.product(weights, repeat=4):
    if (w_llm + w_bscore) == 0 or (w_rwp + w_score) == 0:
        continue
    score = 0
    # reward matching chemist's preferences heavily
    for sid, desired in preferences.items():
        ups = {name: unnormalized_posterior(vals, w_llm, w_bscore, w_rwp, w_score)
               for name, vals in data[sid].items()}
        if max(ups, key=ups.get) == desired:
            score += 10
    # reward consistency on all other samples lightly
    for sid in data:
        if sid in preferences:
            continue
        ups = {name: unnormalized_posterior(vals, w_llm, w_bscore, w_rwp, w_score)
               for name, vals in data[sid].items()}
        if max(ups, key=ups.get) == baseline_argmax[sid]:
            score += 1
    if score > best_score:
        best_score = score
        best_weights = (w_llm, w_bscore, w_rwp, w_score)
# 6) Record objective for every weight combination
all_scores = []
for w_llm, w_bscore, w_rwp, w_score in itertools.product(weights, repeat=4):
    if (w_llm + w_bscore) == 0 or (w_rwp + w_score) == 0:
        continue
    score = 0
    for sid, desired in preferences.items():
        ups = {
            name: unnormalized_posterior(vals, w_llm, w_bscore, w_rwp, w_score)
            for name, vals in data[sid].items()
        }
        if max(ups, key=ups.get) == desired:
            score += 10
    for sid in data:
        if sid in preferences:
            continue
        ups = {
            name: unnormalized_posterior(vals, w_llm, w_bscore, w_rwp, w_score)
            for name, vals in data[sid].items()
        }
        if max(ups, key=ups.get) == baseline_argmax[sid]:
            score += 1
    # all_scores.append((w_llm, w_bscore, w_rwp, w_score, score))
    all_scores.append((
        round(w_llm, 1),
        round(w_bscore, 1),
        round(w_rwp, 1),
        round(w_score, 1),
        score
    ))

# 7) Print all weight‐sets sorted by objective (best→worst)
# print("All weight combinations ranked by objective score:")
# for wllm, wb, wr, ws, obj in sorted(all_scores, key=lambda x: -x[4]):
#     print(f"w_llm={wllm:.1f}, w_bscore={wb:.1f}, w_rwp={wr:.1f}, w_score={ws:.1f}  ->  Objective={int(obj)}")
# 8) List every weight‐set that ties the best objective
print(f"\nWeight combinations achieving the best objective = {best_score}:")
for wllm, wb, wr, ws, obj in all_scores:
    if obj == best_score:
        print(f"  w_llm={wllm:.1f}, w_bscore={wb:.1f}, w_rwp={wr:.1f}, w_score={ws:.1f}")
# after you build all_scores = [(w_llm, w_bscore, w_rwp, w_score, score), ...]
# 8) Lookup your manual combo exactly
target = (0.5, 0.7, 0.9, 0.4)
target = (0.5, 0.7, 1, 0.5)
manual_obj = next(
    obj for w1, w2, w3, w4, obj in all_scores 
    if (w1, w2, w3, w4) == target
)
print(f"\nObjective for manual weights {target}: {manual_obj}")

# 3) Determine new winner under best weights
w_llm, w_bscore, w_rwp, w_score = best_weights
chosen_map = {}
for sid, interps in data.items():
    ups = {name: unnormalized_posterior(vals, *best_weights)
           for name, vals in interps.items()}
    chosen_map[sid] = max(ups, key=ups.get)

# Print the optimal weights
print("Best weights found:")
print(f"  w_llm     = {w_llm:.1f}")
print(f"  w_bscore  = {w_bscore:.1f}")
print(f"  w_rwp     = {w_rwp:.1f}")
print(f"  w_score   = {w_score:.1f}")
print(f"  Objective = {best_score}\n")

# ANSI color codes
RED = '\033[91m'
RESET = '\033[0m'

# ... [all the code up to the printing stage remains the same] ...

# 4) Print ALL samples with red highlight if new != previous
print("All samples — selected interpretation, unnormalized posterior, previous interpretation:")
for sid in sorted(data):
    new = chosen_map[sid]
    posterior_val = unnormalized_posterior(data[sid][new], *best_weights)
    prev = baseline_argmax[sid]
    line = f"  {sid}: {new} (unnormalized posterior = {posterior_val:.4f}), previous = {prev}"
    if new != prev:
        line = RED + line + RESET
    print(line)

# 5) Print preference samples with red highlight if new != desired
print("\nPreference samples — selected interpretation, unnormalized posterior, desired:")
for sid, desired in preferences.items():
    new = chosen_map[sid]
    posterior_val = unnormalized_posterior(data[sid][new], *best_weights)
    line = f"  {sid}: {new} (unnormalized posterior = {posterior_val:.4f}), desired = {desired}"
    if new != desired:
        line = RED + line + RESET
    print(line)

#
# --- Manual evaluation for fixed weights based on full data set ---
manual_weights = (0.5, 0.5, 1, 0.5)
# manual_weights = (0.5, 0.7, 0.9, 0.4)
# --- Manual evaluation for fixed weights based on train data set ---
# manual_weights = (0.5, 0.7, 1, 0.5)
# manual_weights = (0.5, 0.7, 0.9, 0.4) 
mw_llm, mw_bscore, mw_rwp, mw_score = manual_weights


# Compute manual objective score
manual_score = 0
for sid, desired in preferences.items():
    ups = {
        name: unnormalized_posterior(vals, mw_llm, mw_bscore, mw_rwp, mw_score)
        for name, vals in data[sid].items()
    }
    if max(ups, key=ups.get) == desired:
        manual_score += 10

for sid in data:
    if sid in preferences:
        continue
    ups = {
        name: unnormalized_posterior(vals, mw_llm, mw_bscore, mw_rwp, mw_score)
        for name, vals in data[sid].items()
    }
    if max(ups, key=ups.get) == baseline_argmax[sid]:
        manual_score += 1

# Determine manual chosen_map
manual_chosen = {}
for sid, interps in data.items():
    ups = {
        name: unnormalized_posterior(vals, *manual_weights)
        for name, vals in interps.items()
    }
    manual_chosen[sid] = max(ups, key=ups.get)

# Print manual weights and objective
print("\nManual weights evaluation (0.5, 0.5, 1, 0.5):")
print(f"  Objective = {manual_score}\n")

# 1) All samples under manual weights
print("All samples (manual) — selected interpretation, unnormalized posterior, previous interpretation:")
for sid in sorted(data):
    new = manual_chosen[sid]
    post = unnormalized_posterior(data[sid][new], *manual_weights)
    prev = baseline_argmax[sid]
    line = f"  {sid}: {new} (unnormalized posterior = {post:.4f}), previous = {prev}"
    if new != prev:
        line = RED + line + RESET
    print(line)

# 2) Preference samples under manual weights
print("\nPreference samples (manual) — selected interpretation, unnormalized posterior, desired:")
for sid, desired in preferences.items():
    new = manual_chosen[sid]
    post = unnormalized_posterior(data[sid][new], *manual_weights)
    line = f"  {sid}: {new} (unnormalized posterior = {post:.4f}), desired = {desired}"
    if new != desired:
        line = RED + line + RESET
    print(line)