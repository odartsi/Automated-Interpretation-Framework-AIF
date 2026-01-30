import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your JSON file (update to your actual path)
JSON_PATH ='../data/xrd_data/interpretations/interpretations_recomputed_test_new_llm6.json'
# JSON_PATH ="../src/train_new6.json"

# Chemist’s preferred interpretations
PREFERENCES = {
    "TRI_91":    "I_1",
    "TRI_63":    "I_1",
    "TRI_64":    "I_1",
    "TRI_11":    "I_1",
    "PG_1651_1": "I_1",
    "TRI_28":    "I_2",
    "TRI_197":   "I_2",
    "TRI_104":   "I_2",
    "TRI_113":   "I_9",
    "TRI_111":   "I_10"
}

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
    return calc_prior(interp, w_llm, w_bscore) * calc_fit(interp, w_rwp, w_score)

# 1) Load JSON
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# 2) Baseline winners from stored unnormalized_posterior
baseline = {
    sid: max(interps.items(), key=lambda kv: kv[1]['posterior_probability'])[0]
    for sid, interps in data.items()
}

# 3) Grid-search weights
weights = np.linspace(0, 1, 11)
results = []
total_prefs = len(PREFERENCES)

for w_llm, w_bscore, w_rwp, w_score in itertools.product(weights, repeat=4):
    if (w_llm + w_bscore) == 0 or (w_rwp + w_score) == 0:
        continue

    # determine winners
    chosen = {
        sid: max(
            interps.items(),
            key=lambda kv: unnormalized_posterior(kv[1], w_llm, w_bscore, w_rwp, w_score)
        )[0]
        for sid, interps in data.items()
    }

    # which prefs matched
    matched   = [sid for sid, des in PREFERENCES.items() if chosen[sid] == des]
    unmatched = [sid for sid in PREFERENCES if sid not in matched]

    # which non-prefs changed
    changed   = [
        sid for sid in data
        if sid not in PREFERENCES and chosen[sid] != baseline[sid]
    ]

    results.append({
        'w_llm':        w_llm,
        'w_bscore':     w_bscore,
        'w_rwp':        w_rwp,
        'w_score':      w_score,
        # 'matches':      ','.join(matched),
        'unmatches from prefered':    ','.join(unmatched),
        'changed':      ','.join(changed),
        'changed_count': len(changed),
        'matched_ratio': len(matched) / total_prefs
    })

# 4) Build DataFrame and print top 20 by matched_ratio
df = pd.DataFrame(results)
# df_sorted = df.sort_values('matched_ratio', ascending=False).head(60)
df_sorted = (
    df
    .sort_values(['matched_ratio','changed_count'], ascending=[False, True])
    .head(20)
)
# print(
#     df_sorted[
#         ['w_llm','w_bscore','w_rwp','w_score','matches','unmatches','changed','changed_count','matched_ratio']
#     ].to_markdown(index=False)
# )
print(
    df_sorted[
        ['w_llm','w_bscore','w_rwp','w_score','unmatches from prefered','matched_ratio','changed','changed_count']
    ].to_markdown(index=False)
)

# --- NEW: Table for w_score > 0.6 ---
df_score_high = (
    df[df['w_score'] > 0.6]
    .sort_values(['changed_count','matched_ratio'], ascending=[True, False])
    .head(20)
    .reset_index(drop=True)
)

print("\n## Weight Sets with w_score > 0.6 ##\n")
print(
    df_score_high[
        ['w_llm','w_bscore','w_rwp','w_score',
         'unmatches from prefered','matched_ratio','changed','changed_count']
    ].to_markdown(index=False)
)