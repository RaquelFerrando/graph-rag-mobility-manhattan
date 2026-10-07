"""
Deep-dive analysis on top of eval_manhattan_exp1.py's aggregate numbers, aimed
at building a narrative for a non-technical audience: not just "how good is
it" (already answered) but "where exactly does it fail, and does that failure
matter."

Four pieces:
  1. Mode confusion matrix - which real mode gets mistaken for which predicted
     mode, so we can say e.g. "when it's wrong, it's usually predicting public
     transit instead of walking" rather than just a bare accuracy number.
  2. Full per-subgroup KLD/MAE table (not just top-5) - is the worst subgroup
     bad because of a real bias or just small sample size (sample_size column
     tells us that directly).
  3. Concrete error examples - pick actual rows where the model was
     confidently wrong (predicted mode got high weight, true mode got ~0), to
     have real stories instead of abstract percentages.
  4. Duration confusion matrix - same idea as (1), for duration since it's
     the weaker of the two (KLD 0.12 vs 0.06).

Pure pandas over already-saved results/manhattan/exp1_test_1000.json - no LLM
calls, runs in seconds.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from baseline.eval import cal_group_kl_divergence

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
RESULTS_FILE = "results/manhattan/exp1_test_1000.json"
N_ROWS = 1000

print(f"Loading eval data from {EVAL_FILE}...")
eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)

results = json.load(open(RESULTS_FILE))
n = len(results)
sample_df = sample_df.iloc[:n].copy()

for i, r in enumerate(results):
    assert r['true_mode'] == sample_df.loc[i, 'primary_mode']
    assert r['true_duration'] == sample_df.loc[i, 'duration_minutes']

sample_df['predicted_mode'] = [r['predicted_mode'] for r in results]
sample_df['predicted_duration'] = [r['predicted_duration'] for r in results]
sample_df['choice_weights'] = [r['choice_weights'] for r in results]

# ---- 1. Mode confusion matrix ----
print("\n" + "=" * 70)
print("1. MODE CONFUSION MATRIX (rows=real, cols=predicted, counts)")
print("=" * 70)
mode_order = sorted(set(sample_df['primary_mode']) | set(sample_df['predicted_mode']))
confusion = pd.crosstab(sample_df['primary_mode'], sample_df['predicted_mode'])
confusion = confusion.reindex(index=mode_order, columns=mode_order, fill_value=0)
print(confusion.to_string())

print("\nPer-real-mode: % correctly predicted, and top confusion target when wrong")
rows = []
for real_mode in mode_order:
    row = confusion.loc[real_mode]
    total = row.sum()
    if total == 0:
        continue
    correct = row.get(real_mode, 0)
    wrong = row.drop(real_mode, errors='ignore')
    top_confusion = wrong.idxmax() if wrong.sum() > 0 else None
    top_confusion_pct = wrong.max() / total if wrong.sum() > 0 else 0
    rows.append({
        'real_mode': real_mode, 'n': total,
        'pct_correct': correct / total,
        'top_confused_with': top_confusion,
        'pct_confused_with_top': top_confusion_pct,
    })
confusion_summary = pd.DataFrame(rows).sort_values('n', ascending=False)
print(confusion_summary.to_string(index=False))

# ---- 2. Full per-subgroup KLD/MAE table ----
print("\n" + "=" * 70)
print("2. FULL per-subgroup KLD/MAE (sorted worst first)")
print("=" * 70)
kl_df, overall_kl, overall_mae = cal_group_kl_divergence(result_df=sample_df)
kl_df_sorted = kl_df.sort_values('kl_divergence', ascending=False)
print(kl_df_sorted.to_string(index=False))

# ---- 3. Concrete error examples: confidently wrong predictions ----
print("\n" + "=" * 70)
print("3. CONCRETE ERROR EXAMPLES (model confidently predicted the wrong mode)")
print("=" * 70)
error_examples = []
for i, r in enumerate(results):
    if r['predicted_mode'] == r['true_mode']:
        continue
    try:
        weights = json.loads(r['choice_weights']) if isinstance(r['choice_weights'], str) else r['choice_weights']
    except Exception:
        continue
    # each choice_weights entry is a joint (primary_mode, duration_minutes)
    # pair with its own weight - marginalize over duration to get the mode's
    # total weight (sum of weights across all its duration buckets).
    true_weight = sum(w.get('weight', 0.0) for w in weights if w.get('primary_mode') == r['true_mode'])
    pred_weight = sum(w.get('weight', 0.0) for w in weights if w.get('primary_mode') == r['predicted_mode'])
    error_examples.append({
        'idx': i, 'true_mode': r['true_mode'], 'predicted_mode': r['predicted_mode'],
        'true_mode_weight': true_weight, 'pred_mode_weight': pred_weight,
        'confidence_gap': pred_weight - true_weight,
        'age_group': sample_df.loc[i, 'age_group'], 'trip_purpose': sample_df.loc[i, 'trip_purpose'],
        'start_time': sample_df.loc[i, 'start_time'],
    })
error_df = pd.DataFrame(error_examples).sort_values('confidence_gap', ascending=False)
print(f"Total wrong predictions: {len(error_df)}/{n}")
print("\nTop 10 most confidently wrong:")
print(error_df.head(10).to_string(index=False))

# ---- 4. Duration confusion matrix ----
print("\n" + "=" * 70)
print("4. DURATION CONFUSION MATRIX (rows=real, cols=predicted, counts)")
print("=" * 70)
dur_order = sorted(set(sample_df['duration_minutes']) | set(sample_df['predicted_duration']),
                    key=lambda x: (int(x.split('-')[0]) if '-' in str(x) and x.split('-')[0].isdigit() else 999))
dur_confusion = pd.crosstab(sample_df['duration_minutes'], sample_df['predicted_duration'])
dur_confusion = dur_confusion.reindex(index=dur_order, columns=dur_order, fill_value=0)
print(dur_confusion.to_string())

out = {
    'mode_confusion_summary': confusion_summary.to_dict(orient='records'),
    'subgroup_kld_full': kl_df_sorted.to_dict(orient='records'),
    'top_error_examples': error_df.head(20).to_dict(orient='records'),
}
out_path = "results/manhattan/exp1_deepdive.json"
json.dump(out, open(out_path, "w"), indent=2, default=str)
print(f"\nSaved deep-dive details to {out_path}")
