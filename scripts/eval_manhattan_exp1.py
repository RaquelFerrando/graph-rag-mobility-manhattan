"""
Validation metrics for the Manhattan Experiment I run: is the mode/duration
choice pipeline producing a population-level distribution that resembles real
Manhattan trips, and is that better than trivially guessing the majority mode?

Deliberately NOT the paper's with/without-graph ablation - that answers "does
the graph help", not "is this reliable enough to present". Pure pandas, no LLM
calls, runs in seconds against the already-saved results/manhattan/exp1_test_1000.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from baseline.eval import cal_group_kl_divergence

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
RESULTS_FILE = "results/manhattan/exp1_test_1000.json"
N_ROWS = 1000


def aggregate_kl_mae(true_col: pd.Series, pred_col: pd.Series):
    """KLD/MAE between two categorical columns' aggregate distributions (not per-row match)."""
    true_dist = true_col.value_counts(normalize=True)
    pred_dist = pred_col.value_counts(normalize=True)
    all_values = sorted(set(true_dist.index) | set(pred_dist.index))
    p = np.array([true_dist.get(v, 0.0) for v in all_values])
    q = np.array([pred_dist.get(v, 0.0) for v in all_values])
    mae = np.mean(np.abs(p - q))
    smooth = 1e-10
    p = (p + smooth) / (p + smooth).sum()
    q = (q + smooth) / (q + smooth).sum()
    return entropy(p, q), mae


print(f"Loading eval data from {EVAL_FILE}...")
eval_df = load_data(EVAL_FILE)
# same random_state=42 / same sample size as the run -> identical row order
sample_df = eval_df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)

results = json.load(open(RESULTS_FILE))
print(f"Loaded {len(results)} results from {RESULTS_FILE}")

# Sanity check: merge is positional, so true_mode/true_duration from the results
# file must match primary_mode/duration_minutes from the reconstructed sample
# for every row - confirms the row order truly lines up before trusting anything below.
n = len(results)
sample_df = sample_df.iloc[:n].copy()
mismatches = 0
for i, r in enumerate(results):
    if r['true_mode'] != sample_df.loc[i, 'primary_mode'] or r['true_duration'] != sample_df.loc[i, 'duration_minutes']:
        mismatches += 1
if mismatches:
    print(f"WARNING: {mismatches}/{n} rows do not line up - merge is NOT safe, aborting.")
    sys.exit(1)
print(f"Row-order sanity check passed: all {n} rows match true_mode/true_duration.\n")

sample_df['predicted_mode'] = [r['predicted_mode'] for r in results]
sample_df['predicted_duration'] = [r['predicted_duration'] for r in results]
sample_df['choice_weights'] = [r['choice_weights'] for r in results]

# ---- 1. Aggregate population-level KLD/MAE: method vs. real distribution ----
mode_kl, mode_mae = aggregate_kl_mae(sample_df['primary_mode'], sample_df['predicted_mode'])
dur_kl, dur_mae = aggregate_kl_mae(sample_df['duration_minutes'], sample_df['predicted_duration'])

print("=" * 70)
print("1. AGGREGATE distribution match (method vs. real Manhattan trips)")
print("=" * 70)
print(f"  mode     KLD={mode_kl:.4f}  MAE={mode_mae:.4f}")
print(f"  duration KLD={dur_kl:.4f}  MAE={dur_mae:.4f}")

# ---- 2. Trivial baseline: always predict the single most common real mode/duration ----
majority_mode = sample_df['primary_mode'].mode()[0]
majority_duration = sample_df['duration_minutes'].mode()[0]
baseline_mode_pred = pd.Series([majority_mode] * n)
baseline_duration_pred = pd.Series([majority_duration] * n)
base_mode_kl, base_mode_mae = aggregate_kl_mae(sample_df['primary_mode'], baseline_mode_pred)
base_dur_kl, base_dur_mae = aggregate_kl_mae(sample_df['duration_minutes'], baseline_duration_pred)

print()
print("=" * 70)
print(f"2. TRIVIAL baseline (always predict majority: mode='{majority_mode}', duration='{majority_duration}')")
print("=" * 70)
print(f"  mode     KLD={base_mode_kl:.4f}  MAE={base_mode_mae:.4f}")
print(f"  duration KLD={base_dur_kl:.4f}  MAE={base_dur_mae:.4f}")

print()
print("=" * 70)
print("3. Method vs. trivial baseline (lower is better)")
print("=" * 70)
print(f"  mode     KLD: method={mode_kl:.4f} vs baseline={base_mode_kl:.4f}  "
      f"({'method wins' if mode_kl < base_mode_kl else 'baseline wins'})")
print(f"  mode     MAE: method={mode_mae:.4f} vs baseline={base_mode_mae:.4f}  "
      f"({'method wins' if mode_mae < base_mode_mae else 'baseline wins'})")
print(f"  duration KLD: method={dur_kl:.4f} vs baseline={base_dur_kl:.4f}  "
      f"({'method wins' if dur_kl < base_dur_kl else 'baseline wins'})")
print(f"  duration MAE: method={dur_mae:.4f} vs baseline={base_dur_mae:.4f}  "
      f"({'method wins' if dur_mae < base_dur_mae else 'baseline wins'})")

# ---- 4. Paper's own metric: per-demographic-subgroup KLD/MAE, averaged ----
kl_df, overall_kl, overall_mae = cal_group_kl_divergence(result_df=sample_df)
print()
print("=" * 70)
print("4. Paper's own metric: per-subgroup KLD/MAE (averaged across subgroups)")
print("=" * 70)
print(f"  overall_kl={overall_kl:.4f}  overall_mae={overall_mae:.4f}")
print(f"\n  Worst 5 subgroups by KL divergence:")
print(kl_df.sort_values('kl_divergence', ascending=False).head(5).to_string(index=False))

out_summary = {
    'n_rows': n,
    'aggregate': {
        'mode': {'kld': mode_kl, 'mae': mode_mae},
        'duration': {'kld': dur_kl, 'mae': dur_mae},
    },
    'trivial_baseline': {
        'majority_mode': majority_mode,
        'majority_duration': majority_duration,
        'mode': {'kld': base_mode_kl, 'mae': base_mode_mae},
        'duration': {'kld': base_dur_kl, 'mae': base_dur_mae},
    },
    'paper_metric_per_subgroup': {'overall_kl': overall_kl, 'overall_mae': overall_mae},
}
out_path = "results/manhattan/exp1_validation_summary.json"
json.dump(out_summary, open(out_path, "w"), indent=2)
print(f"\nSaved summary to {out_path}")
