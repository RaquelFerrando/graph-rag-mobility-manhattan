"""
Distribution comparison chart for Manhattan Experiment I, matching the style of
the paper's own Figure 3 (Ground Truth vs Preference Chain bars for mode and
duration) - but only 2 columns, not 3, since we deliberately skipped the
no-reference LLM baseline (see plan: the ablation wasn't the point, the
population-level realism was).
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
RESULTS_FILE = "results/manhattan/exp1_test_1000.json"
N_ROWS = 1000

eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)
results = json.load(open(RESULTS_FILE))
n = len(results)
sample_df = sample_df.iloc[:n].copy()
sample_df['predicted_mode'] = [r['predicted_mode'] for r in results]
sample_df['predicted_duration'] = [r['predicted_duration'] for r in results]

DURATION_ORDER = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
mode_order = sample_df['primary_mode'].value_counts(normalize=True).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

def bar(ax, series, order, title, color):
    dist = series.value_counts(normalize=True).reindex(order, fill_value=0)
    ax.bar(range(len(order)), dist.values, color=color)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylim(0, max(0.65, dist.values.max() * 1.15))

bar(axes[0, 0], sample_df['primary_mode'], mode_order, 'Ground Truth', '#4C72B0')
bar(axes[0, 1], sample_df['predicted_mode'], mode_order, 'Preference Chain (Manhattan)', '#DD8452')
bar(axes[1, 0], sample_df['duration_minutes'], DURATION_ORDER, 'Ground Truth', '#4C72B0')
bar(axes[1, 1], sample_df['predicted_duration'], DURATION_ORDER, 'Preference Chain (Manhattan)', '#DD8452')

axes[0, 0].set_ylabel('Transportation Mode')
axes[1, 0].set_ylabel('Duration Minutes')
fig.suptitle('Manhattan Experiment I: real vs. simulated distributions (n=1000)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = "results/manhattan/exp1_distribution_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
