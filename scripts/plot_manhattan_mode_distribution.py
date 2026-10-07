"""
Mode-only version of plot_manhattan_distributions.py's top row, for the README:
the 2x2 chart (mode + duration) reads cluttered at the small size a README needs,
and duration is the harder/less central result, so this isolates just the mode
comparison.
"""
import json
import os
import sys

import matplotlib.pyplot as plt

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

mode_order = sample_df['primary_mode'].value_counts(normalize=True).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

def bar(ax, series, order, title, color):
    dist = series.value_counts(normalize=True).reindex(order, fill_value=0)
    ax.bar(range(len(order)), dist.values, color=color)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylim(0, max(0.65, dist.values.max() * 1.15))

bar(axes[0], sample_df['primary_mode'], mode_order, 'Ground Truth', '#4C72B0')
bar(axes[1], sample_df['predicted_mode'], mode_order, 'Preference Chain (Manhattan)', '#DD8452')

axes[0].set_ylabel('Share of trips')
fig.suptitle('Manhattan Experiment I: transportation mode, real vs. simulated (n=1000)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = "results/manhattan/exp1_mode_distribution.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
