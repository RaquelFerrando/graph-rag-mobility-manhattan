"""
Per-mode discriminative power of the graph's choice_weights: does a higher weight for
mode X actually predict that X was the real true_mode, mode by mode, rather than just
looking at the aggregate KLD/MAE in eval_manhattan_exp1.py?

For each mode, the weight the row's choice_weights assigns to that mode (summed across
all (mode, duration) entries for that mode, since the same mode can appear at more than
one duration) is used as a score to separate rows where that mode WAS the true answer
from rows where it wasn't: AUC (via sklearn, equivalent to the Mann-Whitney U statistic
over n_pos*n_neg) and a one-sided Mann-Whitney U p-value (is weight-when-true greater
than weight-when-false, not just different).
"""
import json

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

RESULTS_FILE = "results/manhattan/exp1_test_1000.json"
OUT_FILE = "results/manhattan/exp1_discriminative_power.json"
MODES = ["auto_passenger", "biking", "on_demand_auto", "other_travel_mode",
         "private_auto", "public_transit", "walking"]

rows = json.load(open(RESULTS_FILE, encoding="utf-8"))

out = {}
for mode in MODES:
    y_true, weight_for_mode = [], []
    for r in rows:
        choice_weights = json.loads(r["choice_weights"])
        weight = sum(c["weight"] for c in choice_weights if c["primary_mode"] == mode)
        weight_for_mode.append(weight)
        y_true.append(1 if r["true_mode"] == mode else 0)
    y_true = np.array(y_true)
    weight_for_mode = np.array(weight_for_mode)
    pos = weight_for_mode[y_true == 1]
    neg = weight_for_mode[y_true == 0]

    auc = roc_auc_score(y_true, weight_for_mode)
    p_value = mannwhitneyu(pos, neg, alternative="greater").pvalue

    out[mode] = {
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "median_weight_when_true": float(np.median(pos)),
        "median_weight_when_false": float(np.median(neg)),
        "mean_weight_when_true": float(np.mean(pos)),
        "mean_weight_when_false": float(np.mean(neg)),
        "auc_discriminative_power": float(auc),
        "p_value": float(p_value),
    }
    print(f"{mode:20s} n_pos={len(pos):4d}  AUC={auc:.4f}  p={p_value:.3e}", flush=True)

json.dump(out, open(OUT_FILE, "w"), indent=2)
print(f"\nSaved {OUT_FILE}")
