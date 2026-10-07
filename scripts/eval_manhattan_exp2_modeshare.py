"""
Experiment II's headline validation: does the FULL pipeline (schedule generation +
mode/duration choice + POI selection, chained together across a whole synthetic day)
still produce a realistic city-wide mode/duration mix - not just a single isolated
choice like Experiment I tested?

Flattens every stop across all 1000 synthetic agents' full-day schedules into one
big list of individual trips, and compares the aggregate mode/duration distribution
against the same real Manhattan eval distribution used throughout this project -
same aggregate_kl_mae methodology as eval_manhattan_exp1.py, so directly comparable.

Also reports trip_purpose (desire) mix and stops-per-agent, since those are free
byproducts of the same flattening and were separately asked about.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
AGENTS_FILE = "results/manhattan/exp2_agents_1000.json"


def aggregate_kl_mae(true_col, pred_col):
    true_dist = pd.Series(true_col).value_counts(normalize=True)
    pred_dist = pd.Series(pred_col).value_counts(normalize=True)
    all_values = sorted(set(true_dist.index) | set(pred_dist.index))
    p = np.array([true_dist.get(v, 0.0) for v in all_values])
    q = np.array([pred_dist.get(v, 0.0) for v in all_values])
    mae = np.mean(np.abs(p - q))
    smooth = 1e-10
    p = (p + smooth) / (p + smooth).sum()
    q = (q + smooth) / (q + smooth).sum()
    return entropy(p, q), mae, all_values, pd.Series(true_col).value_counts(normalize=True), pd.Series(pred_col).value_counts(normalize=True)


print("Loading real Manhattan eval distribution (ground truth)...")
eval_df = load_data(EVAL_FILE)

print(f"Loading {AGENTS_FILE}...")
agents = json.load(open(AGENTS_FILE))
print(f"n agents: {len(agents)}")

all_modes, all_durations, all_desires, stops_per_agent = [], [], [], []
for a in agents:
    schedule = a.get("schedule", [])
    stops_per_agent.append(len(schedule))
    for stop in schedule:
        t = stop.get("transportation") or {}
        mode = t.get("mode")
        duration = t.get("duration")
        desire = stop.get("desire")
        if mode and duration:
            all_modes.append(mode)
            all_durations.append(duration)
        if desire:
            all_desires.append(desire)

n_trips = len(all_modes)
print(f"n total simulated trips (all stops, all agents): {n_trips}")
print(f"avg stops/agent: {np.mean(stops_per_agent):.2f}  median: {np.median(stops_per_agent):.0f}  "
      f"min: {min(stops_per_agent)}  max: {max(stops_per_agent)}")

print("\n" + "=" * 70)
print("1. MODE - aggregate distribution: Experiment II (all trips) vs real eval")
print("=" * 70)
mode_kl, mode_mae, mode_vals, real_mode_dist, sim_mode_dist = aggregate_kl_mae(eval_df["primary_mode"], all_modes)
print(f"KLD={mode_kl:.4f}  MAE={mode_mae:.4f}")
comp = pd.DataFrame({"real_%": (real_mode_dist * 100).round(1), "exp2_sim_%": (sim_mode_dist * 100).round(1)}).reindex(mode_vals)
print(comp.to_string())

print("\n" + "=" * 70)
print("2. DURATION - aggregate distribution: Experiment II (all trips) vs real eval")
print("=" * 70)
dur_kl, dur_mae, dur_vals, real_dur_dist, sim_dur_dist = aggregate_kl_mae(eval_df["duration_minutes"], all_durations)
print(f"KLD={dur_kl:.4f}  MAE={dur_mae:.4f}")
comp2 = pd.DataFrame({"real_%": (real_dur_dist * 100).round(1), "exp2_sim_%": (sim_dur_dist * 100).round(1)}).reindex(dur_vals)
print(comp2.to_string())

print("\n" + "=" * 70)
print("3. TRIP PURPOSE (desire) - Experiment II mix vs real eval trip_purpose")
print("=" * 70)
purp_kl, purp_mae, purp_vals, real_purp_dist, sim_purp_dist = aggregate_kl_mae(eval_df["trip_purpose"], all_desires)
print(f"KLD={purp_kl:.4f}  MAE={purp_mae:.4f}")
comp3 = pd.DataFrame({"real_%": (real_purp_dist * 100).round(1), "exp2_sim_%": (sim_purp_dist * 100).round(1)}).reindex(purp_vals)
print(comp3.to_string())

# Recap Experiment I's own numbers for direct side-by-side comparison
print("\n" + "=" * 70)
print("4. Recap: Experiment I (single isolated choice) for comparison")
print("=" * 70)
exp1 = json.load(open("results/manhattan/exp1_test_1000.json"))
exp1_mode_kl, exp1_mode_mae, *_ = aggregate_kl_mae(eval_df["primary_mode"], [r["predicted_mode"] for r in exp1])
exp1_dur_kl, exp1_dur_mae, *_ = aggregate_kl_mae(eval_df["duration_minutes"], [r["predicted_duration"] for r in exp1])
print(f"Exp1 mode:     KLD={exp1_mode_kl:.4f} MAE={exp1_mode_mae:.4f}")
print(f"Exp2 mode:     KLD={mode_kl:.4f} MAE={mode_mae:.4f}")
print(f"Exp1 duration: KLD={exp1_dur_kl:.4f} MAE={exp1_dur_mae:.4f}")
print(f"Exp2 duration: KLD={dur_kl:.4f} MAE={dur_mae:.4f}")

out = {
    "n_agents": len(agents),
    "n_trips": n_trips,
    "stops_per_agent": {"mean": float(np.mean(stops_per_agent)), "median": float(np.median(stops_per_agent)),
                         "min": int(min(stops_per_agent)), "max": int(max(stops_per_agent))},
    "mode": {"kld": mode_kl, "mae": mode_mae, "real_%": real_mode_dist.round(3).to_dict(), "sim_%": sim_mode_dist.round(3).to_dict()},
    "duration": {"kld": dur_kl, "mae": dur_mae, "real_%": real_dur_dist.round(3).to_dict(), "sim_%": sim_dur_dist.round(3).to_dict()},
    "trip_purpose": {"kld": purp_kl, "mae": purp_mae, "real_%": real_purp_dist.round(3).to_dict(), "sim_%": sim_purp_dist.round(3).to_dict()},
    "exp1_recap": {"mode_kld": exp1_mode_kl, "mode_mae": exp1_mode_mae, "duration_kld": exp1_dur_kl, "duration_mae": exp1_dur_mae},
}
out_path = "results/manhattan/exp2_modeshare_validation.json"
json.dump(out, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
