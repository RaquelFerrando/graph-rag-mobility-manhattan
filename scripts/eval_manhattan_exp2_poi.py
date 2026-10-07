"""
Experiment II's POI-visit validation, matching the ORIGINAL paper's own
methodology exactly (exp2_step2_analysis.ipynb, "POI Visit Counts") - unlike
every other Exp2 check in this project (which are our own additions), this
one IS something the paper itself validated for Cambridge. Applying the same
method to Manhattan for direct comparability.

Method (copied from the paper's notebook, adapted to our file paths):
1. Extract every non-'home' destination visited across all 1000 agents'
   schedules -> count how many times each exact (name, lat, lon) POI was
   visited (SIM_VISIT_COUNTS).
2. Outer-merge with the real POI catalog's RAW_VISIT_COUNTS on
   (LOCATION_NAME, LATITUDE, LONGITUDE) - exact match, since our simulated
   destinations are always rows sourced directly from this same POI file.
3. Add the paper's own min_counts=10 smoothing to both real and simulated
   counts (not our usual 1e-10 - kept faithful to the original notebook so
   this specific number is comparable to what the paper itself would report).
4. KLD via scipy's entropy (which normalizes internally, so the paper's
   divide-by-max step is a no-op, kept anyway for fidelity).
"""
import json

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy

AGENTS_FILE = "results/manhattan/exp2_agents_1000.json"
POI_FILE = "data/geo/replica-manhattan-poi.csv"
MIN_COUNTS = 10  # matches the paper's own notebook exactly

agents = json.load(open(AGENTS_FILE))

poi_list = []
n_stops_total = 0
n_home_skipped = 0
for a in agents:
    for stop in a.get("schedule", []):
        n_stops_total += 1
        if stop.get("desire") == "home":
            n_home_skipped += 1
            continue
        dest = stop.get("destination") or {}
        name = dest.get("name")
        coords = dest.get("coordinates")
        if name and coords and len(coords) == 2:
            try:
                poi_list.append((name, float(coords[0]), float(coords[1])))
            except (TypeError, ValueError):
                continue

print(f"Total stops: {n_stops_total}  (home skipped: {n_home_skipped}, non-home destinations: {len(poi_list)})")

poi_counter = Counter(poi_list)
sim_df = pd.DataFrame(
    [{"LOCATION_NAME": n, "LATITUDE": lat, "LONGITUDE": lon, "SIM_VISIT_COUNTS": c}
     for (n, lat, lon), c in poi_counter.items()]
)
print(f"n distinct simulated POIs visited: {len(sim_df)}")

poi_df = pd.read_csv(POI_FILE)[["LOCATION_NAME", "LATITUDE", "LONGITUDE", "RAW_VISIT_COUNTS"]]
print(f"n real POIs in catalog: {len(poi_df)}")

merged = pd.merge(poi_df, sim_df, on=["LOCATION_NAME", "LATITUDE", "LONGITUDE"], how="outer")
n_matched = merged["RAW_VISIT_COUNTS"].notna().sum() - (merged["RAW_VISIT_COUNTS"].isna()).sum()
n_both = ((merged["RAW_VISIT_COUNTS"].notna()) & (merged["SIM_VISIT_COUNTS"].notna())).sum()
n_real_only = ((merged["RAW_VISIT_COUNTS"].notna()) & (merged["SIM_VISIT_COUNTS"].isna())).sum()
n_sim_only = ((merged["RAW_VISIT_COUNTS"].isna()) & (merged["SIM_VISIT_COUNTS"].notna())).sum()
print(f"POIs visited in BOTH real data and simulation: {n_both}")
print(f"POIs with real visits but never simulated: {n_real_only}")
print(f"POIs simulated-visited but with zero real visit data: {n_sim_only}")

merged = merged.fillna(0)
merged["RAW_VISIT_COUNTS"] += MIN_COUNTS
merged["SIM_VISIT_COUNTS"] += MIN_COUNTS

P = merged["RAW_VISIT_COUNTS"] / merged["RAW_VISIT_COUNTS"].max()
Q = merged["SIM_VISIT_COUNTS"] / merged["SIM_VISIT_COUNTS"].max()
kld = entropy(P, Q)
print(f"\nPOI-visit KLD (paper's own method, min_counts={MIN_COUNTS}): {kld:.4f}")

# also report top simulated POIs vs whether they're realistic (high real visits too)
top_sim = merged.sort_values("SIM_VISIT_COUNTS", ascending=False).head(10)
print("\nTop 10 most-simulated-visited POIs (sim count includes +10 smoothing):")
print(top_sim[["LOCATION_NAME", "RAW_VISIT_COUNTS", "SIM_VISIT_COUNTS"]].to_string(index=False))

out = {
    "n_stops_total": n_stops_total, "n_home_skipped": n_home_skipped,
    "n_distinct_sim_pois": len(sim_df), "n_real_pois_catalog": len(poi_df),
    "n_matched_both": int(n_both), "n_real_only": int(n_real_only), "n_sim_only": int(n_sim_only),
    "kld": float(kld), "min_counts_smoothing": MIN_COUNTS,
}
json.dump(out, open("results/manhattan/exp2_poi_validation.json", "w"), indent=2)
print("\nSaved to results/manhattan/exp2_poi_validation.json")
