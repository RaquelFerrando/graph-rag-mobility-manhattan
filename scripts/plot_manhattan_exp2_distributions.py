"""
Generates all Experiment II validation charts + saves the schedule-coherence
numbers that were only printed to stdout earlier (not persisted anywhere).
Reuses the same real-vs-simulated bar-chart style as
plot_manhattan_distributions.py (Experiment I) for visual consistency.

Produces, all under results/manhattan/:
  exp2_mode_duration_comparison.png
  exp2_trip_purpose_comparison.png
  exp2_demographics_comparison.png
  exp2_geography_decile.png
  exp2_starttime_by_hour.png
  exp2_schedule_coherence.json   (was only printed before, never saved)
"""
import json
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, '.')
from baseline.data import load_data, categorize_age, categorize_income

OUT_DIR = "results/manhattan"
AGENTS_FILE = f"{OUT_DIR}/exp2_agents_1000.json"
EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"

agents = json.load(open(AGENTS_FILE))
eval_df = load_data(EVAL_FILE)

all_modes, all_durations, all_desires, all_hours = [], [], [], []
stops_per_agent, home_per_agent = [], []
first_desire, last_desire = [], []
n_out_of_order = 0

def to_minutes(t):
    h, m = t.split(':')
    return int(h) * 60 + int(m)

for a in agents:
    sched = a.get("schedule", [])
    stops_per_agent.append(len(sched))
    home_per_agent.append(sum(1 for s in sched if s.get("desire") == "home"))
    if sched:
        first_desire.append(sched[0]["desire"])
        last_desire.append(sched[-1]["desire"])
        try:
            times = [to_minutes(s["start_time"]) for s in sched]
            if times != sorted(times):
                n_out_of_order += 1
        except Exception:
            pass
    for stop in sched:
        t = stop.get("transportation") or {}
        if t.get("mode") and t.get("duration"):
            all_modes.append(t["mode"])
            all_durations.append(t["duration"])
        if stop.get("desire"):
            all_desires.append(stop["desire"])
        if stop.get("start_time"):
            try:
                all_hours.append(int(stop["start_time"].split(":")[0]))
            except Exception:
                pass

# ---------- 1. mode + duration comparison (Exp1-style 2x2 panel) ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Manhattan Experiment II: real vs. simulated distributions (n=1000 agents, 8278 trips)")

for ax, (col_real, col_sim, title) in zip(
    axes[0], [(eval_df["primary_mode"], all_modes, "Transportation Mode")]
):
    pass

def bar_panel(ax, real_series, sim_series, title, order=None):
    real_dist = pd.Series(real_series).value_counts(normalize=True)
    sim_dist = pd.Series(sim_series).value_counts(normalize=True)
    if order is None:
        order = real_dist.sort_values(ascending=False).index.tolist()
    real_vals = [real_dist.get(k, 0) for k in order]
    sim_vals = [sim_dist.get(k, 0) for k in order]
    return real_vals, sim_vals, order

mode_order = eval_df["primary_mode"].value_counts().index.tolist()
real_vals, sim_vals, order = bar_panel(None, eval_df["primary_mode"], all_modes, "", mode_order)
axes[0, 0].bar(order, real_vals, color="#4C72B0")
axes[0, 0].set_title("Ground Truth - Mode")
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 1].bar(order, sim_vals, color="#DD8452")
axes[0, 1].set_title("Experiment II Simulated - Mode")
axes[0, 1].tick_params(axis='x', rotation=45)

dur_order = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
real_dist = pd.Series(eval_df["duration_minutes"]).value_counts(normalize=True)
sim_dist = pd.Series(all_durations).value_counts(normalize=True)
real_vals = [real_dist.get(k, 0) for k in dur_order]
sim_vals = [sim_dist.get(k, 0) for k in dur_order]
axes[1, 0].bar(dur_order, real_vals, color="#4C72B0")
axes[1, 0].set_title("Ground Truth - Duration")
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 1].bar(dur_order, sim_vals, color="#DD8452")
axes[1, 1].set_title("Experiment II Simulated - Duration")
axes[1, 1].tick_params(axis='x', rotation=45)
for ax in axes.flat:
    ax.set_ylim(0, max(0.5, max(real_vals + sim_vals) * 1.15))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_mode_duration_comparison.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/exp2_mode_duration_comparison.png")

# ---------- 2. trip purpose comparison ----------
canonical_desire = ['home', 'work', 'shop', 'eat', 'recreation', 'social', 'school', 'maintenance', 'other_activity_type']
real_dist = pd.Series(eval_df["trip_purpose"]).value_counts(normalize=True)
sim_dist = pd.Series(all_desires).value_counts(normalize=True)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(canonical_desire))
w = 0.35
ax.bar(x - w/2, [real_dist.get(k, 0) for k in canonical_desire], w, label="Real", color="#4C72B0")
ax.bar(x + w/2, [sim_dist.get(k, 0) for k in canonical_desire], w, label="Experiment II Simulated", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(canonical_desire, rotation=45, ha='right')
ax.set_title("Trip purpose: real vs. Experiment II simulated")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_trip_purpose_comparison.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/exp2_trip_purpose_comparison.png")

# ---------- 3. demographics comparison (4-panel) ----------
synthetic = pd.DataFrame([json.loads(a["profile"]) if isinstance(a["profile"], str) else a["profile"] for a in agents])
synthetic["age_group"] = synthetic["age"].apply(categorize_age)
real_raw = pd.read_csv("data/reference/replica-manhattan_trips.csv")
real = real_raw.drop_duplicates(subset='trip_taker_person_id').rename(columns={
    'trip_taker_age': 'age', 'trip_taker_employment_status': 'employment_status',
    'trip_taker_household_size': 'household_size', 'trip_taker_education': 'education',
})
real['age_group'] = real['age'].apply(categorize_age)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fields = [
    ("age_group", ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']),
    ("employment_status", None),
    ("education", None),
    ("household_size", None),
]
for ax, (field, order) in zip(axes.flat, fields):
    real_d = real[field].value_counts(normalize=True)
    sim_d = synthetic[field].value_counts(normalize=True)
    cats = order if order else sorted(set(real_d.index) | set(sim_d.index), key=lambda k: str(k))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, [real_d.get(k, 0) for k in cats], w, label="Real", color="#4C72B0")
    ax.bar(x + w/2, [sim_d.get(k, 0) for k in cats], w, label="Simulated", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cats], rotation=45, ha='right')
    ax.set_title(field)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_demographics_comparison.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/exp2_demographics_comparison.png")

# ---------- 4. geography: decile density comparison ----------
homes = [a["home"]["coordinates"] for a in agents if a.get("home")]
tracts = gpd.read_file("data/geo/replica-manhattan-cencus.geojson")
tracts["peopleByHome"] = tracts["peopleByHome"].astype(float)
tracts_sorted = tracts.sort_values("peopleByHome").reset_index(drop=True)
tracts_sorted["cum_pop"] = tracts_sorted["peopleByHome"].cumsum()
total = tracts_sorted["peopleByHome"].sum()
tracts_sorted["decile"] = pd.cut(tracts_sorted["cum_pop"], bins=np.linspace(0, total, 11), labels=False, include_lowest=True)
points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in homes], crs=tracts.crs)
joined = gpd.sjoin(points, tracts_sorted[["customGeoId", "decile", "geometry"]], how="left", predicate="within")
sim_dist = joined["decile"].value_counts(normalize=True).sort_index()
real_dist = tracts_sorted.groupby("decile")["peopleByHome"].sum()
real_dist = (real_dist / real_dist.sum()).sort_index()
all_d = sorted(set(sim_dist.index) | set(real_dist.index))
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(all_d))
w = 0.35
ax.bar(x - w/2, [real_dist.get(d, 0) for d in all_d], w, label="Real (population share)", color="#4C72B0")
ax.bar(x + w/2, [sim_dist.get(d, 0) for d in all_d], w, label="Simulated (agent homes)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels([f"D{int(d)+1}" for d in all_d])
ax.axhline(0.10, color="gray", linestyle="--", linewidth=1, label="Expected (10%/decile)")
ax.set_title("Home placement by real population density decile (D1=lowest, D10=highest)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_geography_decile.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/exp2_geography_decile.png")

# ---------- 5. start time by hour ----------
real_hours = eval_df["start_time"].astype(int)
real_dist = real_hours.value_counts(normalize=True).sort_index()
sim_dist = pd.Series(all_hours).value_counts(normalize=True).sort_index()
all_h = list(range(24))
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(all_h, [real_dist.get(h, 0) for h in all_h], marker='o', label="Real", color="#4C72B0")
ax.plot(all_h, [sim_dist.get(h, 0) for h in all_h], marker='o', label="Experiment II Simulated", color="#DD8452")
ax.set_xticks(all_h)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Share of trips")
ax.set_title("Activity start time by hour: real vs. Experiment II simulated")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_starttime_by_hour.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/exp2_starttime_by_hour.png")

# ---------- 6. save schedule coherence numbers (previously only printed) ----------
from collections import Counter
coherence = {
    "n_agents": len(agents),
    "n_agents_with_schedule": sum(1 for a in agents if a.get("schedule")),
    "avg_stops_per_agent": float(np.mean(stops_per_agent)),
    "median_stops_per_agent": float(np.median(stops_per_agent)),
    "min_stops": int(min(stops_per_agent)),
    "max_stops": int(max(stops_per_agent)),
    "avg_home_stops_per_agent": float(np.mean(home_per_agent)),
    "avg_non_home_stops_per_agent": float(np.mean(stops_per_agent) - np.mean(home_per_agent)),
    "pct_schedules_out_of_order": n_out_of_order / max(sum(1 for a in agents if a.get("schedule")), 1),
    "first_activity_counts": dict(Counter(first_desire).most_common()),
    "last_activity_counts": dict(Counter(last_desire).most_common()),
    "pct_last_activity_is_home": Counter(last_desire).get("home", 0) / max(len(last_desire), 1),
    "starttime_by_hour_kld_vs_real": None,  # filled below
}
from scipy.stats import entropy
p = np.array([real_dist.get(h, 0.0) for h in all_h])
q = np.array([sim_dist.get(h, 0.0) for h in all_h])
smooth = 1e-10
coherence["starttime_by_hour_kld_vs_real"] = float(entropy((p+smooth)/(p+smooth).sum(), (q+smooth)/(q+smooth).sum()))
coherence["starttime_by_hour_mae_vs_real"] = float(np.mean(np.abs(p - q)))

json.dump(coherence, open(f"{OUT_DIR}/exp2_schedule_coherence.json", "w"), indent=2)
print(f"Saved {OUT_DIR}/exp2_schedule_coherence.json")
