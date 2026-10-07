"""
Geographic realism check at full scale (1000 agents): does the census-tract
distribution of synthetic agents' homes match the real population-by-tract
weights (peopleByHome), not just "look spread out"?

Point-in-polygon match of each agent's home lat/lon against Manhattan's 310
real census tracts (data/geo/replica-manhattan-cencus.geojson), then aggregate
KLD/MAE between the resulting tract distribution and the real peopleByHome
weights - same methodology as every other validation in this project, applied
here instead of the earlier neighborhood-name reverse-geocoding spot-check.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import entropy
from shapely.geometry import Point

AGENTS_FILE = "results/manhattan/exp2_agents_1000.json"
CENSUS_FILE = "data/geo/replica-manhattan-cencus.geojson"

agents = json.load(open(AGENTS_FILE))
homes = [a["home"]["coordinates"] for a in agents if a.get("home")]
print(f"n agents with home: {len(homes)}")

tracts = gpd.read_file(CENSUS_FILE)
tracts["peopleByHome"] = tracts["peopleByHome"].astype(float)
total_pop = tracts["peopleByHome"].sum()
print(f"n tracts: {len(tracts)}  total real population (peopleByHome): {total_pop:.0f}")

# point-in-polygon: home coords are stored as [lat, lon]
points = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in homes],
    crs=tracts.crs,
)
joined = gpd.sjoin(points, tracts[["customGeoId", "peopleByHome", "geometry"]], how="left", predicate="within")
n_matched = joined["customGeoId"].notna().sum()
print(f"n homes matched to a tract: {n_matched}/{len(homes)}")

sim_counts = joined["customGeoId"].value_counts()
sim_dist = sim_counts / sim_counts.sum()

real_dist = tracts.set_index("customGeoId")["peopleByHome"] / total_pop

all_ids = sorted(set(sim_dist.index) | set(real_dist.index))
p = np.array([real_dist.get(i, 0.0) for i in all_ids])
q = np.array([sim_dist.get(i, 0.0) for i in all_ids])
mae = np.mean(np.abs(p - q))
smooth = 1e-10
p_n = (p + smooth) / (p + smooth).sum()
q_n = (q + smooth) / (q + smooth).sum()
kld = entropy(p_n, q_n)

print(f"\nTract-level home distribution: KLD={kld:.4f}  MAE={mae:.6f}  (n_tracts_compared={len(all_ids)})")

n_tracts_used = (sim_counts > 0).sum()
n_tracts_with_pop = (tracts["peopleByHome"] > 0).sum()
print(f"n tracts with >=1 simulated agent: {n_tracts_used} / {len(tracts)} total tracts "
      f"({n_tracts_with_pop} have nonzero real population)")

avg_density_agents = joined["peopleByHome"].mean()
avg_density_all_tracts = tracts["peopleByHome"].mean()
print(f"\nAvg peopleByHome of tract where an agent lives: {avg_density_agents:.0f}")
print(f"Avg peopleByHome across ALL Manhattan tracts (unweighted): {avg_density_all_tracts:.0f}")

out = {
    "n_homes": len(homes), "n_matched": int(n_matched),
    "kld": kld, "mae": mae,
    "n_tracts_used": int(n_tracts_used), "n_tracts_total": len(tracts),
    "avg_density_agent_tracts": float(avg_density_agents),
    "avg_density_all_tracts": float(avg_density_all_tracts),
}
json.dump(out, open("results/manhattan/exp2_geography_validation.json", "w"), indent=2)
print("\nSaved to results/manhattan/exp2_geography_validation.json")
