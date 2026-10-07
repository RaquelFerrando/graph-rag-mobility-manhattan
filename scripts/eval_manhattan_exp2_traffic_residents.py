"""
Fairer version of the traffic validation: route REAL Manhattan-resident trips
(from our own reference+eval files, which are already 100% resident by home
county - confirmed empirically) onto the same street network, instead of
comparing against Replica's "Network Link Volumes" product (which aggregates
ALL traffic - residents, commuters, tourists, commercial). This makes the
comparison resident-vs-resident on both sides, like the mode-choice
validation already was, instead of resident-simulation-vs-everyone.

Only trips with BOTH origin and destination inside Manhattan are routable on
our Manhattan-only street graph (69% of the reference+eval rows) - the rest
leave the island and can't be placed on this network, same limitation our
own simulated agents don't have (they never leave Manhattan in the first
place, since their homes/POIs are all Manhattan-scoped).
"""
import json
import re
from collections import Counter

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import entropy, spearmanr
from shapely.geometry import LineString

ROADS_FILE = "data/geo/replica-manhattan-roads-real.geojson"
CENSUS_FILE = "data/geo/replica-manhattan-cencus.geojson"
REF_FILE = "data/reference/replica-manhattan_trips.csv"
EVAL_FILE = "data/eval/replica-manhattan_trips_eval.csv"
SIM_EDGES_FILE = "results/manhattan/exp2_traffic_edges.geojson"


def tract_to_geoid(trct_str):
    """'206 (New York, NY)' -> 36061020600 (STATE+COUNTY+TRACT, matches census geojson customGeoId)."""
    m = re.match(r"([\d.]+)", str(trct_str))
    if not m:
        return None
    tract = m.group(1)
    if "." in tract:
        whole, frac = tract.split(".")
        code = whole.zfill(4) + frac.ljust(2, "0")
    else:
        code = tract.zfill(4) + "00"
    return int("36061" + code)


print("Loading real resident trips (reference + eval, both confirmed 100% Manhattan-home)...")
ref = pd.read_csv(REF_FILE)
ev = pd.read_csv(EVAL_FILE)
trips = pd.concat([ref, ev], ignore_index=True)
print(f"n total resident trips: {len(trips)}")

both_manhattan = (trips["origin_cty_2020"] == "New York County, NY") & (trips["destination_cty_2020"] == "New York County, NY")
trips = trips[both_manhattan].copy()
print(f"n trips with both origin and destination in Manhattan (routable on our network): {len(trips)}")

trips["origin_geoid"] = trips["origin_trct_2020"].apply(tract_to_geoid)
trips["dest_geoid"] = trips["destination_trct_2020"].apply(tract_to_geoid)

tracts = gpd.read_file(CENSUS_FILE)
centroid = tracts.set_index("customGeoId")[["centroidLat", "centroidLon"]]

trips = trips.join(centroid.rename(columns={"centroidLat": "oLat", "centroidLon": "oLon"}), on="origin_geoid")
trips = trips.join(centroid.rename(columns={"centroidLat": "dLat", "centroidLon": "dLon"}), on="dest_geoid")
trips = trips.dropna(subset=["oLat", "oLon", "dLat", "dLon"])
print(f"n trips with resolved tract centroids: {len(trips)}")

print("Building routing graph (same as the simulated-traffic script)...")
roads = gpd.read_file(ROADS_FILE)
G = nx.Graph()
for _, row in roads.iterrows():
    geom = row.geometry
    if geom is None or geom.geom_type != "LineString":
        continue
    coords = list(geom.coords)
    edge_attrs = {k: v for k, v in row.items() if k != "geometry"}
    for i in range(len(coords) - 1):
        u = f"{coords[i][0]:.6f},{coords[i][1]:.6f}"
        v = f"{coords[i+1][0]:.6f},{coords[i+1][1]:.6f}"
        if u not in G:
            G.add_node(u, x=coords[i][0], y=coords[i][1])
        if v not in G:
            G.add_node(v, x=coords[i+1][0], y=coords[i+1][1])
        seg = LineString([coords[i], coords[i+1]])
        G.add_edge(u, v, length=seg.length, geometry=seg, **edge_attrs)
print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

node_ids = list(G.nodes)
node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])
tree = cKDTree(node_xy)


def nearest_node(lat, lng):
    _, idx = tree.query([lng, lat])
    return node_ids[idx]


edge_counter = Counter()
n_routed, n_no_path = 0, 0
for i, row in enumerate(trips.itertuples(), 1):
    u = nearest_node(row.oLat, row.oLon)
    v = nearest_node(row.dLat, row.dLon)
    if u == v:
        continue
    try:
        path = nx.shortest_path(G, u, v, weight="length")
        edges = [tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])]
        edge_counter.update(edges)
        n_routed += 1
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        n_no_path += 1
    if i % 1000 == 0:
        print(f"  ...{i}/{len(trips)} routed", flush=True)

print(f"Real resident trips routed: {n_routed}, no path: {n_no_path}")

real_res_rows = []
for (u, v), count in edge_counter.items():
    attrs = G.get_edge_data(u, v)
    real_res_rows.append({"osmId": attrs.get("osmId"), "real_resident_count": count, "geometry": attrs.get("geometry")})
real_res_gdf = gpd.GeoDataFrame(real_res_rows, geometry="geometry")
print(f"n distinct edges with real resident traffic: {len(real_res_gdf)}")
real_res_gdf.to_file("results/manhattan/exp2_traffic_residents_edges.geojson", driver="GeoJSON")
print("Saved results/manhattan/exp2_traffic_residents_edges.geojson")

print("Loading simulated edges for comparison...")
sim = gpd.read_file(SIM_EDGES_FILE)

merged = pd.merge(
    real_res_gdf[["osmId", "real_resident_count"]],
    sim[["osmId", "sim_count"]],
    on="osmId", how="outer",
).fillna(0)
merged = merged.groupby("osmId", as_index=False).sum()  # osmId can repeat across sub-segments

P = merged["real_resident_count"] + 1
Q = merged["sim_count"] + 1
kld = entropy(P / P.max(), Q / Q.max())
rho, pval = spearmanr(merged["real_resident_count"], merged["sim_count"])
print(f"\nRESIDENT-VS-RESIDENT traffic KLD: {kld:.4f}")
print(f"Spearman rank correlation: {rho:.4f} (p={pval:.2e})")
print(f"(For reference: resident-sim vs. EVERYONE'S real traffic KLD was 2.69, rank corr 0.18)")

summary = {
    "n_real_resident_trips_routed": n_routed,
    "n_real_resident_no_path": n_no_path,
    "n_edges_compared": len(merged),
    "kld_resident_vs_resident": float(kld),
    "spearman_resident_vs_resident": float(rho),
    "spearman_pvalue": float(pval),
    "kld_resident_sim_vs_everyone_real": 2.6929,
    "spearman_resident_sim_vs_everyone_real": 0.1799,
}
json.dump(summary, open("results/manhattan/exp2_traffic_residents_validation.json", "w"), indent=2)
print("\nSaved to results/manhattan/exp2_traffic_residents_validation.json")
