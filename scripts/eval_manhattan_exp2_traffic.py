"""
Manhattan road-traffic validation, replicating the ORIGINAL paper's own method
exactly (exp2_step2_analysis.ipynb, "Trip Visualization" section) - the paper's
Fig 9-style comparison: does the aggregate simulated traffic flow, routed onto
the real street network, match real per-segment trip_count?

Same algorithm as the paper's notebook:
  1. Build a networkx graph from the roads geojson - every LineString's
     individual coordinate segments become graph edges, keyed by
     "lon,lat" node ids, carrying the parent link's full attributes
     (trip_count, roadName, highway, osmId...) on every sub-edge.
  2. For each agent, take the ORDERED sequence of destination coordinates
     across their whole day (home included, unlike the POI script which
     skips home) - each consecutive pair is one simulated trip leg.
  3. For each leg, snap both points to their nearest graph node and compute
     the shortest path (Dijkstra, weighted by segment length); count every
     edge on that path (Counter, aggregated across all agents' legs).
  4. Merge simulated edge counts with each edge's real trip_count; KLD via
     scipy's entropy, each column normalized by its own max (matches the
     paper's own normalization, which its notebook actually needed since -
     unlike our other scripts - it does NOT add smoothing here).

Deliberate improvement over the paper's own code (result-identical, just
faster): nearest-node lookup uses a KDTree instead of a brute-force O(n)
scan over every node for every query - correctness is unchanged (same
nearest node found), only the search algorithm is faster, which matters
here since Manhattan's real network is far larger than Cambridge's.
"""
import json
from collections import Counter

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import entropy
from shapely.geometry import LineString

ROADS_FILE = "data/geo/replica-manhattan-roads-real.geojson"
AGENTS_FILE = "results/manhattan/exp2_agents_1000.json"
OUT_EDGES = "results/manhattan/exp2_traffic_edges.geojson"
OUT_SUMMARY = "results/manhattan/exp2_traffic_validation.json"

print(f"Loading road network from {ROADS_FILE}...")
roads = gpd.read_file(ROADS_FILE)
print(f"n links: {len(roads)}")

print("Building routing graph...")
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

print(f"Created network with {len(G.nodes)} nodes and {len(G.edges)} edges")

# KDTree over all node coords for fast nearest-neighbor snapping (same result
# as the paper's brute-force scan, just not O(n) per lookup).
node_ids = list(G.nodes)
node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])
tree = cKDTree(node_xy)


def nearest_node(lat, lng):
    _, idx = tree.query([lng, lat])
    return node_ids[idx]


agents = json.load(open(AGENTS_FILE))
edge_counter = Counter()
n_legs_total = 0
n_legs_routed = 0
n_legs_no_path = 0

for a_i, a in enumerate(agents, 1):
    sched = a.get("schedule", [])
    coords = []
    for stop in sched:
        dest = stop.get("destination") or {}
        c = dest.get("coordinates")
        if c and len(c) == 2:
            try:
                coords.append((float(c[0]), float(c[1])))  # (lat, lng)
            except (TypeError, ValueError):
                continue
    for i in range(len(coords) - 1):
        n_legs_total += 1
        lat1, lng1 = coords[i]
        lat2, lng2 = coords[i + 1]
        if (lat1, lng1) == (lat2, lng2):
            continue
        u = nearest_node(lat1, lng1)
        v = nearest_node(lat2, lng2)
        try:
            path = nx.shortest_path(G, u, v, weight="length")
            # G is an undirected Graph, but a plain (a,b) tuple treats a
            # path traversed one way and the reverse-traversed one as two
            # DIFFERENT dict keys - the paper's own notebook has this same
            # bug (confirmed: it produced more "edges with sim traffic" than
            # the graph actually has). Sorting each pair collapses both
            # directions onto the one real undirected edge they represent.
            edges = [tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])]
            edge_counter.update(edges)
            n_legs_routed += 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            n_legs_no_path += 1
    if a_i % 100 == 0:
        print(f"  ...{a_i}/{len(agents)} agents processed, {n_legs_routed} legs routed so far", flush=True)

print(f"\nLegs: total={n_legs_total} routed={n_legs_routed} no_path={n_legs_no_path}")

edge_rows = []
for (u, v), count in edge_counter.items():
    attrs = G.get_edge_data(u, v)
    edge_rows.append({
        "osmId": attrs.get("osmId"), "roadName": attrs.get("roadName"),
        "highway": attrs.get("highway"), "trip_count": attrs.get("trip_count"),
        "sim_count": count, "geometry": attrs.get("geometry"),
    })
edge_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry")
print(f"n distinct edges with >=1 simulated trip: {len(edge_gdf)}")

P = edge_gdf["trip_count"] / edge_gdf["trip_count"].max()
Q = edge_gdf["sim_count"] / edge_gdf["sim_count"].max()
kld = entropy(P, Q)
mae = float(np.mean(np.abs(P / P.sum() - Q / Q.sum())))
print(f"\nRoad-traffic KLD (paper's own method): {kld:.4f}  MAE={mae:.6f}")
print(f"(Paper's own Cambridge result for reference: no-ref=0.8144, with-ref=0.6215)")

edge_gdf.to_file(OUT_EDGES, driver="GeoJSON")
summary = {
    "n_links_total": len(roads), "n_graph_nodes": len(G.nodes), "n_graph_edges": len(G.edges),
    "n_legs_total": n_legs_total, "n_legs_routed": n_legs_routed, "n_legs_no_path": n_legs_no_path,
    "n_edges_with_sim_traffic": len(edge_gdf), "kld": float(kld), "mae": mae,
    "paper_cambridge_kld_noref": 0.8144221537414615, "paper_cambridge_kld_ref": 0.6215056909459673,
}
json.dump(summary, open(OUT_SUMMARY, "w"), indent=2)
print(f"\nSaved {OUT_EDGES} and {OUT_SUMMARY}")
