"""
Exports real Manhattan resident trips and our simulated agents' trips as
Profile-shaped JSON (matching mobility-agents-manhattan's src/types/profile.ts:
rutas_diarias + streetRoute), so the animation screen there can reuse the
app's OWN buildTripData()/buildCitizenLayers() rendering pipeline as-is
(person dots, mode-colored comet trails, the pointillist "imprint" trace)
instead of a from-scratch simplified renderer - keeping the exact original
visual language for this comparison.

Reuses the same graph-building + nearest-node-snapping + shortest-path code
as eval_manhattan_exp2_traffic.py / eval_manhattan_exp2_traffic_residents.py.

Mode strings are written using the Spanish substrings modeGroupOf() in
citizenLayers.ts already matches ('bici'->bike, 'coche'->car, 'metro'->transit,
else->walk) - no changes to that app code, just speaking its vocabulary.

Output (both consumed by mobility-agents-manhattan's TrafficCompareScreen):
  results/manhattan/anim_real_trips.json  - one pseudo-profile per real trip
  results/manhattan/anim_sim_trips.json   - one profile per simulated agent
"""
import json
import re

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString

ROADS_FILE = "data/geo/replica-manhattan-roads-real.geojson"
CENSUS_FILE = "data/geo/replica-manhattan-cencus.geojson"
REF_FILE = "data/reference/replica-manhattan_trips.csv"
EVAL_FILE = "data/eval/replica-manhattan_trips_eval.csv"
AGENTS_FILE = "results/manhattan/exp2_agents_1000.json"
OUT_REAL = "results/manhattan/anim_real_trips.json"
OUT_SIM = "results/manhattan/anim_sim_trips.json"

DURATION_BUCKET_MID = {
    "0-10": 5, "10-20": 15, "20-30": 25, "30-40": 35, "40-50": 45, "50-60": 55,
}

# citizenLayers.ts's modeGroupOf() matches these Spanish substrings - speak
# its vocabulary instead of touching that code.
MODE_TO_ES = {
    "walking": "a pie",
    "biking": "bici",
    "auto_passenger": "coche",
    "private_auto": "coche",
    "on_demand_auto": "taxi",
    "public_transit": "metro",
    "other_travel_mode": "otro",
}


def tract_to_geoid(trct_str):
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


def minutes_to_hora(m):
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


print("Building routing graph...")
roads = gpd.read_file(ROADS_FILE)
G = nx.Graph()
for _, row in roads.iterrows():
    geom = row.geometry
    if geom is None or geom.geom_type != "LineString":
        continue
    coords = list(geom.coords)
    for i in range(len(coords) - 1):
        u = f"{coords[i][0]:.6f},{coords[i][1]:.6f}"
        v = f"{coords[i+1][0]:.6f},{coords[i+1][1]:.6f}"
        if u not in G:
            G.add_node(u, x=coords[i][0], y=coords[i][1])
        if v not in G:
            G.add_node(v, x=coords[i + 1][0], y=coords[i + 1][1])
        seg = LineString([coords[i], coords[i + 1]])
        G.add_edge(u, v, length=seg.length)
print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

node_ids = list(G.nodes)
node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])
tree = cKDTree(node_xy)


def nearest_node(lat, lng):
    _, idx = tree.query([lng, lat])
    return node_ids[idx]


# Our routing graph has one node per individual OSM vertex (curves can have
# dozens of them per block), so a cross-town shortest path can carry hundreds
# of nodes - far more than the original app's routes were ever built for.
# citizenLayers.ts's buildImprint() stipples once per consecutive node PAIR
# regardless of real-world spacing, so node count (not trip distance) is what
# drove the render to millions of dots and locked up the tab at ~1,000
# profiles. Simplifying the geometry (Douglas-Peucker, ~9m tolerance - well
# under the imprint's own 8m dot spacing, so the visible path is unchanged)
# fixes the actual cause instead of just showing fewer people.
SIMPLIFY_TOLERANCE_DEG = 0.00008  # ~8-9m at NYC's latitude


def route_leg_latlng(lat1, lng1, lat2, lng2):
    """Returns [[lat,lng], ...] simplified path (StreetSegment convention) or None."""
    if (lat1, lng1) == (lat2, lng2):
        return None
    u, v = nearest_node(lat1, lng1), nearest_node(lat2, lng2)
    try:
        path_nodes = nx.shortest_path(G, u, v, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(path_nodes) < 2:
        return None
    lnglat = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path_nodes]
    if len(lnglat) > 2:
        simplified = LineString(lnglat).simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=False)
        lnglat = list(simplified.coords)
    return [[lat, lng] for lng, lat in lnglat]  # (lat,lng)


# ---------------- REAL resident trips (one pseudo-profile per trip) ----------------
print("\nLoading real resident trips...")
ref = pd.read_csv(REF_FILE)
ev = pd.read_csv(EVAL_FILE)
trips = pd.concat([ref, ev], ignore_index=True)
both_manhattan = (trips["origin_cty_2020"] == "New York County, NY") & (trips["destination_cty_2020"] == "New York County, NY")
trips = trips[both_manhattan].copy()
trips["origin_geoid"] = trips["origin_trct_2020"].apply(tract_to_geoid)
trips["dest_geoid"] = trips["destination_trct_2020"].apply(tract_to_geoid)

tracts = gpd.read_file(CENSUS_FILE)
centroid = tracts.set_index("customGeoId")[["centroidLat", "centroidLon"]]
trips = trips.join(centroid.rename(columns={"centroidLat": "oLat", "centroidLon": "oLon"}), on="origin_geoid")
trips = trips.join(centroid.rename(columns={"centroidLat": "dLat", "centroidLon": "dLon"}), on="dest_geoid")
trips = trips.dropna(subset=["oLat", "oLon", "dLat", "dLon", "trip_start_time", "trip_duration_minutes"])
print(f"  {len(trips)} routable real trips")

real_out = []
n_ok = 0
for i, row in enumerate(trips.itertuples(), 1):
    h, m, s = map(int, row.trip_start_time.split(":"))
    t0 = h * 60 + m + s / 60
    dur = float(row.trip_duration_minutes)
    if dur <= 0 or dur > 180:
        continue
    coords = route_leg_latlng(row.oLat, row.oLon, row.dLat, row.dLon)
    if coords is None:
        continue
    t1 = t0 + dur
    mode_es = MODE_TO_ES.get(getattr(row, "primary_mode", None), "otro")
    real_out.append({
        "id": n_ok,
        "rutas_diarias": [
            {"hora": minutes_to_hora(t0), "minutos": round(t0, 2), "lugar": "origin", "lat": row.oLat, "lng": row.oLon, "transporte": mode_es, "tipo_lugar": "origen"},
            {"hora": minutes_to_hora(t1), "minutos": round(t1, 2), "lugar": "destination", "lat": row.dLat, "lng": row.dLon, "transporte": mode_es, "tipo_lugar": "destino"},
        ],
        "streetRoute": [
            {"startMin": round(t0, 2), "endMin": round(t1, 2), "transport": mode_es, "coords": [[round(a, 6), round(b, 6)] for a, b in coords]},
        ],
    })
    n_ok += 1
    if i % 500 == 0:
        print(f"  ...{i}/{len(trips)} processed, {n_ok} routed", flush=True)
print(f"  real profiles: {n_ok}")
json.dump(real_out, open(OUT_REAL, "w"))
print(f"  saved {OUT_REAL}")

# ---------------- SIMULATED agent trips (one profile per agent) ----------------
print("\nLoading simulated agents...")
agents = json.load(open(AGENTS_FILE))
sim_out = []
n_legs_ok = 0
n_legs_total = 0
for a_i, a in enumerate(agents, 1):
    home = a.get("home", {}).get("coordinates")
    if not home or len(home) != 2:
        continue
    try:
        home_lat, home_lng = float(home[0]), float(home[1])
    except (TypeError, ValueError):
        continue

    schedule = a.get("schedule", [])
    # Parse every stop first (times, coords, mode) regardless of routing success -
    # rutas_diarias gets ALL of them (interp() straight-lines across any gap
    # where streetRoute has no segment), only streetRoute needs the route.
    stops = []
    for stop in schedule:
        dest = (stop.get("destination") or {}).get("coordinates")
        if not dest or len(dest) != 2:
            continue
        try:
            lat, lng = float(dest[0]), float(dest[1])
            hh, mm = map(int, stop["start_time"].split(":"))
        except (TypeError, ValueError, KeyError):
            continue
        dur = DURATION_BUCKET_MID.get((stop.get("transportation") or {}).get("duration"), 15)
        mode = (stop.get("transportation") or {}).get("mode")
        stops.append({
            "t1": hh * 60 + mm, "dur": dur, "lat": lat, "lng": lng,
            "mode_es": MODE_TO_ES.get(mode, "otro"), "lugar": stop.get("desire", ""),
        })
    if not stops:
        continue

    first_t0 = stops[0]["t1"] - stops[0]["dur"]
    rutas = [{
        "hora": minutes_to_hora(first_t0), "minutos": round(first_t0, 2), "lugar": "home",
        "lat": home_lat, "lng": home_lng, "transporte": "a pie", "tipo_lugar": "casa",
    }]
    street_route = []
    prev_lat, prev_lng = home_lat, home_lng
    for s in stops:
        t0 = s["t1"] - s["dur"]
        rutas.append({
            "hora": minutes_to_hora(s["t1"]), "minutos": round(s["t1"], 2), "lugar": s["lugar"],
            "lat": s["lat"], "lng": s["lng"], "transporte": s["mode_es"], "tipo_lugar": s["lugar"],
        })
        n_legs_total += 1
        coords = route_leg_latlng(prev_lat, prev_lng, s["lat"], s["lng"])
        if coords is not None:
            street_route.append({
                "startMin": round(t0, 2), "endMin": round(s["t1"], 2), "transport": s["mode_es"],
                "coords": [[round(a, 6), round(b, 6)] for a, b in coords],
            })
            n_legs_ok += 1
        prev_lat, prev_lng = s["lat"], s["lng"]

    sim_out.append({"id": a_i - 1, "rutas_diarias": rutas, "streetRoute": street_route})
    if a_i % 100 == 0:
        print(f"  ...{a_i}/{len(agents)} agents processed, {n_legs_ok}/{n_legs_total} legs routed", flush=True)

print(f"  simulated profiles: {len(sim_out)}, legs routed: {n_legs_ok}/{n_legs_total}")
json.dump(sim_out, open(OUT_SIM, "w"))
print(f"  saved {OUT_SIM}")
