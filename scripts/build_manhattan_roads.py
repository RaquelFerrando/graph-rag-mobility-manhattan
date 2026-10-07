"""
Download Manhattan's real street network from OpenStreetMap (via osmnx) for
routing Experiment II's simulated trips - independent of whether a real
Replica "roads" export (with real per-segment trip_count, like
replica-cambridge-roads.geojson has) becomes available. Without real
trip_count, this can't reproduce the paper's KLD traffic-flow metric
(Fig 9a-c), but it does let us (a) route simulated trips onto real streets
and (b) see qualitatively which streets attract the most simulated traffic.

Saved in the same GeoJSON LineString-per-edge shape as
data/geo/replica-cambridge-roads.geojson (minus trip_count, which OSM
doesn't have) so the same downstream routing/aggregation code can read either.
"""
import json

import osmnx as ox

PLACE = "Manhattan, New York County, New York, USA"
OUT_PATH = "data/geo/replica-manhattan-roads.geojson"

print(f"Downloading drivable+walkable street network for: {PLACE}", flush=True)
G = ox.graph_from_place(PLACE, network_type="all", simplify=True)
print(f"Downloaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges", flush=True)

features = []
for u, v, k, data in G.edges(keys=True, data=True):
    geom = data.get("geometry")
    if geom is not None:
        coords = list(geom.coords)
    else:
        # no explicit geometry - straight line between the two endpoint nodes
        coords = [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
    features.append({
        "type": "Feature",
        "properties": {
            "osmId": data.get("osmid"),
            "roadName": data.get("name"),
            "highway": data.get("highway"),
            "distance_m": data.get("length"),
            "startLat": coords[0][1], "startLon": coords[0][0],
            "endLat": coords[-1][1], "endLon": coords[-1][0],
        },
        "geometry": {"type": "LineString", "coordinates": coords},
    })

geojson = {"type": "FeatureCollection", "features": features}
json.dump(geojson, open(OUT_PATH, "w"))
print(f"Saved {len(features)} street segments to {OUT_PATH}", flush=True)
