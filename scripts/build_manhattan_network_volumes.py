"""
Filter the raw Replica "Network Link Volumes" export (Geo Area: New York, NY -
covers much more than Manhattan, e.g. includes the Long Island Expressway in
Queens) down to segments actually within Manhattan, using the union of the 310
real census tract polygons we already have as the Manhattan boundary.

Every Experiment II simulated trip starts and ends within Manhattan (homes,
workplaces and POIs are all Manhattan-scoped by construction), so for
comparing simulated vs. real per-segment traffic flow, only Manhattan-internal
segments are relevant - a segment is kept only if BOTH its start and end point
fall inside Manhattan (drops bridge/tunnel segments that cross the boundary,
which our simulation never uses anyway).

Output keeps the exact same schema as the raw file / as replica-cambridge-roads.geojson:
networkLinkId, trip_count, roadName, direction, startLat, startLon, endLat, endLon,
highway, distance, osmId - so downstream routing code can treat Manhattan and
Cambridge identically.
"""
import json

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

RAW_FILE = "data/geo/manhattan_raw/replica-network-link-volumes-raw.geojson"
CENSUS_FILE = "data/geo/replica-manhattan-cencus.geojson"
OUT_FILE = "data/geo/replica-manhattan-roads-real.geojson"

print("Loading Manhattan boundary (union of 310 census tracts)...")
tracts = gpd.read_file(CENSUS_FILE)
manhattan_boundary = unary_union(tracts.geometry)

print(f"Loading raw network link volumes ({RAW_FILE})...")
raw = gpd.read_file(RAW_FILE)
print(f"n raw segments (region: New York, NY - wider than Manhattan): {len(raw)}")

print("Filtering to segments with both endpoints inside Manhattan...")
start_pts = gpd.GeoSeries([Point(lon, lat) for lat, lon in zip(raw["startLat"], raw["startLon"])], crs=raw.crs)
end_pts = gpd.GeoSeries([Point(lon, lat) for lat, lon in zip(raw["endLat"], raw["endLon"])], crs=raw.crs)
inside = start_pts.within(manhattan_boundary) & end_pts.within(manhattan_boundary)
filtered = raw[inside.values].copy()
print(f"n segments fully inside Manhattan: {len(filtered)} ({len(filtered)/len(raw):.1%} of raw)")

print("\nHighway class breakdown (Manhattan-filtered):")
print(filtered["highway"].value_counts())

filtered.to_file(OUT_FILE, driver="GeoJSON")
print(f"\nSaved {len(filtered)} Manhattan segments to {OUT_FILE}")
