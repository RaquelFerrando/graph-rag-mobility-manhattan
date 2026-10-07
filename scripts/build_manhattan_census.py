"""
Build data/geo/replica-manhattan-cencus.geojson for mobility_agent_ny.

Sources (both free, no Replica export needed):
- Tract polygons: NYC Open Data "2020 Census Tracts" (data.cityofnewyork.us/resource/63ge-mke6),
  filtered to boroname=Manhattan -> data/geo/manhattan_tracts_raw.geojson (already WGS84 lon/lat)
- Tract population: Census Bureau ACS 5-year 2022, table B01003_001E, for New York County (36061)
  -> data/geo/acs_manhattan_population_raw.json

Output schema matches what mobility_agent/tools.py::ProfileGeneratorTool.get_home() actually reads:
  customGeoId (int, = GEOID), geometry (Polygon/MultiPolygon), peopleByHome (int, ACS total population).
geoName/centroidLat/centroidLon/squareMiles are kept only for parity with the original Cambridge file;
they are not consumed by any code in mobility_agent. No geopandas/shapely dependency (not installed
in this environment) - pure json, matching how enrich_replica_coords.py/join_replica_safegraph.py
were written earlier in this project.
"""
import json

GEO_DIR = "data/geo"

tracts = json.load(open(f"{GEO_DIR}/manhattan_tracts_raw.geojson"))

acs_raw = json.load(open(f"{GEO_DIR}/acs_manhattan_population_raw.json"))
header, *rows = acs_raw
idx = {name: i for i, name in enumerate(header)}
pop_by_geoid = {
    row[idx["state"]] + row[idx["county"]] + row[idx["tract"]]: int(row[idx["B01003_001E"]])
    for row in rows
}


def bbox_center(geometry):
    """Rough (non-area-weighted) centroid via bounding box. Only used for an informational
    field the pipeline doesn't read, so exact area-weighted centroid isn't needed."""
    coords_type = geometry["type"]
    rings = geometry["coordinates"]
    if coords_type == "Polygon":
        polygons = [rings]
    elif coords_type == "MultiPolygon":
        polygons = rings
    else:
        raise ValueError(f"unexpected geometry type {coords_type}")

    lons, lats = [], []
    for polygon in polygons:
        for ring in polygon:
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
    return (sum(lons) / len(lons), sum(lats) / len(lats)), (
        min(lons), min(lats), max(lons), max(lats)
    )


features_out = []
missing = []
for feat in tracts["features"]:
    props = feat["properties"]
    geoid = props["geoid"]
    pop = pop_by_geoid.get(geoid)
    if pop is None:
        missing.append(geoid)
        continue

    (centroid_lon, centroid_lat), _bounds = bbox_center(feat["geometry"])
    square_miles = round(float(props["shape_area"]) / 27_878_400, 4)  # NYC shape_area is in sq ft

    features_out.append({
        "type": "Feature",
        "properties": {
            "customGeoId": int(geoid),
            "geoName": f"Tract {props['ctlabel']}, New York County, NY",
            "centroidLat": round(centroid_lat, 4),
            "centroidLon": round(centroid_lon, 4),
            "squareMiles": square_miles,
            "peopleByHome": pop,
        },
        "geometry": feat["geometry"],
    })

if missing:
    raise ValueError(f"{len(missing)} tracts have no ACS population match: {missing}")

out = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": features_out,
}

out_path = f"{GEO_DIR}/replica-manhattan-cencus.geojson"
json.dump(out, open(out_path, "w"))

total_pop = sum(f["properties"]["peopleByHome"] for f in features_out)
print(f"wrote {len(features_out)} tracts to {out_path}")
print(f"total peopleByHome (Manhattan pop, ACS 2022 5yr): {total_pop:,}")
for f in features_out[:5]:
    print(f["properties"])
