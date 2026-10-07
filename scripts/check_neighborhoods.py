"""
Reverse-geocode each agent's home coordinate to a real Manhattan neighborhood
name (via Nominatim/OSM), then compare how many agents landed in each
neighborhood against how much of Manhattan's real population lives there
(summed peopleByHome across that neighborhood's census tracts) - a much more
intuitive check than a single aggregate density number.
"""
import json
import time

import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Point

agents = json.load(open("results/manhattan/exp2_agents_1000.json"))
homes = [tuple(a["home"]["coordinates"]) for a in agents if a.get("home")]
print(f"n agentes con casa: {len(homes)}")

geolocator = Nominatim(user_agent="mobility_agent_ny_neighborhood_check")

agent_neighborhoods = []
for i, (lat, lon) in enumerate(homes, 1):
    try:
        loc = geolocator.reverse((lat, lon), exactly_one=True, addressdetails=True, timeout=10)
        addr = loc.raw.get("address", {}) if loc else {}
        neighborhood = addr.get("neighbourhood") or addr.get("suburb") or addr.get("quarter") or "unknown"
    except Exception as e:
        neighborhood = "unknown"
    agent_neighborhoods.append(neighborhood)
    if i % 50 == 0:
        print(f"  ...{i}/{len(homes)} geocoded", flush=True)
    time.sleep(1.0)  # Nominatim usage policy: max 1 req/sec

from collections import Counter
agent_counts = Counter(agent_neighborhoods)

print(f"\nDistribucion de agentes por barrio (n={len(homes)}):")
for name, count in agent_counts.most_common(20):
    print(f"  {name:<30} {count:>4} ({count/len(homes):.1%})")

json.dump(agent_neighborhoods, open("results/manhattan/exp2_agent_neighborhoods.json", "w"), indent=2)
print("\nsaved to results/manhattan/exp2_agent_neighborhoods.json")
