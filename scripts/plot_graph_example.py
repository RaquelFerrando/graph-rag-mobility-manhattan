"""
Render an actual small example of the BehaviorGraph's retrieval subgraph,
using graph.py's own built-in visualize_graph() (already written for exactly
this purpose - colored edges by relationship type, agent/person/desire/
intention node layers, legend). Uses a small k=5 (not the production k=20)
purely so the diagram stays legible for a slide - the mechanism is identical.
"""
import json
import sys

sys.path.insert(0, ".")
import matplotlib.pyplot as plt

from mobility_agent.graph import BehaviorGraph

CITY_REF_FILE = "data/reference/replica-manhattan_trips.csv"
SAMPLE_NUM = 200

print("Loading/building the reference graph (cached, should be fast)...")
graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=CITY_REF_FILE)

profile = json.dumps({
    "age_group": "25-34", "income_group": "$50k-$100k", "employment_status": "employed",
    "household_size": 1, "available_vehicles": "zero", "education": "advanced_degree",
})
desire = "recreation"
time = 11

print("Querying a small (k=5) example subgraph for one real agent profile...")
subgraph, weights = graph.preference_modelling(profile=profile, desire=desire, time=time, k=5, depth=4)
print(f"Subgraph: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")

graph.visualize_graph(subgraph, desire=desire, time=time)
plt.savefig("results/manhattan/graph_example.png", dpi=180, bbox_inches="tight")
print("Saved results/manhattan/graph_example.png")
