"""
Sanity check: does the embedding-based similarity search actually retrieve
genuinely similar reference people, or is it noise? Prints, for a few real
query profiles, the top-5 "most similar" reference people's actual attributes
side by side with the query, so it can be eyeballed directly.

Worth double-checking one representational mismatch while at it: the QUERY
profile uses bucketed categories (age_group='25-34', income_group='$50k-$100k',
from baseline/data.py), but graph.py's Person nodes embed RAW age (a number)
and RAW household_income (a number), not the same buckets - see graph.py's
attribute_dict['person']. The embedding model has to bridge that representational
gap on its own; this script surfaces whether that's actually working or not.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from mobility_agent.graph import BehaviorGraph

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']
SAMPLE_NUM = 200
K = 5
N_QUERIES = 1

eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=1000, random_state=42)

print(f"Building graph with sample_num={SAMPLE_NUM}...", flush=True)
graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=REFERENCE_FILE)

person_props = {n: d['props'] for n, d in graph.behavior_graph.nodes(data=True) if d.get('type') == 'person'}

for qi in range(N_QUERIES):
    row = sample_df.iloc[qi]
    profile = row[PROFILE_COLUMNS].to_dict()
    query_text = json.dumps(profile)
    print(f"\n{'='*80}")
    print(f"QUERY {qi}: {profile}")
    print(f"(this is the text actually embedded: {query_text})")

    results = graph._similarity_search(query_text=query_text, k=K)
    print(f"\nTop-{K} most similar reference people:")
    for node_id, score in results:
        p = person_props[node_id]
        print(f"  score={score:.4f}  name={p['name']:<20} age_group={p['age_group']:<10} gender={p['gender']:<7} "
              f"employment={p['employment_status']:<18} household_size={p['household_size']:<3} "
              f"income_group={p['income_group']:<12} available_vehicles={p['available_vehicles']:<20} "
              f"education={p['education']}")
