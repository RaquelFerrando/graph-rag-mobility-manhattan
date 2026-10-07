"""
Diagnostic: for how many real Manhattan queries does the graph's prior_choices
even include 'biking' (or other rare modes) as an option at all? Only hits the
embedding model, not the main LLM - cheap, no need to touch the 1000-row results.

Hypothesis to test: with sample_num=50 and biking at only ~4% of the population
(1-2 people in any 50-person sample), the k=20 similarity search + depth=4 path
search may very often retrieve zero biking-related paths for a given query -
and critically, get_mode_prefernce's reference_prompt (agent.py use_reference=True
branch) ONLY shows prior_choices, never the full mode list like the no-reference
branch does. So if biking isn't retrieved for a query, the LLM is never even told
it's a valid option for that query - not "biased against it", structurally absent.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from mobility_agent.graph import BehaviorGraph

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']
N_CHECK = 100

# Usage: python check_graph_mode_coverage.py [sample_num] [k]
SAMPLE_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20

eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=1000, random_state=42).iloc[:N_CHECK]

print(f"Building graph with sample_num={SAMPLE_NUM}, querying with k={K}...", flush=True)
graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=REFERENCE_FILE)

mode_appears = Counter()
n_no_biking = 0
n_rows_with_real_biking = 0

for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
    profile = json.dumps(row[PROFILE_COLUMNS].to_dict())
    desire = row['trip_purpose']
    time_ = row['start_time']
    _, weights = graph.preference_modelling(profile=profile, desire=desire, time=time_, k=K, depth=4)
    modes_in_prior = set(w['primary_mode'] for w in weights)
    for m in modes_in_prior:
        mode_appears[m] += 1
    if 'biking' not in modes_in_prior:
        n_no_biking += 1
    if row['primary_mode'] == 'biking':
        n_rows_with_real_biking += 1
    if i % 20 == 0:
        print(f"[{i}/{N_CHECK}] processed", flush=True)

print(f"\nOut of {N_CHECK} real queries:")
print(f"  'biking' was MISSING from prior_choices in {n_no_biking}/{N_CHECK} queries "
      f"({n_no_biking/N_CHECK:.0%})")
print(f"  {n_rows_with_real_biking} of these {N_CHECK} rows had a REAL true_mode of 'biking'")
print(f"\nHow often each mode appeared in ANY query's prior_choices (out of {N_CHECK} queries):")
for mode, count in mode_appears.most_common():
    print(f"  {mode}: {count}/{N_CHECK} ({count/N_CHECK:.0%})")
