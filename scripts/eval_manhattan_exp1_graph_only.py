"""
Ablation for Manhattan Experiment I: what does the aggregate mode/duration
distribution look like if we skip the LLM entirely and just sample directly
from the graph's own retrieval-based prior (prior_weight)?

This isolates how much of the full pipeline's accuracy comes from the graph
(profile similarity x desire similarity x time closeness over the 200-person
reference sample, k=20 nearest neighbors - see graph.py) versus the LLM's own
reasoning on top of that prior. Same eval rows, same sample_num=200/k=20,
same random-choices-by-weight sampling method as run_experiment1_manhattan.py,
so it's directly comparable to results/manhattan/exp1_test_1000.json.

Cheap and fast: no LLM generation calls at all, only embedding lookups
(mxbai-embed-large) for the profile/desire similarity search - a few seconds
per row instead of the ~10-60s a full get_mode_prefernce LLM call took.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from mobility_agent.graph import BehaviorGraph

CITY = "Manhattan, New York"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']
N_ROWS = 1000
SAMPLE_NUM = 200  # same tuned value as run_experiment1_manhattan.py

out_path = f"results/manhattan/exp1_graph_only_{N_ROWS}.json"
os.makedirs("results/manhattan", exist_ok=True)

print(f"Loading eval data from {EVAL_FILE}...", flush=True)
eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=N_ROWS, random_state=42)  # identical rows/order to Experiment I

print(f"Building reference graph (sample_num={SAMPLE_NUM})...", flush=True)
graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=REFERENCE_FILE)

results = json.load(open(out_path)) if os.path.exists(out_path) else []
start_at = len(results)
if start_at:
    print(f"Resuming from row {start_at+1}/{N_ROWS} ({start_at} already saved)", flush=True)

t0 = time.time()
for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
    if i <= start_at:
        continue
    t_row = time.time()
    try:
        profile = json.dumps(row[PROFILE_COLUMNS].to_dict())
        _, weights = graph.preference_modelling(profile=profile, desire=row['trip_purpose'], time=row['start_time'])
        modes = [w['primary_mode'] for w in weights]
        w_vals = [w['weight'] for w in weights]
        selected = weights[random.choices(range(len(modes)), weights=w_vals, k=1)[0]]
        results.append({
            'true_mode': row['primary_mode'],
            'true_duration': row['duration_minutes'],
            'predicted_mode': selected['primary_mode'],
            'predicted_duration': selected['duration_minutes'],
            'choice_weights': json.dumps([{'primary_mode': w['primary_mode'], 'duration_minutes': w['duration_minutes'], 'weight': float(w['weight'])} for w in weights]),
        })
        print(f"[{i}/{N_ROWS}] [{time.time()-t_row:.1f}s] true={row['primary_mode']}/{row['duration_minutes']} "
              f"pred={selected['primary_mode']}/{selected['duration_minutes']}", flush=True)
    except Exception as e:
        print(f"[{i}/{N_ROWS}] ERROR: {e}", flush=True)

    if i % 25 == 0 or i == N_ROWS:
        json.dump(results, open(out_path, "w"), indent=2)

print(f"\ntotal time: {time.time()-t0:.1f}s for {len(results)}/{N_ROWS} rows", flush=True)
json.dump(results, open(out_path, "w"), indent=2)
print(f"saved to {out_path}")
