"""
Correct version of the coverage diagnostic: instead of asking "does mode X appear
in ANY query's prior_choices" (which conflates queries where X genuinely isn't
relevant with queries where the graph failed to retrieve it), ask the targeted
question: for the rows where the REAL true_mode actually was X, does the graph's
prior_choices for THAT row include X at all (recall), and is it the top-weighted
option (top-1)? This isolates real retrieval failures from cases where a mode
is correctly absent because it wasn't a relevant choice for that trip.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from mobility_agent.graph import BehaviorGraph

EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']

# Usage: python check_graph_recall.py [sample_num] [k] [n_check]
SAMPLE_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_CHECK = int(sys.argv[3]) if len(sys.argv) > 3 else 200

eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=1000, random_state=42).iloc[:N_CHECK]

print(f"Building graph with sample_num={SAMPLE_NUM}, querying with k={K}, n_check={N_CHECK}...", flush=True)
graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=REFERENCE_FILE)

# per true_mode: how many rows, how many had it recalled anywhere, how many had it as top-1
stats = defaultdict(lambda: {'n': 0, 'recalled': 0, 'top1': 0})

for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
    profile = json.dumps(row[PROFILE_COLUMNS].to_dict())
    desire = row['trip_purpose']
    time_ = row['start_time']
    true_mode = row['primary_mode']

    _, weights = graph.preference_modelling(profile=profile, desire=desire, time=time_, k=K, depth=4)
    stats[true_mode]['n'] += 1
    if not weights:
        pass
    else:
        modes_present = set(w['primary_mode'] for w in weights)
        if true_mode in modes_present:
            stats[true_mode]['recalled'] += 1
        top_choice = max(weights, key=lambda w: w['weight'])
        if top_choice['primary_mode'] == true_mode:
            stats[true_mode]['top1'] += 1

    if i % 40 == 0:
        print(f"[{i}/{N_CHECK}] processed", flush=True)

print(f"\nRecall of the REAL true_mode in prior_choices, by mode (sample_num={SAMPLE_NUM}, k={K}, n={N_CHECK}):")
print(f"{'mode':<18}{'n_rows':>8}{'recalled':>10}{'recall%':>9}{'top1':>7}{'top1%':>8}")
total_n = total_recalled = total_top1 = 0
for mode, s in sorted(stats.items(), key=lambda x: -x[1]['n']):
    recall_pct = s['recalled'] / s['n'] * 100 if s['n'] else 0
    top1_pct = s['top1'] / s['n'] * 100 if s['n'] else 0
    print(f"{mode:<18}{s['n']:>8}{s['recalled']:>10}{recall_pct:>8.1f}%{s['top1']:>7}{top1_pct:>7.1f}%")
    total_n += s['n']
    total_recalled += s['recalled']
    total_top1 += s['top1']
print(f"{'TOTAL':<18}{total_n:>8}{total_recalled:>10}{total_recalled/total_n*100:>8.1f}%{total_top1:>7}{total_top1/total_n*100:>7.1f}%")
