"""
Concrete before/after numeric example of the choose_to (time closeness) weight fix
(mobility_agent/graph.py, see commit de73f73), using the REAL cached reference graph
and a REAL eval row (idx=926 in the exp1 1000-row sample - the "walking predicted as
other_travel_mode with 100% confidence" example already flagged in exp1_deepdive.json).

Reconstructs the exact subgraph get_mode_prefernce would have queried for this row
(same ref_file, sample_num=200, k=20, depth=4 as run_experiment1_manhattan.py), then
recomputes both the OLD (buggy) and NEW (fixed) choose_to weight from the same real
retrieved neighbors, to see whether the fix actually changes which neighbor dominates
for this specific case. No LLM generation call needed - _get_reference_graph only
needs the embedding model (already cached/available), the choose_to arithmetic itself
is pure math.
"""
import json
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from mobility_agent.graph import BehaviorGraph


def intention_weights(subgraph):
    """Reimplements preference_modelling's path-weight aggregation (graph.py:333-353)
    on a given subgraph, so it can be run twice (old vs. new choose_to weights)
    without needing a second _get_reference_graph call (no repeat embedding cost)."""
    agent_node = [n for n, d in subgraph.nodes(data=True) if d['type'] == 'agent'][0]
    intention_nodes = [n for n, d in subgraph.nodes(data=True) if d['type'] == 'intention']
    out = {}
    for i in intention_nodes:
        w = 0
        for path in nx.all_simple_paths(subgraph, source=agent_node, target=i, cutoff=4):
            pw = 1
            for n in range(len(path) - 1):
                pw *= subgraph[path[n]][path[n + 1]]['weight']
            w += pw
        props = subgraph.nodes[i]['props']
        out[(props['primary_mode'], props['duration_minutes'])] = w
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total else out

REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']
SAMPLE_NUM = 200
ROW_IDX = 926  # the walking -> other_travel_mode, 100% confidence example

eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=1000, random_state=42).reset_index(drop=True)
row = sample_df.iloc[ROW_IDX]

profile = row[PROFILE_COLUMNS].to_dict()
desire = row['trip_purpose']
query_time = row['start_time']

print(f"Row {ROW_IDX}: profile={profile}")
print(f"desire={desire!r}, query_time={query_time} (real mode was {row['primary_mode']!r})")
print()

graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=REFERENCE_FILE)
subgraph = graph._get_reference_graph(query_text=json.dumps(profile), desire=desire, time=query_time, k=20, depth=4)

# Find the desire node(s) reached, and for each, the intention(s) it connects to via choose_to
desire_nodes = [n for n, d in subgraph.nodes(data=True) if d['type'] == 'desire']

rows = []
for dn in desire_nodes:
    start_time = subgraph.nodes[dn]['props']['start_time']
    old_raw = abs(query_time - start_time) / 24
    new_raw = 1 - old_raw
    for nbr in subgraph.neighbors(dn):
        if subgraph.nodes[nbr]['type'] == 'intention':
            iprops = subgraph.nodes[nbr]['props']
            rows.append({
                'start_time': start_time,
                'gap_hours': abs(query_time - start_time),
                'mode': iprops['primary_mode'],
                'duration': iprops['duration_minutes'],
                'old_raw': old_raw,
                'new_raw': new_raw,
                'current_edge_weight_in_code': subgraph[dn][nbr]['weight'],  # already normalized w/ the FIXED formula
            })

if not rows:
    print("No choose_to edges found for this query (empty subgraph) - try a different ROW_IDX.")
    sys.exit(0)

max_old = max(r['old_raw'] for r in rows)
max_new = max(r['new_raw'] for r in rows)
for r in rows:
    r['old_normalized'] = r['old_raw'] / max_old if max_old else 0
    r['new_normalized'] = r['new_raw'] / max_new if max_new else 0

rows.sort(key=lambda r: -r['old_normalized'])

print(f"{len(rows)} choose_to edges (real neighbors) found in this query's subgraph.\n")
print(f"{'mode':<18}{'duration':<10}{'gap_hrs':>8}{'OLD weight':>12}{'NEW weight':>12}")
for r in rows:
    print(f"{r['mode']:<18}{r['duration']:<10}{r['gap_hours']:>8.2f}{r['old_normalized']:>12.3f}{r['new_normalized']:>12.3f}")

print()
print("Sanity check - current code (post-fix) 'weight' on these edges matches NEW column:",
      all(abs(r['current_edge_weight_in_code'] - r['new_normalized']) < 1e-9 for r in rows))

# --- Full prior_weight effect: aggregate across profile_similarity x desire_similarity x
# time_closeness paths, exactly like preference_modelling does, comparing OLD vs NEW ---
import copy
new_weights = intention_weights(subgraph)

old_subgraph = copy.deepcopy(subgraph)
for dn in desire_nodes:
    start_time = old_subgraph.nodes[dn]['props']['start_time']
    old_raw = abs(query_time - start_time) / 24
    for nbr in old_subgraph.neighbors(dn):
        if old_subgraph.nodes[nbr]['type'] == 'intention':
            old_subgraph[dn][nbr]['weight'] = old_raw
# renormalize choose_to edges by their own max, matching graph.py:322-330
max_old_edge = max(d['weight'] for u, v, d in old_subgraph.edges(data=True) if d['type'] == 'choose_to')
for u, v, d in old_subgraph.edges(data=True):
    if d['type'] == 'choose_to':
        old_subgraph[u][v]['weight'] = d['weight'] / max_old_edge
old_weights = intention_weights(old_subgraph)

print("\nFull prior_weight (profile_similarity x desire_similarity x time_closeness, summed across paths, normalized):\n")
all_keys = sorted(set(old_weights) | set(new_weights), key=lambda k: -new_weights.get(k, 0))
print(f"{'mode':<18}{'duration':<10}{'OLD prior_weight':>18}{'NEW prior_weight':>18}")
for mode, dur in all_keys:
    print(f"{mode:<18}{dur:<10}{old_weights.get((mode, dur), 0):>18.3f}{new_weights.get((mode, dur), 0):>18.3f}")

old_top = max(old_weights, key=old_weights.get)
new_top = max(new_weights, key=new_weights.get)
print(f"\nOLD formula's top pick: {old_top} ({old_weights[old_top]:.1%}) | real mode was: {row['primary_mode']}")
print(f"NEW formula's top pick: {new_top} ({new_weights[new_top]:.1%}) | real mode was: {row['primary_mode']}")
