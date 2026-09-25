"""
Experiment I for Manhattan: mode/duration choice prediction against real eval trips.

Adapted from mobility_agent/exp1_step1_agent.ipynb (process_row/run_experiments) and
baseline/data.py::load_data - reused as-is, not reimplemented, since they're generic
(no Cambridge-specific logic). Unlike the full MobilityAgent pipeline we were fighting
with, this passes each eval row's REAL profile straight to MobilityAgent(profile=...),
which skips profile generation entirely (see agent.py: only calls _generate_profile()
`if not profile`) - one LLM call per row (get_mode_prefernce), not 10-16.

sample_num=50 matches what the paper actually found sufficient (and used for the full
Experiment II population), not the 5000 we built for exploring the graph earlier -
building this smaller reference graph is a one-time ~1-2 min cost (50 embedding calls),
cached separately from the 5000-sample graph.

EVAL_FILE points at a freshly-drawn 1000-row eval set (replica-manhattan_trips_eval_max60_1000.csv,
built by build_manhattan_eval_max60.py), not the original raw export. The paper's own
schema (Table 1) only ever offers the LLM/graph 6 duration buckets, 0-10 through 50-60 -
it implicitly assumes no trip takes over an hour. 11.6% of the original raw Manhattan
eval rows (116/1000) exceeded that, some by a lot (max 1415 minutes). Mode and duration
aren't chosen independently here - get_mode_prefernce always returns them as a single
(mode, duration) pair with one weight (see the graph's merged Intention nodes and the
choice_weights prompt format in agent.py) - so for those rows the correct (mode,
real-duration) pair was never even an option, which could bias the predicted MODE too,
not just duration. Rather than filter the original 1000 down to 884, this eval set was
redrawn fresh (same reference-graph exclusion, disjoint from the 5000 reference rows) to
keep the full N=1000 the paper itself used, while staying within the schema it was built
for - a documented methodological choice, not a silent exclusion.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import load_data
from mobility_agent.agent import MobilityAgent

CITY = "Manhattan, New York"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']

N_ROWS = 1000      # full paper-matching run, after clean 20/20 and 100/100 test runs
# Paper's own sweet spot (50) leaves rare modes like biking (~4% of Manhattan
# trips) at just ~1 person in the reference sample - confirmed empirically to
# NEVER recall biking (0/14 real biking rows) even when it's the correct
# answer (scripts/check_graph_recall.py). Swept sample_num in {50,100,200}
# with k=20 (graph.py's own default): 200 gives the best overall recall
# (96.5% vs 88% at 50) and fixes biking (0%->100%) and private_auto/on_demand_auto
# without introducing any other mode stuck at 0% recall on a meaningfully-sized
# subgroup - a deliberate, evidence-based deviation from the paper's default.
SAMPLE_NUM = 200


def process_row(row, num_samples):
    profile = row[PROFILE_COLUMNS].to_dict()
    desire = row['trip_purpose']
    time_ = row['start_time']
    # Real ground-truth distance from Replica (baseline/data.py now keeps this
    # column) - previously always 'unknown' since nothing upstream of
    # get_mode_prefernce ever computed it for this direct-from-eval-row path.
    distance = f"{row['trip_distance_miles']} miles"

    agent = MobilityAgent(
        profile=json.dumps(profile),
        sample_num=num_samples,
        city=CITY,
        reference_city=CITY,
        reference_file=REFERENCE_FILE,
    )
    agent.working_memory = ["Today is a normal weekday"]

    mode_preference = agent.get_mode_prefernce(
        desire=desire, time=time_, use_reference=(num_samples > 0), distance=distance
    )
    choice_weights = mode_preference['choice_weights']
    modes = [c['primary_mode'] for c in choice_weights]
    weights = [c['weight'] for c in choice_weights]
    selected = choice_weights[random.choices(range(len(modes)), weights=weights, k=1)[0]]

    return {
        'true_mode': row['primary_mode'],
        'true_duration': row['duration_minutes'],
        'predicted_mode': selected['primary_mode'],
        'predicted_duration': selected['duration_minutes'],
        'choice_weights': json.dumps(choice_weights),
    }


out_path = f"results/manhattan/exp1_test_{N_ROWS}.json"
os.makedirs("results/manhattan", exist_ok=True)

print(f"Loading eval data from {EVAL_FILE}...", flush=True)
eval_df = load_data(EVAL_FILE)
# random_state=42 is fixed, so this is the exact same 1000-row order every run -
# safe to resume from a previous partial run's checkpoint.
sample_df = eval_df.sample(n=N_ROWS, random_state=42)
print(f"Loaded {len(eval_df)} rows, testing on {N_ROWS} random rows, sample_num={SAMPLE_NUM}", flush=True)

# Resume support: the previous run got killed (session ended) at row 675/1000 -
# the incremental save every 25 rows meant that work wasn't lost. Pick up from
# there instead of redoing it.
results = json.load(open(out_path)) if os.path.exists(out_path) else []
start_at = len(results)
if start_at:
    print(f"Resuming from row {start_at+1}/{N_ROWS} ({len(results)} already saved)", flush=True)

t0 = time.time()
for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
    if i <= start_at:
        continue
    t_row = time.time()
    try:
        result = process_row(row, SAMPLE_NUM)
        results.append(result)
        match_mode = "OK" if result['true_mode'] == result['predicted_mode'] else "diff"
        n_match_so_far = sum(1 for r in results if r['true_mode'] == r['predicted_mode'])
        print(f"[{i}/{N_ROWS}] [{time.time()-t_row:.1f}s] true={result['true_mode']}/{result['true_duration']} "
              f"pred={result['predicted_mode']}/{result['predicted_duration']} ({match_mode}) "
              f"running_acc={n_match_so_far}/{len(results)}={n_match_so_far/len(results):.1%}", flush=True)
    except Exception as e:
        print(f"[{i}/{N_ROWS}] ERROR: {e}", flush=True)

    # save incrementally every 25 rows - a 1000-row run takes hours, don't lose
    # completed work if something crashes or gets killed partway through
    if i % 25 == 0 or i == N_ROWS:
        json.dump(results, open(out_path, "w"), indent=2)

print(f"\ntotal time: {time.time()-t0:.1f}s for {len(results)}/{N_ROWS} successful rows", flush=True)

n_mode_match = sum(1 for r in results if r['true_mode'] == r['predicted_mode'])
print(f"mode exact-match accuracy: {n_mode_match}/{len(results)} = {n_mode_match/max(len(results),1):.1%}", flush=True)

json.dump(results, open(out_path, "w"), indent=2)
print(f"saved to {out_path}")
