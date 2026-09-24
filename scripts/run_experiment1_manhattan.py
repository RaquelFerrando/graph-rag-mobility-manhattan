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

EVAL_FILE points at a filtered eval set (replica-manhattan_trips_eval_max60.csv), not
the raw 1000-row export. The paper's own schema (Table 1) only ever offers the LLM/graph
6 duration buckets, 0-10 through 50-60 - it implicitly assumes no trip takes over an
hour. 11.6% of our raw Manhattan eval rows (116/1000) exceed that, some by a lot (max
1415 minutes). Mode and duration aren't chosen independently here - get_mode_prefernce
always returns them as a single (mode, duration) pair with one weight (see the graph's
merged Intention nodes and the choice_weights prompt format in agent.py) - so for those
rows the correct (mode, real-duration) pair was never even an option, which could bias
the predicted MODE too, not just duration. Filtering to <=60min (884/1000 rows) keeps
this reproduction comparable to what the paper's schema was actually built for; this is
a documented methodological choice, not a silent exclusion.
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
EVAL_FILE = "data/eval/replica-manhattan_trips_eval_max60.csv"
PROFILE_COLUMNS = ['age_group', 'income_group', 'employment_status', 'household_size', 'available_vehicles', 'education']

N_ROWS = 100       # scaled up after the clean 20-row run (20/20 ok, ~21s/row)
SAMPLE_NUM = 50    # paper's own sweet spot for reference graph size


def process_row(row, num_samples):
    profile = row[PROFILE_COLUMNS].to_dict()
    desire = row['trip_purpose']
    time_ = row['start_time']

    agent = MobilityAgent(
        profile=json.dumps(profile),
        sample_num=num_samples,
        city=CITY,
        reference_city=CITY,
        reference_file=REFERENCE_FILE,
    )
    agent.working_memory = ["Today is a normal weekday"]

    mode_preference = agent.get_mode_prefernce(
        desire=desire, time=time_, use_reference=(num_samples > 0)
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


print(f"Loading eval data from {EVAL_FILE}...")
eval_df = load_data(EVAL_FILE)
sample_df = eval_df.sample(n=N_ROWS, random_state=42)
print(f"Loaded {len(eval_df)} rows, testing on {N_ROWS} random rows, sample_num={SAMPLE_NUM}")

results = []
t0 = time.time()
for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
    t_row = time.time()
    try:
        result = process_row(row, SAMPLE_NUM)
        results.append(result)
        match_mode = "OK" if result['true_mode'] == result['predicted_mode'] else "diff"
        print(f"[{i}/{N_ROWS}] [{time.time()-t_row:.1f}s] true={result['true_mode']}/{result['true_duration']} "
              f"pred={result['predicted_mode']}/{result['predicted_duration']} ({match_mode})")
    except Exception as e:
        print(f"[{i}/{N_ROWS}] ERROR: {e}")

print(f"\ntotal time: {time.time()-t0:.1f}s for {len(results)}/{N_ROWS} successful rows")

n_mode_match = sum(1 for r in results if r['true_mode'] == r['predicted_mode'])
print(f"mode exact-match accuracy: {n_mode_match}/{len(results)} = {n_mode_match/max(len(results),1):.1%}")

os.makedirs("results/manhattan", exist_ok=True)
out_path = f"results/manhattan/exp1_test_{N_ROWS}.json"
json.dump(results, open(out_path, "w"), indent=2)
print(f"saved to {out_path}")
