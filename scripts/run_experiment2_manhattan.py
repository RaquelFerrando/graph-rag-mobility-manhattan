"""
Experiment II for Manhattan: full daily-agent simulation (schedule generation +
mode/duration choice + POI selection per stop), matching the paper's Mobility
Agent pipeline (Fig 8) - unlike Experiment I, each "agent" here lives a FULL
DAY, not a single trip, so it costs ~10-16 LLM calls instead of 1.

Uses the paper's ORIGINAL synthetic ProfileGeneratorTool (profile=None), not
real individual profiles - a deliberate choice (see conversation/README) to
stay faithful to what the paper's own Experiment II demonstrates (a synthetic
population built from aggregate statistics, not real individual records).
This is only trustworthy because the generator was calibrated first:
  - age_ranges rebuilt from the full ~889k-person People export with complete
    coverage (0-120), closing a gap where under-18/over-85 fell outside every
    bucket (scripts/build_manhattan_demographics.py)
  - age/employment_status are sampled CONDITIONALLY (not independently) so a
    minor's employment_status is always 'under_16' by construction, not left
    to an unreliable LLM correction pass (mobility_agent/tools.py)
  - AgentProfile's employment_status/education/household_income are
    Literal-constrained so the LLM correction step can't drift into invented
    categories (confirmed empirically: "retired"/"homemaker"/"bachelor" etc.
    before this fix)
  - verified via scripts/eval_profile_calibration.py: KLD/MAE against real
    Manhattan demographics now in the same range across all fields (age/
    employment/education/income/household_size), with 0/100 age-employment
    mismatches in the final calibration run.

Designed to be incremental in AGENT COUNT: saves after every single completed
agent (not batched like Experiment I's every-25-rows, since losing one of
these expensive agents hurts far more than losing one cheap Experiment I row),
and downstream analysis normalizes by however many agents actually finished -
so stopping at 300 or 500 instead of 1000 still produces a valid, usable
result, not a wasted partial run.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobility_agent.agent import MobilityAgent

CITY = "Manhattan, New York"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
SAMPLE_NUM = 200  # matches Experiment I's tuned value (README/check_graph_recall.py)
N_AGENTS = 1000   # target - safe to stop early, see module docstring

out_path = f"results/manhattan/exp2_agents_{N_AGENTS}.json"
os.makedirs("results/manhattan", exist_ok=True)
results = json.load(open(out_path)) if os.path.exists(out_path) else []
start_at = len(results)
if start_at:
    print(f"Resuming from agent {start_at+1}/{N_AGENTS} ({start_at} already saved)", flush=True)

t0 = time.time()
for i in range(1, N_AGENTS + 1):
    if i <= start_at:
        continue
    t_agent = time.time()
    try:
        # profile=None -> MobilityAgent.__init__ calls _generate_profile(),
        # which uses ProfileGeneratorTool to sample a calibrated synthetic
        # profile AND assign a real (population-weighted random) home census
        # tract - both happen before get_time_schedule() is ever called, so
        # its "home not in memory_locations" fallback never fires here.
        agent = MobilityAgent(
            profile=None, sample_num=SAMPLE_NUM,
            city=CITY, reference_city=CITY, reference_file=REFERENCE_FILE,
        )
        agent.working_memory = ["Today is a normal weekday"]

        schedule = agent.get_time_schedule(use_reference=True)
        results.append({
            'profile': agent.profile,
            'home': agent.memory_locations.get('home'),
            'schedule': schedule,
        })
        print(f"[{i}/{N_AGENTS}] [{time.time()-t_agent:.1f}s] {len(schedule)} stops saved", flush=True)
    except Exception as e:
        print(f"[{i}/{N_AGENTS}] ERROR: {e}", flush=True)

    # Save after EVERY agent, not batched - each one costs ~10-16 LLM calls,
    # losing even one to a crash is expensive (unlike Experiment I's cheap
    # single-call rows, where batching every 25 was an acceptable tradeoff).
    json.dump(results, open(out_path, "w"), indent=2)

print(f"\ntotal time: {time.time()-t0:.1f}s for {len(results)} agents saved", flush=True)
print(f"saved to {out_path}")
