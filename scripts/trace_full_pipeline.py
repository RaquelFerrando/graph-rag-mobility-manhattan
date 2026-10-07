"""
Full end-to-end trace of ONE synthetic agent's day, with every real LLM call
(profile correction, schedule generation, mode/duration choice, POI selection)
printed as it happens - a final sanity check before launching Experiment II
for real. Uses the paper's original synthetic ProfileGeneratorTool (profile=None),
now calibrated (see calibrate_profile_generator.py / eval_profile_calibration.py).
"""
import json
import sys

sys.path.insert(0, '.')

from openai.resources.chat.completions.completions import Completions

CITY = "Manhattan, New York"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"
SAMPLE_NUM = 200

call_n = [0]
_orig_parse = Completions.parse

def traced_parse(self, *args, **kwargs):
    call_n[0] += 1
    n = call_n[0]
    print(f"\n{'#'*70}\nLLM CALL #{n}")
    print(f"{'#'*70}")
    for m in kwargs.get("messages", []):
        print(f"--- {m['role']} ---")
        print(m['content'])
    response = _orig_parse(self, *args, **kwargs)
    print(f"--- RAW RESPONSE ---")
    print(response.choices[0].message.content)
    return response

Completions.parse = traced_parse

from mobility_agent.agent import MobilityAgent  # noqa: E402 (must patch before import triggers client creation... but patching the class method works regardless of instance creation order)

print("Creating a fully synthetic agent (profile=None -> ProfileGeneratorTool)...")
agent = MobilityAgent(
    profile=None, sample_num=SAMPLE_NUM,
    city=CITY, reference_city=CITY, reference_file=REFERENCE_FILE,
)
print(f"\n{'='*70}\nGENERATED PROFILE: {agent.profile}")
print(f"HOME: {agent.memory_locations.get('home')}")
print(f"{'='*70}")

schedule = agent.get_time_schedule(use_reference=True)

print(f"\n{'='*70}\nFINAL SCHEDULE ({len(schedule)} stops):")
for stop in schedule:
    print(f"  {stop}")
print(f"\nTotal LLM calls made: {call_n[0]}")
