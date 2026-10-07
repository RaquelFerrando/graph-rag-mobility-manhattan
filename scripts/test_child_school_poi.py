"""
Targeted check: does a child agent now get a K-12 school instead of a
college/university for desire='school', after the NAICS-matching fix?
"""
import json
import sys

sys.path.insert(0, '.')

from openai.resources.chat.completions.completions import Completions

_orig_parse = Completions.parse
_orig_create = Completions.create

def traced_parse(self, *args, **kwargs):
    print(f"\n{'#'*70}\n--- [.parse] messages sent ---")
    for m in kwargs.get("messages", []):
        print(f"[{m['role']}] {m['content'][:600]}")
    response = _orig_parse(self, *args, **kwargs)
    msg = response.choices[0].message
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"--- TOOL CALL: {tc.function.name}({tc.function.arguments}) ---")
    else:
        print(f"--- RESPONSE ---\n{msg.content}")
    return response

def traced_create(self, *args, **kwargs):
    print(f"\n{'#'*70}\n--- [.create, tools={bool(kwargs.get('tools'))}] messages sent ---")
    for m in kwargs.get("messages", []):
        print(f"[{m['role']}] {str(m['content'])[:800]}")
    response = _orig_create(self, *args, **kwargs)
    msg = response.choices[0].message
    print(f"--- finish_reason: {response.choices[0].finish_reason} ---")
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"--- TOOL CALL: {tc.function.name}({tc.function.arguments}) ---")
    else:
        print(f"--- NO TOOL CALL. raw content: {msg.content!r} ---")
    return response

Completions.parse = traced_parse
Completions.create = traced_create

from mobility_agent.agent import MobilityAgent

CITY = "Manhattan, New York"
REFERENCE_FILE = "data/reference/replica-manhattan_trips.csv"

child_profile = {
    "age_group": "Under 18", "income_group": "$100k-$150k", "employment_status": "under_16",
    "household_size": 2, "available_vehicles": "zero", "education": "k_12",
}
agent = MobilityAgent(
    profile=json.dumps(child_profile), sample_num=200,
    city=CITY, reference_city=CITY, reference_file=REFERENCE_FILE,
)
result = agent.get_location_choice(current_lat=40.7858, current_lon=-73.9765, desire="school", time=8.5, use_reference=True)
print(f"\n{'='*70}\nFINAL RESULT: {result}")
