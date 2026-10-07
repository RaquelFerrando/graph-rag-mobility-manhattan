"""
Print the EXACT prompt text that get_mode_prefernce would send to the LLM for a
real eval row, using the currently-cached graph (reuses the pickle, no new
embedding calls beyond the query itself) - without calling the LLM.
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
SAMPLE_NUM = 100  # reuses the already-cached graph from the sweep

eval_df = load_data(EVAL_FILE)
row = eval_df.sample(n=1000, random_state=42).iloc[0]

profile = row[PROFILE_COLUMNS].to_dict()
profile_str = json.dumps(profile)
desire = row['trip_purpose']
time_num = row['start_time']
distance = f"{row['trip_distance_miles']} miles"
working_memory = ["Today is a normal weekday"]
reference_city = city = "Manhattan, New York"

graph = BehaviorGraph(sample_num=SAMPLE_NUM, ref_file=REFERENCE_FILE)
_, weights = graph.preference_modelling(profile=profile_str, desire=desire, time=time_num)
prior_choices = sorted(
    [
        {'primary_mode': d['primary_mode'], 'duration_minutes': d['duration_minutes'], 'prior_weight': round(float(d['weight']), 3)}
        for d in weights
    ],
    key=lambda c: c['prior_weight'], reverse=True,
)

hours = int(time_num)
minutes = int((time_num - hours) * 60)
time_str = f"{hours:02d}:{minutes:02d}"

reference_prompt = f"""Available Mode Choices : ['walking', 'biking', 'auto_passenger','public_transit','private_auto', 'on_demand_auto','other_travel_mode']
        Available Duration Choices :[ '0-10','10-20','20-30', '30-40', '40-50', '50-60']

        Context: This is the reference behavior of people with similar profiles in {reference_city} in a normal workday, computed from a graph-based prior model. Higher prior_weight means more likely to be chosen by people with similar profile. A combination not listed below simply has no data from similar reference people in this case - it is not excluded.
        Reference choices with their prior probability (treat these as a starting point, not a fixed answer):
        {prior_choices}
        """
cultural_note = (
    f" the possible cultural difference between {reference_city} and {city}, and"
    if reference_city != city else ""
)
reference_note = f"- Start from the prior_weight values above, then adjust them based on your profile,{cultural_note} the weather (if provided). You may also pick an available option that has no prior_weight if it genuinely fits your profile better. Briefly note in your reasoning if and why you deviated from the prior."

choice_template = f""" You are {profile_str}, tasked with simulating realistic transportation behavior based on reference data from people with similar profiles.

Current Time: {time_str}
Current Desire: {desire}
Current memory: {working_memory}
Target Distance: {distance}
{reference_prompt}

What transportation mode and duration minutes do you prefer?

Note:
- Output your reason and preference weight of different choices
- The sum of all weights should equal 1.
{reference_note}

Answer Format:
{{
'think': [list your reason in short sentences (up to 4 points)]
'choice_weights':[
{{'primary_mode': mode1, 'duration_minutes': duration1, 'weight': weight1 }},
{{'primary_mode': mode2, 'duration_minutes': duration2, 'weight': weight2 }},
...
]
}}
"""

print(f"[real trip_purpose={desire}, true primary_mode={row['primary_mode']}, true duration={row['duration_minutes']}, sample_num={SAMPLE_NUM}]\n")
print(choice_template)
