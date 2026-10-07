"""
Calibration check for ProfileGeneratorTool (the paper's original synthetic
population generator) before committing to a full Experiment II run: does the
distribution of profiles it actually produces (after the LLM consistency-
correction step) match real Manhattan demographics? And how often does the
independently-sampled age/employment_status combination end up contradictory
(e.g. "45-54" + "under_16") and go uncorrected?

Cheap relative to a full agent (1 LLM call per profile here, vs ~10-16 for a
full simulated day), so run this BEFORE spending hours on Experiment II.
"""
import json
import sys

sys.path.insert(0, '.')

from mobility_agent.tools import ProfileGeneratorTool

N = 100  # first pass: see what actually comes back before committing to a bigger N

tool = ProfileGeneratorTool()
tool.client.timeout = 120

raw_profiles = []
for i in range(N):
    result = tool.generate_profile(city="Manhattan, New York")
    # generate_profile returns either an AgentProfile instance (client=None path,
    # not used here) or a raw JSON string (LLM correction path) - handle both.
    profile = json.loads(result) if isinstance(result, str) else result.model_dump()
    raw_profiles.append(profile)
    print(f"[{i+1}/{N}] {profile}", flush=True)

print(f"\nuncorrected_profile_count: {tool.uncorrected_profile_count}/{N}")
json.dump(raw_profiles, open("results/manhattan/exp2_profile_calibration_sample.json", "w"), indent=2)
print("saved to results/manhattan/exp2_profile_calibration_sample.json")
