"""
Compare the synthetic ProfileGeneratorTool's output distribution against real
Manhattan demographics, using the same aggregate KLD/MAE approach as
eval_manhattan_exp1.py - the question isn't "did we generate the right person"
(there's no such thing for a synthetic population), it's "does the population
of profiles we generate, in aggregate, look like real Manhattan?"
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, '.')

from baseline.data import categorize_age, categorize_income

SYNTHETIC_FILE = "results/manhattan/exp2_profile_calibration_sample.json"
REAL_FILE = "data/reference/replica-manhattan_trips.csv"


def aggregate_kl_mae(p_series, q_series):
    p_dist = p_series.value_counts(normalize=True)
    q_dist = q_series.value_counts(normalize=True)
    all_values = sorted(set(p_dist.index) | set(q_dist.index))
    p = np.array([p_dist.get(v, 0.0) for v in all_values])
    q = np.array([q_dist.get(v, 0.0) for v in all_values])
    mae = np.mean(np.abs(p - q))
    smooth = 1e-10
    p = (p + smooth) / (p + smooth).sum()
    q = (q + smooth) / (q + smooth).sum()
    return entropy(p, q), mae, all_values, p_dist, q_dist


synthetic = pd.DataFrame(json.load(open(SYNTHETIC_FILE)))
print(f"Loaded {len(synthetic)} synthetic profiles")

real_raw = pd.read_csv(REAL_FILE)
real = real_raw.drop_duplicates(subset='trip_taker_person_id').rename(columns={
    'trip_taker_age': 'age', 'trip_taker_employment_status': 'employment_status',
    'trip_taker_household_size': 'household_size', 'trip_taker_household_income': 'household_income',
    'trip_taker_available_vehicles': 'available_vehicles', 'trip_taker_education': 'education',
})
real['age_group_real'] = real['age'].apply(categorize_age)
print(f"Real reference population: {len(real)} unique people")

# household_income: real data is numeric (categorize_income buckets it into
# "$50k-$100k" style); synthetic is already bucketed but in a DIFFERENT
# schema ("50k-75k") - map both to a common coarse schema for a fair comparison.
def coarse_income_real(income_group):
    # baseline/data.py's categorize_income buckets: Under $10k / $10k-$50k / $50k-$100k / $100k-$150k / $150k-$200k / $200k-$300k / $300k+
    mapping = {
        'Under $10k': 'Under 25k', '$10k-$50k': 'Under 25k',  # coarsest common ground
        '$50k-$100k': '50k-100k', '$100k-$150k': '100k-150k',
        '$150k-$200k': '150k+', '$200k-$300k': '150k+', '$300k+': '150k+',
    }
    return mapping.get(income_group, 'unknown')

def coarse_income_synth(income_bucket):
    mapping = {
        'Under 25k': 'Under 25k', '25k-50k': 'Under 25k',
        '50k-75k': '50k-100k', '75k-100k': '50k-100k',
        '100k-150k': '100k-150k', '150k+': '150k+',
    }
    return mapping.get(income_bucket, 'unknown')

real['income_group_real'] = real['household_income'].apply(categorize_income).apply(coarse_income_real)
synthetic['income_group_synth'] = synthetic['household_income'].apply(coarse_income_synth)

comparisons = [
    ('age_group', real['age_group_real'], synthetic['age_group'] if 'age_group' in synthetic else None),
    ('employment_status', real['employment_status'], synthetic['employment_status']),
    ('education', real['education'], synthetic['education']),
    ('household_size', real['household_size'].clip(upper=7), synthetic['household_size'].clip(upper=7)),
    ('income (coarse)', real['income_group_real'], synthetic['income_group_synth']),
]

print(f"\n{'field':<20}{'KLD':>10}{'MAE':>10}")
for name, real_col, synth_col in comparisons:
    if synth_col is None:
        print(f"{name:<20}{'N/A (synthetic profile has no age_group field, only raw age)':>10}")
        continue
    kld, mae, values, p_dist, q_dist = aggregate_kl_mae(real_col, synth_col)
    print(f"{name:<20}{kld:>10.4f}{mae:>10.4f}")
    for v in values:
        print(f"    {str(v):<20} real={p_dist.get(v,0):.3f}  synthetic={q_dist.get(v,0):.3f}")
