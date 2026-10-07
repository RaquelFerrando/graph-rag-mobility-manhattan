"""
Build data/geo/replica-manhattan-demographics.json for mobility_agent_ny, in the exact
schema ProfileGeneratorTool.generate_profile() expects (same as
data/geo/replica-cambrdige-demographics.json), from the Manhattan People export.

The People export is one row per TRIP (not per person - same schema as the trips export),
so we deduplicate by trip_taker_person_id (first occurrence wins) before counting, same as
scripts/explore_manhattan_demographics.py did to inspect the raw category values.

Category mapping decisions (see explore_manhattan_demographics.py output):
- employment_status, available_vehicles, education: Replica's raw values already match
  Cambridge's category names exactly (employed/not_in_labor_force/unemployed/under_16;
  zero/one/two/three_plus/unknown_num_vehicles; no_school/k_12/high_school/some_college/
  bachelors_degree/advanced_degree) - no remapping needed.
- age: originally copied Cambridge's 7 adult-only ranges (18-24 ... 76-85), leaving
  age_weights summing to ~0.84 for Manhattan (not 1.0) - anyone under 18 or over 85 fell
  outside every bucket. This wasn't just a cosmetic gap: employment_statuses separately
  includes "under_16" at a real, correctly-measured 12.46% of this same population, so the
  two fields structurally contradicted each other (no age bucket a synthetic "under_16"
  person could ever land in). Confirmed empirically: 0/200 profiles generated with
  age_ranges missing children ever got assigned employment_status="under_16" - the LLM
  consistency-correction step (see tools.py) always resolved the contradiction by silently
  discarding the "under_16" employment value instead, erasing an entire real demographic
  from the synthetic population. Added (0,17) and (86,120) buckets so every real Manhattan
  resident falls in exactly one age bucket and age_weights sum to 1.0.
- income: Cambridge's ranges ("Under 25k".."150k+") are bucketed from
  trip_taker_household_income (not individual_income, which is 0 for most non-earners e.g.
  minors/dependents and would badly skew the distribution).
- household_size: Cambridge tops out at 7 ("7" = "7 or more"); Manhattan data has households
  up to 13, so sizes >=7 are folded into the "7" bucket.
"""
import csv
import json

PEOPLE_CSV = "data/people/manhattan_raw/replica-manhattan_people-09_23_26-trips_dataset.csv"
OUT_PATH = "data/geo/replica-manhattan-demographics.json"

AGE_RANGES = [(0, 17), (18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 75), (76, 85), (86, 120)]
GENDERS = ["male", "female"]
EMPLOYMENT_STATUSES = ["employed", "not_in_labor_force", "unemployed", "under_16"]
VEHICLES = ["zero", "one", "two", "three_plus", "unknown_num_vehicles"]
EDUCATION_LEVELS = ["no_school", "k_12", "high_school", "some_college", "bachelors_degree", "advanced_degree"]
INCOME_RANGES = ["Under 25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k+"]
HOUSEHOLD_SIZES = [1, 2, 3, 4, 5, 6, 7]


def bucket_age(age):
    for i, (lo, hi) in enumerate(AGE_RANGES):
        if lo <= age <= hi:
            return i
    return None


def bucket_income(income):
    if income < 25_000:
        return 0
    if income < 50_000:
        return 1
    if income < 75_000:
        return 2
    if income < 100_000:
        return 3
    if income < 150_000:
        return 4
    return 5


def bucket_household_size(size):
    return min(size, 7) - 1


age_counts = [0] * len(AGE_RANGES)
gender_counts = [0] * len(GENDERS)
employment_counts = [0] * len(EMPLOYMENT_STATUSES)
vehicle_counts = [0] * len(VEHICLES)
education_counts = [0] * len(EDUCATION_LEVELS)
income_counts = [0] * len(INCOME_RANGES)
household_counts = [0] * len(HOUSEHOLD_SIZES)

seen_persons = set()
total = 0

with open(PEOPLE_CSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    idx = {name: i for i, name in enumerate(header)}

    for row in reader:
        pid = row[idx["trip_taker_person_id"]]
        if pid in seen_persons:
            continue
        seen_persons.add(pid)
        total += 1

        age_bucket = bucket_age(int(row[idx["trip_taker_age"]]))
        if age_bucket is not None:
            age_counts[age_bucket] += 1
        gender_counts[GENDERS.index(row[idx["trip_taker_sex"]])] += 1
        employment_counts[EMPLOYMENT_STATUSES.index(row[idx["trip_taker_employment_status"]])] += 1
        vehicle_counts[VEHICLES.index(row[idx["trip_taker_available_vehicles"]])] += 1
        education_counts[EDUCATION_LEVELS.index(row[idx["trip_taker_education"]])] += 1
        income_counts[bucket_income(float(row[idx["trip_taker_household_income"]]))] += 1
        household_counts[bucket_household_size(int(row[idx["trip_taker_household_size"]]))] += 1

demographics = {
    "age_ranges": [list(r) for r in AGE_RANGES],
    "age_weights": [round(c / total, 4) for c in age_counts],
    "genders": GENDERS,
    "gender_weights": [round(c / total, 4) for c in gender_counts],
    "employment_statuses": EMPLOYMENT_STATUSES,
    "employment_weights": [round(c / total, 4) for c in employment_counts],
    "vehicles": VEHICLES,
    "vehicle_weights": [round(c / total, 4) for c in vehicle_counts],
    "education_levels": EDUCATION_LEVELS,
    "education_weights": [round(c / total, 4) for c in education_counts],
    "income_ranges": INCOME_RANGES,
    "income_weights": [round(c / total, 4) for c in income_counts],
    "household_size": HOUSEHOLD_SIZES,
    "household_weights": [round(c / total, 4) for c in household_counts],
}

json.dump(demographics, open(OUT_PATH, "w"), indent=2)
print(f"wrote {OUT_PATH} from {total} unique persons")
print(json.dumps(demographics, indent=2))
