"""
Stream through the Manhattan People export (one row per trip, not per person - only ~2GB
free RAM available, so no pandas.read_csv of the whole 2.7GB file) and, deduplicating by
trip_taker_person_id, print the raw category values Replica actually uses for this dataset.
This is step 1: see the real values before deciding how to bucket them into the same
schema as data/geo/replica-cambrdige-demographics.json (age_ranges, employment_statuses,
vehicles, education_levels, income_ranges, household_size).
"""
import csv
from collections import Counter

PEOPLE_CSV = "data/people/manhattan_raw/replica-manhattan_people-09_23_26-trips_dataset.csv"

FIELDS = [
    "trip_taker_sex",
    "trip_taker_employment_status",
    "trip_taker_available_vehicles",
    "trip_taker_education",
    "trip_taker_household_size",
]

seen_persons = set()
counters = {f: Counter() for f in FIELDS}
age_counter = Counter()
indiv_income_sample = []
hh_income_sample = []

with open(PEOPLE_CSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    idx = {name: i for i, name in enumerate(header)}
    pid_i = idx["trip_taker_person_id"]

    for n, row in enumerate(reader, 1):
        pid = row[pid_i]
        if pid in seen_persons:
            continue
        seen_persons.add(pid)

        for f in FIELDS:
            counters[f][row[idx[f]]] += 1

        age = row[idx["trip_taker_age"]]
        age_counter[age] += 1

        if len(indiv_income_sample) < 20:
            indiv_income_sample.append(row[idx["trip_taker_individual_income"]])
            hh_income_sample.append(row[idx["trip_taker_household_income"]])

        if n % 500_000 == 0:
            print(f"...{n} rows scanned, {len(seen_persons)} unique persons so far")

print(f"\ntotal rows scanned: {n}")
print(f"unique persons: {len(seen_persons)}")

for f in FIELDS:
    print(f"\n--- {f} ---")
    for val, cnt in counters[f].most_common():
        print(f"  {val!r}: {cnt}")

print("\n--- trip_taker_age (sample of distinct raw values) ---")
print(sorted(age_counter.keys(), key=lambda x: (len(x), x))[:30])

print("\n--- trip_taker_individual_income (sample) ---")
print(indiv_income_sample)
print("\n--- trip_taker_household_income (sample) ---")
print(hh_income_sample)
