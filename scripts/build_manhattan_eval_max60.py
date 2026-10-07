"""
Build a fresh 1000-row eval set for Experiment I, filtered to trip_duration_minutes<=60
from the start (matching the paper's own duration schema, see run_experiment1_manhattan.py's
docstring), instead of filtering down from the existing 1000-row eval set (which left only
884 usable rows). Disjoint from the 5000 rows already used to build the reference graph -
reproduces build_manhattan_ref_eval_split.py's exact ref_indices (same seed) to know which
rows to exclude, then samples 1000 fresh rows meeting the duration filter from the rest.
"""
import csv
import random

FULL_CSV = "data/reference/manhattan_raw/replica-manhattan-09_23_26-trips_dataset.csv"
OUT_PATH = "data/eval/replica-manhattan_trips_eval_max60_1000.csv"

TOTAL_ROWS = 3_770_086
REF_SIZE = 5000
EVAL_SIZE = 1000

# Reproduce build_manhattan_ref_eval_split.py's exact draw to get the reference indices
# (already baked into the cached graph) so this new eval set stays disjoint from them.
random.seed(42)
chosen = random.sample(range(TOTAL_ROWS), REF_SIZE + EVAL_SIZE)
ref_indices = set(chosen[:REF_SIZE])
old_eval_indices = set(chosen[REF_SIZE:])  # the original (unfiltered) eval rows

print("Pass 1: scanning for eligible rows (duration<=60, not in reference set)...")
eligible_indices = []
with open(FULL_CSV, encoding="utf-8") as f_in:
    reader = csv.reader(f_in)
    header = next(reader)
    duration_i = header.index("trip_duration_minutes")

    for i, row in enumerate(reader):
        if i in ref_indices:
            continue
        try:
            duration = float(row[duration_i])
        except ValueError:
            continue
        if duration <= 60:
            eligible_indices.append(i)
        if i % 1_000_000 == 0:
            print(f"...scanned {i} rows, {len(eligible_indices)} eligible so far")

print(f"total eligible rows (excluding reference): {len(eligible_indices)}")

random.seed(43)  # different seed from the ref/eval split - independent draw
new_eval_indices = set(random.sample(eligible_indices, EVAL_SIZE))

print("Pass 2: writing the 1000 selected rows...")
with open(FULL_CSV, encoding="utf-8") as f_in, \
     open(OUT_PATH, "w", newline="", encoding="utf-8") as f_out:
    reader = csv.reader(f_in)
    header = next(reader)
    writer = csv.writer(f_out)
    writer.writerow(header)

    written = 0
    for i, row in enumerate(reader):
        if i in new_eval_indices:
            writer.writerow(row)
            written += 1

print(f"done: wrote {written}/{EVAL_SIZE} rows to {OUT_PATH}")
overlap_with_old = len(new_eval_indices & old_eval_indices)
print(f"overlap with the original (unfiltered) eval set: {overlap_with_old} rows (expected: some, harmless)")
