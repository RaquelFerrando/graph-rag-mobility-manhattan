"""
Build data/reference/replica-manhattan_trips.csv (5000 rows) and
data/eval/replica-manhattan_trips_eval.csv (1000 rows, disjoint from the reference set)
from the full raw Manhattan trips export (3.77M rows, 2.7GB).

BehaviorGraph._load_data() (mobility_agent/graph.py) expects the SAME raw Replica trip
column schema our full CSV already has - it does pd.read_csv() then its own .sample(n=...)
internally, exactly like the original replica-cambridge_trips.csv (which is just a
1000-row raw Replica export, not a specially preprocessed file). So no column
transformation is needed here - just a memory-safe, non-overlapping random split, since
loading the full 2.7GB file into pandas isn't safe with ~2GB free RAM, and pointing both
REF_FILE and an eval file at the same full file wouldn't guarantee non-overlapping rows.

Two-pass approach, both memory-safe (streams the file, never loads it whole):
1. Count total data rows (already known: 3,770,086), pick REF_SIZE+EVAL_SIZE random row
   indices with random.sample (no replacement), split into two disjoint sets.
2. Stream through the file once, writing each row to the matching output as its index is hit.
"""
import csv
import random

FULL_CSV = "data/reference/manhattan_raw/replica-manhattan-09_23_26-trips_dataset.csv"
REF_OUT = "data/reference/replica-manhattan_trips.csv"
EVAL_OUT = "data/eval/replica-manhattan_trips_eval.csv"

TOTAL_ROWS = 3_770_086
REF_SIZE = 5000
EVAL_SIZE = 1000

random.seed(42)
chosen = random.sample(range(TOTAL_ROWS), REF_SIZE + EVAL_SIZE)
ref_indices = set(chosen[:REF_SIZE])
eval_indices = set(chosen[REF_SIZE:])

with open(FULL_CSV, encoding="utf-8") as f_in, \
     open(REF_OUT, "w", newline="", encoding="utf-8") as f_ref, \
     open(EVAL_OUT, "w", newline="", encoding="utf-8") as f_eval:

    reader = csv.reader(f_in)
    header = next(reader)

    ref_writer = csv.writer(f_ref)
    eval_writer = csv.writer(f_eval)
    ref_writer.writerow(header)
    eval_writer.writerow(header)

    ref_written = 0
    eval_written = 0
    for i, row in enumerate(reader):
        if i in ref_indices:
            ref_writer.writerow(row)
            ref_written += 1
        elif i in eval_indices:
            eval_writer.writerow(row)
            eval_written += 1

        if i % 1_000_000 == 0:
            print(f"...scanned {i} rows (ref={ref_written}, eval={eval_written})")

print(f"done: ref={ref_written}/{REF_SIZE} written to {REF_OUT}")
print(f"done: eval={eval_written}/{EVAL_SIZE} written to {EVAL_OUT}")
