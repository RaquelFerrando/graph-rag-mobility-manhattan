"""
Compare the run's on-screen accuracy (random weighted sampling, matching the
paper's own methodology for generating a synthetic population) against what
accuracy would be if we always picked the LLM's own top-weighted choice
instead. Uses only the already-saved choice_weights - no re-run needed.
"""
import json
import sys

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "results/manhattan/exp1_test_1000.json"

data = json.load(open(RESULTS_FILE))
n = len(data)
n_sample_acc = sum(1 for r in data if r['true_mode'] == r['predicted_mode'])
n_max_acc = 0
for r in data:
    choices = json.loads(r['choice_weights'])
    top = max(choices, key=lambda c: c['weight'])
    if top['primary_mode'] == r['true_mode']:
        n_max_acc += 1

print(f"n={n}")
print(f"accuracy (sampling, matches on-screen running_acc): {n_sample_acc}/{n} = {n_sample_acc/n:.1%}")
print(f"accuracy (always picking the top-weight choice):    {n_max_acc}/{n} = {n_max_acc/n:.1%}")
