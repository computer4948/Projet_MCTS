"""Lance toutes les expériences (exp1..exp5) puis génère les figures."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

steps = [
    [PY, "experiments/exp1_round_robin.py", "--size", "5", "--budget", "200", "--games", "20"],
    [PY, "experiments/exp2_budget.py", "--size", "5", "--games", "16",
     "--budgets", "50", "100", "200", "400"],
    [PY, "experiments/exp3_heavy.py", "--size", "5", "--budget", "200", "--games", "16"],
    [PY, "experiments/exp4_uct_c.py", "--size", "5", "--budget", "200",
     "--games", "20", "--c-values", "0.1", "0.3", "0.5", "0.8", "1.4"],
    [PY, "experiments/exp5_size.py", "--budget", "150", "--games", "8",
     "--sizes", "4", "5", "6", "7"],
    [PY, "experiments/exp6_multiseed.py", "--size", "5", "--budget", "200",
     "--games", "10", "--seeds", "2026", "4242", "9001"],
    [PY, "experiments/exp7_levels.py", "--size", "5", "--budget", "200", "--games", "16"],
    [PY, "experiments/exp8_hyperparams.py", "--size", "5", "--budget", "200", "--games", "16"],
    [PY, "experiments/exp9_6x6.py", "--budget", "150", "--games", "10"],
    [PY, "experiments/exp10_ppatcs.py", "--size", "5", "--budget", "200", "--games", "16"],
    [PY, "experiments/exp11_extended.py", "--size", "5", "--budget", "200", "--games", "16"],
    [PY, "experiments/exp12_features.py", "--size", "5", "--budget", "200", "--games", "16", "--level", "1"],
    [PY, "experiments/exp13_nrpa2p.py", "--size", "5", "--budget", "200", "--games", "16"],
    [PY, "experiments/make_domineering_figure.py"],
    [PY, "experiments/make_plots.py"],
]

t0 = time.perf_counter()
for s in steps:
    print(">>>", " ".join(s), flush=True)
    rc = subprocess.call(s, cwd=ROOT)
    if rc != 0:
        print("step failed, abort")
        sys.exit(rc)
print(f"All done in {time.perf_counter() - t0:.1f}s")
