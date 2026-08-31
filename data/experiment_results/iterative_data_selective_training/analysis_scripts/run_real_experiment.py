"""Real-data experiment: 3 encodings x 3 policies x seeds on iris_extended.

Writes one row per (encoding, policy, seed, iteration) to
data/experiment_results/real_data_example/real_runs.csv (resumable), plus the
per-round categorical distributions as JSON strings.
"""
import json
import os
import sys
import time
from pathlib import Path

APP = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application"
sys.path.insert(0, APP)
os.chdir(APP)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.realdata import loop_real as R

OUT = Path("data/experiment_results/real_data_example")
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "real_runs.csv"

SEEDS = [42, 0, 7, 1, 3, 5, 11, 17, 23, 99, 2, 4, 6, 8, 13]
# setting: "raw" = the dataset as-is; "deficit" = virginica withheld from the
# initial training set and thinned 90% from the pool (the papers' scarce-pool
# analogue on real data).
SETTINGS = {"raw": {}, "deficit": dict(deficit_col="species",
                                       deficit_val="virginica",
                                       deficit_frac=0.9)}
GRID = [(st, enc, pol, s) for st in SETTINGS for enc in R.ENCODINGS
        for pol in R.POLICIES for s in SEEDS]

done = set()
if CSV.exists():
    d = pd.read_csv(CSV, usecols=["setting", "encoding", "policy", "seed"]).drop_duplicates()
    done = {(r.setting, r.encoding, r.policy, int(r.seed)) for r in d.itertuples()}
todo = [g for g in GRID if g not in done]
print(f"{len(GRID)} runs total, {len(todo)} to do", flush=True)


def one(st, enc, pol, seed):
    res = R.run_real(dict(encoding=enc, policy=pol, seed=seed, **SETTINGS[st]))
    rows = []
    for i in range(len(res["mae_g"])):
        row = dict(setting=st, encoding=enc, policy=pol, seed=seed, iteration=i,
                   mae_g=res["mae_g"][i], mae_r=res["mae_r"][i],
                   gap=res["mae_g"][i] - res["mae_r"][i])
        if i > 0:
            row.update(sev=res["sev"][i - 1], mix=res["mix"][i - 1],
                       cat_weak=json.dumps(res["cat_weak"][i - 1]),
                       cat_pick=json.dumps(res["cat_pick"][i - 1]))
        rows.append(row)
    return rows


t0 = time.time()
BATCH = 14
for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    results = Parallel(n_jobs=14)(delayed(one)(*g) for g in chunk)
    new = [r for rows in results for r in rows]
    pd.DataFrame(new).to_csv(CSV, mode="a" if CSV.exists() else "w",
                             header=not CSV.exists(), index=False)
    print(f"{min(i + BATCH, len(todo))}/{len(todo)}  "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
print("REAL EXPERIMENT COMPLETE", flush=True)
