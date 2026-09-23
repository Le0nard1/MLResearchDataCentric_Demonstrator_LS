"""
Pilot for the revised Section 4 setup: choose the initial model and the gap position
from the validity of the *diagnosis* alone, never from a guided-vs-random outcome.

For each candidate (initial iterations, training-set size, gap centre, radius) and
pilot seeds disjoint from every reported seed, it records
    init_mae / const_mae   the initial model against predicting the mean (trained?)
    ratio                  initial error inside / outside the gap (gap is a weakspot?)
    dist_gap               detected centre to the gap centre, gap induced
    dist_nogap             the same with no gap (does the model fail there anyway?)
Diagnosis as in the baseline experiment: kNN performance mapping on the noisy-label
candidate pool. No retraining takes place.

Rule, fixed before running: among settings with init_mae <= 0.5 const_mae,
ratio >= 2, dist_gap <= 0.1 and dist_nogap >= 0.2, take n_train = 100 if it
qualifies, then the fewest initial iterations.

    python -m scripts.dataselect.pilot_setup --workers 12
"""
from __future__ import annotations

import argparse
import itertools
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.weakspot.models import build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import extract_weakspot
from scripts.dataselect import pipeline as P

OUT = "data/experiment_results/data_selective_training/pilot_setup.csv"
SEEDS = list(range(2000, 2020))
ITERS = [12, 50, 100, 200, 400]
N_TRAIN = [100, 200, 400]
CENTRES = [(0.25, 0.75), (0.75, 0.25)]
RADII = [0.2, 0.25]


def run(iters, n_train, centre, radius, seed):
    rng = np.random.RandomState(seed)
    centre = np.asarray(centre)
    X_all = P.sample_inputs(2000, rng)
    y_all = P.label(X_all, rng, n_bumps=5, noise_std=0.05)
    out = []
    for gap in (True, False):
        r2 = np.random.RandomState(seed)       # same draws with and without the gap
        r2.uniform(size=1)
        X_keep, y_keep = ((P.induce_weakspot(X_all, y_all, centre, radius)[:2])
                          if gap else (X_all, y_all))
        tr = r2.choice(len(X_keep), size=n_train, replace=False)
        m = build_model("mlp", complexity=0.5, iterations=iters, warm_start=True,
                        early_stopping=True)
        m.fit(X_keep[tr], y_keep[tr])
        X_eval = P.sample_inputs(1000, r2)
        y_eval = P.true_function(X_eval, n_bumps=5)
        err = np.abs(y_eval - m.predict(X_eval))
        e_in, e_out = P.region_error(X_eval, err, centre, radius)
        X_c = P.sample_inputs(2000, r2)
        y_c = P.label(X_c, r2, n_bumps=5, noise_std=0.05)
        xx, yy, grid = create_grid(resolution=35)
        surf = P.normalize_surface(DETECTION_METHODS["kNN Performance Mapping"](
            X_c, np.abs(y_c - m.predict(X_c)), grid))
        ext = extract_weakspot(surf, xx, yy, threshold_quantile=0.85)
        c = np.asarray(ext["center"]) if ext is not None else grid[int(np.argmax(surf))]
        out.append(dict(iters=iters, n_train=n_train, cx=centre[0], cy=centre[1],
                        radius=radius, seed=seed, gap=gap, init_mae=err.mean(),
                        const_mae=np.abs(y_eval - y_eval.mean()).mean(),
                        ratio=e_in / e_out, dist=float(np.hypot(*(c - centre)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = list(itertools.product(ITERS, N_TRAIN, CENTRES, RADII, SEEDS))
    rows = Parallel(n_jobs=a.workers)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in rows for r in rr])
    df.to_csv(OUT, index=False)
    k = ["iters", "n_train", "cx", "cy", "radius"]
    g = df[df.gap].groupby(k).agg(init_mae=("init_mae", "mean"),
                                  const_mae=("const_mae", "mean"),
                                  ratio=("ratio", "mean"), dist_gap=("dist", "mean"))
    g["dist_nogap"] = df[~df.gap].groupby(k)["dist"].mean()
    g["trained"] = g.init_mae / g.const_mae
    g["ok"] = ((g.trained <= 0.5) & (g.ratio >= 2) & (g.dist_gap <= 0.1)
               & (g.dist_nogap >= 0.2))
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
