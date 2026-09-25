"""
Detector pilot for budgeted weakspot selection: which identification method of the
ensemble study (DETECTION_METHODS, 17 detectors) selects the best data?

Protocol of budget_v2 (fixed dataset, reserve, budget 200, retraining on the initial
set plus the chosen points). Every detector is cross-fitted in two folds over the
reserve (fitted on one half's errors, evaluated at the other half's points), so that no
detector scores a point with that point's own error; kNN-LOO (the default of
budget_v2) is kept as reference. Selection: region selection at the tuning pilot's
setting, alpha = 0.5, the top 15% of the surface within each fold, the rest uniform
rehearsal. Detectors run as in the ensemble study (same code and defaults) on inputs
rescaled to the spread of the unit square.

Per detector: localisation precision (share of its top 15% in the region; the
region's share of the reserve is chance), whole-test and in-region error, runtime.

    python -m scripts.dataselect.detector_pilot --workers 19
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scripts.weakspot.models import build_model
from scripts.weakspot.detection import DETECTION_METHODS
from scripts.dataselect.baselines import with_rehearsal
from scripts.dataselect.budget import landscape
from scripts.dataselect import budget_v2 as B

OUT = B.RES / "detector_pilot.csv"
SEEDS = list(range(6200, 6205))
REGIONS = ["hard", "sparse_all"]
ALPHA, TOP_Q = 0.5, 0.85


def cross_fit(det, U, err, fold):
    s = np.full(len(U), np.nan)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        v = np.asarray(DETECTION_METHODS[det](U[tr], err[tr], U[te]), float)
        s[te] = np.nan_to_num(v, nan=np.nanmin(v) if np.isfinite(v).any() else 0.0)
    return s


def top_by_fold(s, fold):
    """Members of the top (1 - TOP_Q) of s within each fold."""
    m = np.zeros(len(s), bool)
    for f in (0, 1):
        i = np.flatnonzero(fold == f)
        m[i] = s[i] >= np.quantile(s[i], TOP_Q)
    return m


def run_one(name, region, noise, seed):
    f = B.FIXED
    X, y, I, R, X_ev, y_ev, region_fn, to_U, rng = B.make(name, region, noise, seed)
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    U_R = to_U(X_R)
    in_ev, in_R = region_fn(X_ev), region_fn(X_R)
    m0 = build_model("mlp", complexity=f["complexity"], iterations=f["iters_initial"],
                     warm_start=True, early_stopping=True)
    m0.fit(X_I, y_I)

    def retrain(idx):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        m.fit(np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]]))
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()))

    err_R = np.abs(y_R - m0.predict(X_R))
    fold = np.random.RandomState(seed + 23).permutation(len(R)) % 2
    n, nR = f["n_select"], len(R)
    k = int(round(ALPHA * n))
    base = dict(dataset=name, region=region, noise=noise, seed=seed,
                region_share_R=float(in_R.mean()))
    rand = [retrain(np.random.RandomState(seed * 100 + d).choice(nR, n, replace=False))
            for d in range(B.N_RANDOM)]
    rows = [{**base, "detector": "random",
             **{m: float(np.mean([r[m] for r in rand])) for m in rand[0]}}]

    surfaces = {"kNN (LOO, default)": lambda: landscape(U_R, err_R, f["k_diag"])}
    surfaces.update({d: (lambda d=d: cross_fit(d, U_R, err_R, fold))
                     for d in DETECTION_METHODS})
    for j, (det, fn) in enumerate(surfaces.items()):
        t0 = time.time()
        try:
            s = fn()
            secs = time.time() - t0
            mask = (top_by_fold(s, fold) if det != "kNN (LOO, default)"
                    else s >= np.quantile(s, TOP_Q))
            members = np.flatnonzero(mask)
            r = np.random.RandomState(seed * 1000 + j)
            g = r.choice(members, min(k, len(members)), replace=False)
            res = retrain(with_rehearsal(g, n, nR, r))
            rows.append({**base, "detector": det, **res, "secs": secs,
                         "prec": float(in_R[mask].mean())})
        except Exception as e:
            rows.append({**base, "detector": det, "error": repr(e)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=19)
    a = ap.parse_args()
    for name in B.DATASETS:
        if name != "synth2d":
            B._openml(name)
    jobs = list(itertools.product(B.DATASETS, REGIONS, B.NOISES, SEEDS))
    OUT.unlink(missing_ok=True)
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    res = Parallel(n_jobs=a.workers)(delayed(run_one)(*j) for j in jobs)
    df = pd.DataFrame([r for o in res for r in o])
    df.to_csv(OUT, index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
