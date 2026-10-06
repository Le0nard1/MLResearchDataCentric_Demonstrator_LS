"""
Pilot: weakspot guidance combined with coverage, in the training-budget protocol of the
baseline comparison (paper Section 4.4, ``budget_v2 --stage main``).

Data, initial model, reserve, budget and retraining are exactly those of ``run_main``,
so the rows pair with budget_v2_main.csv and budget_v2_main_a1.csv (``uniform`` is
recomputed as a check of that pairing). The weakspot region is that of the tuned
setting (kNN performance mapping at the reserve points, budget_v2_tuned.json).

    hyb_ws        half of the budget uniformly from the detected weakspot region, the other
                  half by k-center over the rest of the reserve (train and weakspot picks
                  counted as covered)
    hyb_randreg   the same with a region of equal size around a random reserve point
                  (location control)
    kc_half       half k-center, half random (what half the coverage achieves alone)
    kc_land       k-center whose farthest-point distance is scaled by the error surface,
                  d(x) * (0.1 + e_hat(x)), over the whole budget
    kc_land_perm  the same with the surface of randomly permuted errors (control: is the
                  gain the error landscape, or the reshaping of k-center?)

    python -m scripts.dataselect.pilot_hybrid --seeds 10 --workers 18          # pilot
    python -m scripts.dataselect.pilot_hybrid --seeds 50 --out hybrid_main.csv   # full
    python -m scripts.dataselect.pilot_hybrid --bench --out hybrid_bench.csv     # further
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.weakspot.models import build_model
from scripts.dataselect.baselines import pick_kcenter
from scripts.dataselect.budget import pick_stratified
from scripts.dataselect import budget_v2 as B

OUT = B.RES / "pilot_hybrid.csv"


def kcenter_rest(U_R, U_cov, taken, k):
    """k-center over the reserve points not yet taken, with U_cov already covered."""
    rest = np.setdiff1d(np.arange(len(U_R)), taken)
    return rest[pick_kcenter(U_R[rest], U_cov, k)]


def kcenter_land(U_R, U_I, land, k):
    d = cKDTree(U_I).query(U_R)[0]
    s = 0.1 + land
    idx = []
    for _ in range(k):
        i = int(np.argmax(d * s))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(U_R - U_R[i], axis=1))
        d[i] = -1.0
    return np.asarray(idx)


def run(name, region, noise, seed):
    f = B.FIXED
    t = json.loads(B.TUNED_FILE.read_text())       # read per job: workers are fresh processes
    X, y, I, R, X_ev, y_ev, region_fn, to_U, rng = B.make(name, region, noise, seed)
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    U_I, U_R = to_U(X_I), to_U(X_R)
    in_ev, in_R = region_fn(X_ev), region_fn(X_R)
    m0 = build_model("mlp", complexity=f["complexity"], iterations=f["iters_initial"],
                     warm_start=True, early_stopping=True)
    m0.fit(X_I, y_I)

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()),
                    err_out=float(e[~in_ev].mean()))

    def retrain(idx):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        m.fit(np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]]))
        return score(m)

    err_R = np.abs(y_R - m0.predict(X_R))
    land = B.surface("knn", U_R, err_R)
    s0 = score(m0)
    base = dict(dataset=name, region=region, noise=noise, seed=seed,
                init_mae=s0["mae"], init_in=s0["err_in"], init_out=s0["err_out"])
    rows = []
    n, nR = f["n_select"], len(R)
    h = n // 2

    def rs(salt):
        return np.random.RandomState(seed * 100 + salt)

    def add(arm, idx):
        idx = np.asarray(idx, int)
        assert len(idx) == n and len(np.unique(idx)) == n, arm
        rows.append({**base, "arm": arm, **retrain(idx), "n_in_region": int(in_R[idx].sum())})

    add("uniform", pick_stratified(U_R, n, rs(50)))              # pairing check

    members = np.flatnonzero(land >= np.quantile(land, t["top_q"]))
    ws = rs(130).choice(members, min(h, len(members)), replace=False)
    add("hyb_ws", np.concatenate([ws, kcenter_rest(U_R, np.vstack([U_I, U_R[ws]]), ws,
                                                   n - len(ws))]))
    a = U_R[rs(131).randint(nR)]
    rand_reg = np.argsort(np.linalg.norm(U_R - a, axis=1))[: len(members)]
    rr = rs(132).choice(rand_reg, min(h, len(rand_reg)), replace=False)
    add("hyb_randreg", np.concatenate([rr, kcenter_rest(U_R, np.vstack([U_I, U_R[rr]]), rr,
                                                        n - len(rr))]))
    kc = pick_kcenter(U_R, U_I, h)
    rest = np.setdiff1d(np.arange(nR), kc)
    add("kc_half", np.concatenate([kc, rs(133).choice(rest, n - h, replace=False)]))
    add("kc_land", kcenter_land(U_R, U_I, land, n))
    land_perm = B.surface("knn", U_R, rs(134).permutation(err_R))
    add("kc_land_perm", kcenter_land(U_R, U_I, land_perm, n))
    return rows


def _safe(args):
    try:
        return run(*args)
    except Exception as e:
        return [dict(zip(["dataset", "region", "noise", "seed"], args), arm="ERROR",
                     error=repr(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--bench", action="store_true",
                    help="the four further datasets of Table 1 (seeds 7000-7009)")
    ap.add_argument("--out", default=OUT.name)
    a = ap.parse_args()
    out_file = B.RES / a.out
    if a.bench:
        jobs = list(itertools.product(B.BENCH_DATASETS, B.REGIONS, B.NOISES, B.BENCH_SEEDS))
    else:
        jobs = list(itertools.product(B.DATASETS, B.REGIONS, B.NOISES,
                                      B.MAIN_SEEDS[: a.seeds]))
    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=a.workers, verbose=5)(delayed(_safe)(j) for j in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(out_file, index=False)
    print(f"{len(jobs)} jobs, {time.time() - t0:.0f}s -> {out_file}")


if __name__ == "__main__":
    main()
