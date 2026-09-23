"""
Baseline comparison for weakspot-guided selection (paper Section 4.3).

Holds everything of the Section 4.3/4.4 setup fixed (task, MLP, warm-started
retraining on the selected points only, budget ``n_select``) and swaps only the
*targeting signal* that chooses the guided share of the budget:

    random      uniform draw, repeated ``N_RANDOM`` times per seed and averaged
    stratified  one uniform draw per cell of a regular grid (space-filling coverage)
    kcenter     k-center greedy core-set (Sener & Savarese, 2018): farthest point
                from the initial training set and the points already chosen
    density     sample proportional to the distance to the nearest training point
                (soft, stochastic coverage of under-sampled regions)
    loss        highest per-point loss on the pool (per-example error signal)
    weakspot    Gaussian kernel around the detected weakspot centre (ours)

The four targeted arms run at guidance fractions ``ALPHAS``; the remaining
(1-alpha) of the budget is uniform rehearsal, exactly as in Eq. (1). Coverage
arms need no error signal; loss and weakspot read the initial model's error on
the *candidate pool itself* with its noisy labels, so no noiseless or extra
sample enters the diagnosis.

Conditions: initial training-set size x induced gap radius (0 = no induced gap,
the weakspot is whatever the model fits worst) x budget. Seeds are fresh (never
used when the operating point was chosen).

    python -m scripts.dataselect.baselines --workers 12        # full run
    python -m scripts.dataselect.baselines --smoke             # timing check
    python -m scripts.dataselect.baselines --retrain cum --sigma 0.5         --out data/experiment_results/data_selective_training/baselines_v2_cum.csv

``--retrain new`` (default) fine-tunes on the selected points only; ``--retrain cum``
fine-tunes on the initial training set plus the selected points.
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
from scipy.spatial import cKDTree

from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import extract_weakspot
from scripts.dataselect import pipeline as P

OUT = Path("data/experiment_results/data_selective_training/baselines_v2.csv")

# Fixed at the revised Section 4 values (chosen by pilot_setup.py).
FIXED = dict(model_name="MLP Neural Network", complexity=0.5, n_bumps=5,
             noise_std=0.05, n_pool_total=2000, n_candidate=2000, n_eval=1000,
             center=(0.25, 0.75), iters_initial=200, iters_retrain=400, sel_sigma=0.1,
             grid_res=35, extract_q=0.85, detector="kNN Performance Mapping")

GRID = dict(n_train=[100, 200, 400, 800], radius=[0.0, 0.2], n_select=[50, 100, 200])
SEEDS = list(range(1000, 1050))          # fresh: disjoint from all earlier seeds
ALPHAS = [0.2, 1.0]
TARGETED = ["kcenter", "density", "loss", "weakspot"]
N_RANDOM = 5
COLS = ["n_train", "radius", "n_select", "seed", "init_mae", "detect_dist", "arm",
        "alpha", "draw", "mae", "err_in", "err_out", "n_in_gap", "error"]
STRATA = 10                              # 10 x 10 cells for stratified coverage


# ─────────────────────────────────────────────────────────────
# Targeting signals: each returns ``k`` pool indices
# ─────────────────────────────────────────────────────────────
def pick_kcenter(X_cand, X_tr, k):
    """k-center greedy: repeatedly take the candidate farthest from train ∪ chosen."""
    d = cKDTree(X_tr).query(X_cand)[0]
    idx = []
    for _ in range(k):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(X_cand - X_cand[i], axis=1))
        d[i] = -1.0
    return np.array(idx, dtype=int)


def pick_density(X_cand, X_tr, k, rng):
    """Sample proportional to the distance to the nearest training point."""
    w = cKDTree(X_tr).query(X_cand)[0] + 1e-12
    return rng.choice(len(X_cand), size=k, replace=False, p=w / w.sum())


def pick_loss(err_cand, k):
    """Top-k per-point loss (noisy labels)."""
    return np.argsort(err_cand)[-k:][::-1].astype(int)


def pick_weakspot(X_cand, centre, sigma, k, rng):
    w = P.gaussian_weights(X_cand, centre, sigma) + 1e-12
    return rng.choice(len(X_cand), size=k, replace=False, p=w / w.sum())


def pick_stratified(X_cand, n, rng):
    """Space-filling uniform: spread ``n`` picks over STRATA x STRATA cells."""
    cell = (np.clip((X_cand * STRATA).astype(int), 0, STRATA - 1)
            @ np.array([STRATA, 1]))
    order = rng.permutation(STRATA * STRATA)
    per = np.bincount(np.arange(n) % (STRATA * STRATA), minlength=STRATA * STRATA)
    idx = []
    for c, m in zip(order, per[: len(order)]):
        members = np.flatnonzero(cell == c)
        if m and len(members):
            idx.extend(rng.choice(members, size=min(m, len(members)), replace=False))
    short = n - len(idx)                  # empty cells: top up uniformly
    if short > 0:
        rest = np.setdiff1d(np.arange(len(X_cand)), idx)
        idx.extend(rng.choice(rest, size=short, replace=False))
    return np.array(idx, dtype=int)


def with_rehearsal(guided_idx, n, n_pool, rng):
    """Append (n - len(guided)) uniform picks from the rest of the pool."""
    rest = np.setdiff1d(np.arange(n_pool), guided_idx)
    u = rng.choice(rest, size=n - len(guided_idx), replace=False)
    return np.concatenate([guided_idx, u]).astype(int)


# ─────────────────────────────────────────────────────────────
# One (condition, seed)
# ─────────────────────────────────────────────────────────────
def run_one(n_train, radius, n_select, seed):
    rng = np.random.RandomState(seed)
    nb, noise = FIXED["n_bumps"], FIXED["noise_std"]
    centre = np.asarray(FIXED["center"])

    X_all = P.sample_inputs(FIXED["n_pool_total"], rng)
    y_all = P.label(X_all, rng, n_bumps=nb, noise_std=noise)
    if radius > 0:
        X_keep, y_keep, _, _ = P.induce_weakspot(X_all, y_all, centre, radius)
    else:
        X_keep, y_keep = X_all, y_all
    tr = rng.choice(len(X_keep), size=min(n_train, len(X_keep)), replace=False)
    X_tr, y_tr = X_keep[tr], y_keep[tr]

    X_eval = P.sample_inputs(FIXED["n_eval"], rng)
    y_eval = P.true_function(X_eval, n_bumps=nb)
    X_cand = P.sample_inputs(FIXED["n_candidate"], rng)
    y_cand = P.label(X_cand, rng, n_bumps=nb, noise_std=noise)

    key = AVAILABLE_MODELS[FIXED["model_name"]]
    model0 = build_model(key, complexity=FIXED["complexity"],
                         iterations=FIXED["iters_initial"], warm_start=True,
                         early_stopping=True)
    model0.fit(X_tr, y_tr)
    _, err0, m0 = P.evaluate(model0, X_eval, y_eval)

    def retrain_eval(idx):
        m = copy.deepcopy(model0)
        m.named_steps["model"].max_iter = FIXED["iters_retrain"]
        if FIXED.get("retrain", "new") == "cum":
            m.fit(np.vstack([X_tr, X_cand[idx]]), np.concatenate([y_tr, y_cand[idx]]))
        else:
            m.fit(X_cand[idx], y_cand[idx])
        _, err, met = P.evaluate(m, X_eval, y_eval)
        ein, eout = (P.region_error(X_eval, err, centre, radius)
                     if radius > 0 else (np.nan, np.nan))
        return met["MAE"], ein, eout

    # Diagnosis on the labelled pool itself (noisy labels, never trained on).
    err_cand = np.abs(y_cand - model0.predict(X_cand))
    xx, yy, grid_flat = create_grid(resolution=FIXED["grid_res"])
    surf = P.normalize_surface(
        DETECTION_METHODS[FIXED["detector"]](X_cand, err_cand, grid_flat))
    ext = extract_weakspot(surf, xx, yy, threshold_quantile=FIXED["extract_q"])
    c_hat = (np.asarray(ext["center"]) if ext is not None
             else grid_flat[int(np.argmax(surf))])
    dist = float(np.hypot(*(c_hat - centre))) if radius > 0 else np.nan

    base = dict(n_train=n_train, radius=radius, n_select=n_select, seed=seed,
                init_mae=m0["MAE"], detect_dist=dist)
    rows = []

    def add(arm, alpha, draw, idx):
        mae, ein, eout = retrain_eval(idx)
        n_in = (int((np.linalg.norm(X_cand[idx] - centre, axis=1) <= radius).sum())
                if radius > 0 else np.nan)
        rows.append({**base, "arm": arm, "alpha": alpha, "draw": draw, "mae": mae,
                     "err_in": ein, "err_out": eout, "n_in_gap": n_in})

    n_pool = len(X_cand)
    for d in range(N_RANDOM):             # independent random draws (own streams)
        r = np.random.RandomState(seed * 100 + d)
        add("random", 0.0, d, r.choice(n_pool, size=n_select, replace=False))
    add("stratified", 0.0, 0,
        pick_stratified(X_cand, n_select, np.random.RandomState(seed * 100 + 50)))

    for arm_i, arm in enumerate(TARGETED):
        for alpha in ALPHAS:
            r = np.random.RandomState(seed * 100 + 60 + 10 * arm_i + int(alpha * 5))
            k = int(round(alpha * n_select))
            if arm == "kcenter":
                g = pick_kcenter(X_cand, X_tr, k)
            elif arm == "density":
                g = pick_density(X_cand, X_tr, k, r)
            elif arm == "loss":
                g = pick_loss(err_cand, k)
            else:
                g = pick_weakspot(X_cand, c_hat, FIXED["sel_sigma"], k, r)
            add(arm, alpha, 0, with_rehearsal(g, n_select, n_pool, r))
    return rows


def _safe(args, fixed=None):
    if fixed is not None:                 # worker processes do not see main()'s edits
        FIXED.update(fixed)
    try:
        return run_one(*args)
    except Exception as e:                # keep the batch alive, record the failure
        n_train, radius, n_select, seed = args
        return [dict(n_train=n_train, radius=radius, n_select=n_select, seed=seed,
                     arm="ERROR", error=str(e))]


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--retrain", choices=["new", "cum"], default="new")
    ap.add_argument("--sigma", type=float, default=FIXED["sel_sigma"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    OUT = a.out or OUT
    FIXED.update(retrain=a.retrain, sel_sigma=a.sigma)

    jobs = list(itertools.product(GRID["n_train"], GRID["radius"],
                                  GRID["n_select"], SEEDS))
    if a.smoke:
        jobs = [(400, 0.2, 100, 1000), (800, 0.0, 200, 1001)]
    done = set()
    if OUT.exists() and not a.smoke:
        prev = pd.read_csv(OUT)
        done = set(map(tuple, prev[["n_train", "radius", "n_select", "seed"]]
                       .drop_duplicates().itertuples(index=False)))
    jobs = [j for j in jobs if (j[0], float(j[1]), j[2], j[3]) not in done]
    print(f"{len(jobs)} jobs on {a.workers} workers")

    from joblib import Parallel, delayed
    t0 = time.time()
    chunk = max(a.workers * 4, 1)
    for s in range(0, len(jobs), chunk):
        out = Parallel(n_jobs=a.workers)(delayed(_safe)(j, dict(FIXED)) for j in jobs[s:s + chunk])
        rows = [r for o in out for r in o]
        if a.smoke:
            print(pd.DataFrame(rows).groupby(["arm", "alpha"])["mae"].mean())
        else:
            df = pd.DataFrame(rows).reindex(columns=COLS)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        el = time.time() - t0
        n = min(s + chunk, len(jobs))
        print(f"{n}/{len(jobs)}  {el/60:.1f} min  ETA {(el/n)*(len(jobs)-n)/60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
