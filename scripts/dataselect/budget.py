"""
Budgeted curation from a fixed labelled dataset (paper Section 4.3, new protocol).

One fixed labelled dataset D (n_data points) is all there is. An initial training set
I is drawn from D with an under-represented region: the q_region-fraction of D nearest
a random anchor (standardised inputs, a new anchor per seed) is kept in I at relative
density rho (0 = empty, 1 = not under-represented). The rest of D is the reserve R.
The initial model is trained on I and diagnosed on R (labelled, never trained on); every
method then chooses a budget of n_select reserve points, and the model is retrained,
warm-started, on I plus the chosen points for a fixed number of epochs (equal compute).
Training on all of D is reported as a reference for what the budget costs.

Methods (the guided share alpha * n_select comes from the signal, the rest is uniform
rehearsal from the reserve; random and stratified use the full budget):
    random         uniform draw, N_RANDOM draws averaged
    stratified     k-means on the reserve into n_select clusters, one point per cluster
    kcenter        k-center greedy core-set (Sener & Savarese, 2018)
    density        sample proportional to the distance to the nearest point of I
    loss           highest initial-model error on the reserve
    rho            RHO-LOSS (Mindermann et al., 2022): highest reducible loss, the
                   initial model's error minus a cross-fitted irreducible-loss model's
                   (two folds over R, each IL model trained on I plus the other fold);
                   static selection once per pass
    rho_landscape  reducible loss (clipped at 0) times the approximated error landscape
    weakspot       rank kernel around the detected centre (ours)
    all_data       reference: retrain on all of D

Detection is grid-free in every dataset, so that the 2-D task and the benchmarks use
the same estimator: kNN performance mapping evaluated at the reserve points themselves
(distance-weighted, the point left out); the error landscape is that smoothed error, and
the weakspot centre is its maximum. The selection kernel acts on distance ranks,
K_i = exp(-rank_i / (q_kernel |R|)), as in fixed_data_hd.py.

Datasets
    synth2d          the Gaussian-bump target on [0,1]^2, Gaussian label noise, evaluated
                     against the noiseless target
    houses           California housing, all 8 features (OpenML 44138, Grinsztajn suite)
    medical_charges  OpenML 44146 (3 features, Grinsztajn suite)
Targets are standardised with the statistics of D, so MAE is in units of std(y).
Label conditions: clean, or 5% of D's labels (I and R) shifted by N(0, 2^2).
Metrics on the test set: whole MAE and MAE inside / outside the region (test points
closer to the anchor than the q_region-quantile distance of D), method-independent.

    python -m scripts.dataselect.budget --smoke
    python -m scripts.dataselect.budget --workers 19
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
from sklearn.cluster import KMeans

from scripts.weakspot.models import build_model
from scripts.dataselect import pipeline as P
from scripts.dataselect.baselines import pick_kcenter, with_rehearsal

OUT = Path("data/experiment_results/data_selective_training/budget.csv")

FIXED = dict(n_data=2000, n_init=500, n_select=200, n_eval=4000, q_region=0.1,
             noise_std=0.05, n_bumps=5, outlier_frac=0.05, outlier_std=2.0,
             complexity=0.5, iters_initial=400, iters_retrain=200, k_diag=10,
             q_kernel=0.3)
# Grinsztajn et al. (2022) numerical regression suite (OpenML study 336): data id and
# target; the suite's preprocessed parquet files, cached under ~/scikit_learn_data.
OPENML = {"houses": (44138, "medianhousevalue"),
          "medical_charges": (44146, "AverageTotalPayments")}
SUITE_DIR = Path.home() / "scikit_learn_data" / "grinsztajn"
DATASETS = ["synth2d", "houses", "medical_charges"]
RHOS = [0.0, 0.25, 1.0]
NOISES = ["clean", "outlier"]
SEEDS = list(range(5000, 5050))
ALPHAS = [0.2, 1.0]
TARGETED = ["kcenter", "density", "loss", "rho", "rho_landscape", "weakspot"]
N_RANDOM = 5
COLS = ["dataset", "rho", "noise", "seed", "init_mae", "init_in", "init_out",
        "detect_in_region", "arm", "alpha", "mae", "err_in", "err_out", "n_in_region",
        "error"]

_RAW = {}


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
def _openml(name):
    if name not in _RAW:
        did, tgt = OPENML[name]
        f = SUITE_DIR / f"{name}.pq"
        if not f.exists():
            import urllib.request
            f.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"https://data.openml.org/datasets/{did // 10000:04d}/{did}/dataset_{did}.pq", f)
        d = pd.read_parquet(f)
        _RAW[name] = (d.drop(columns=[tgt]).to_numpy(float), d[tgt].to_numpy(float))
    return _RAW[name]


def make(name, noise, seed):
    """Fixed dataset D (noisy labels) and test set T (clean labels), y standardised on D."""
    rng = np.random.RandomState(seed)
    n, n_ev = FIXED["n_data"], FIXED["n_eval"]
    if name == "synth2d":
        X = P.sample_inputs(n, rng)
        y = P.label(X, rng, n_bumps=FIXED["n_bumps"], noise_std=FIXED["noise_std"])
        X_ev = P.sample_inputs(n_ev, rng)
        y_ev = P.true_function(X_ev, n_bumps=FIXED["n_bumps"])
    else:
        X_all, y_all = _openml(name)
        idx = rng.permutation(len(X_all))[: n + n_ev]
        X, y = X_all[idx[:n]], y_all[idx[:n]]
        X_ev, y_ev = X_all[idx[n:]], y_all[idx[n:]]
    mu, sd = y.mean(), y.std()
    y, y_ev = (y - mu) / sd, (y_ev - mu) / sd
    if noise == "outlier":
        o = rng.rand(n) < FIXED["outlier_frac"]
        y = y + o * rng.normal(0, FIXED["outlier_std"], n)
    return X, y, X_ev, y_ev, rng


def split(X, rho, rng):
    """Initial set I (region thinned to relative density rho) and reserve R."""
    xm, xs = X.mean(0), X.std(0) + 1e-12
    Z = (X - xm) / xs
    anchor = Z[rng.randint(len(Z))]
    d = np.linalg.norm(Z - anchor, axis=1)
    r_q = np.quantile(d, FIXED["q_region"])
    p = np.where(d <= r_q, rho, 1.0)
    I = rng.choice(len(X), FIXED["n_init"], replace=False, p=p / p.sum())
    R = np.setdiff1d(np.arange(len(X)), I)
    region = lambda Xq: np.linalg.norm((Xq - xm) / xs - anchor, axis=1) <= r_q
    return I, R, region, (xm, xs)


# ─────────────────────────────────────────────────────────────
# Diagnosis and signals
# ─────────────────────────────────────────────────────────────
def landscape(Z, err, k):
    """Leave-one-out distance-weighted kNN mean of the errors at every point."""
    dist, idx = cKDTree(Z).query(Z, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    w = 1.0 / np.maximum(dist, 1e-12)
    return (w * err[idx]).sum(1) / w.sum(1)


def rank_kernel(Z, c, q):
    rank = np.argsort(np.argsort(((Z - c) ** 2).sum(1)))
    return np.exp(-rank / (q * len(Z)))


def irreducible_loss(X_I, y_I, X_R, y_R, seed):
    fold = np.random.RandomState(seed + 17).permutation(len(X_R)) % 2
    il = np.empty(len(X_R))
    for f in (0, 1):
        m = build_model("mlp", complexity=FIXED["complexity"],
                        iterations=FIXED["iters_initial"], warm_start=False,
                        early_stopping=True)
        o = fold != f
        m.fit(np.vstack([X_I, X_R[o]]), np.concatenate([y_I, y_R[o]]))
        il[fold == f] = np.abs(y_R[fold == f] - m.predict(X_R[fold == f]))
    return il


def pick_stratified(Z_R, n, rng):
    lab = KMeans(n_clusters=n, n_init=1, random_state=rng.randint(1 << 30)).fit_predict(Z_R)
    return np.array([rng.choice(np.flatnonzero(lab == c)) for c in range(n)], dtype=int)


def top(score, k):
    return np.argsort(score)[-k:][::-1].astype(int)


# ─────────────────────────────────────────────────────────────
# One (dataset, rho, noise, seed)
# ─────────────────────────────────────────────────────────────
def run_one(name, rho, noise, seed):
    X, y, X_ev, y_ev, rng = make(name, noise, seed)
    I, R, region, (xm, xs) = split(X, rho, rng)
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    Z_R = (X_R - xm) / xs
    in_ev = region(X_ev)

    m0 = build_model("mlp", complexity=FIXED["complexity"],
                     iterations=FIXED["iters_initial"], warm_start=True,
                     early_stopping=True)
    m0.fit(X_I, y_I)

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()),
                    err_out=float(e[~in_ev].mean()))

    def retrain(idx, X_fit=None, y_fit=None):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]      # fixed compute: no early or convergence stop
        mlp.set_params(max_iter=FIXED["iters_retrain"], early_stopping=False,
                       n_iter_no_change=FIXED["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        if X_fit is None:
            X_fit, y_fit = np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]])
        m.fit(X_fit, y_fit)
        return score(m)

    err_R = np.abs(y_R - m0.predict(X_R))
    land = landscape(Z_R, err_R, FIXED["k_diag"])
    c_hat = Z_R[int(np.argmax(land))]
    reducible = err_R - irreducible_loss(X_I, y_I, X_R, y_R, seed)
    land01 = (land - land.min()) / (np.ptp(land) + 1e-12)

    s0 = score(m0)
    anchor_hit = bool(region(X_R[int(np.argmax(land))][None])[0])
    base = dict(dataset=name, rho=rho, noise=noise, seed=seed, init_mae=s0["mae"],
                init_in=s0["err_in"], init_out=s0["err_out"], detect_in_region=anchor_hit)
    rows = []

    def add(arm, alpha, idx=None, **kw):
        s = retrain(idx, **kw)
        n_in = int(region(X_R[idx]).sum()) if idx is not None else np.nan
        rows.append({**base, "arm": arm, "alpha": alpha, **s, "n_in_region": n_in})

    n, nR = FIXED["n_select"], len(R)
    add("all_data", np.nan, X_fit=X, y_fit=y)
    rand = []
    for d in range(N_RANDOM):
        r = np.random.RandomState(seed * 100 + d)
        rand.append(retrain(r.choice(nR, n, replace=False)))
    rows.append({**base, "arm": "random", "alpha": 0.0,
                 **{k: float(np.mean([s[k] for s in rand])) for k in rand[0]}})
    add("stratified", 0.0, pick_stratified(Z_R, n, np.random.RandomState(seed * 100 + 50)))

    Z_I = (X_I - xm) / xs
    for i, arm in enumerate(TARGETED):
        for alpha in ALPHAS:
            r = np.random.RandomState(seed * 100 + 60 + 10 * i + int(alpha * 5))
            k = int(round(alpha * n))
            if arm == "kcenter":
                g = pick_kcenter(Z_R, Z_I, k)
            elif arm == "density":
                w = cKDTree(Z_I).query(Z_R)[0] + 1e-12
                g = r.choice(nR, k, replace=False, p=w / w.sum())
            elif arm == "loss":
                g = top(err_R, k)
            elif arm == "rho":
                g = top(reducible, k)
            elif arm == "rho_landscape":
                g = top(np.maximum(reducible, 0.0) * land01, k)
            else:
                w = rank_kernel(Z_R, c_hat, FIXED["q_kernel"]) + 1e-12
                g = r.choice(nR, k, replace=False, p=w / w.sum())
            add(arm, alpha, with_rehearsal(g, n, nR, r))
    return rows


def _safe(args, fixed=None):
    if fixed is not None:
        FIXED.update(fixed)
    try:
        return run_one(*args)
    except Exception as e:
        name, rho, noise, seed = args
        return [dict(dataset=name, rho=rho, noise=noise, seed=seed, arm="ERROR",
                     error=repr(e))]


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    OUT = a.out or OUT

    for name in a.datasets:               # download once, before the workers start
        if name in OPENML:
            _openml(name)
    jobs = list(itertools.product(a.datasets, RHOS, NOISES, SEEDS[: a.seeds]))
    if a.smoke:
        jobs = [(d, 0.25, nz, 5000) for d in a.datasets for nz in NOISES]
    done = set()
    if OUT.exists() and not a.smoke:
        prev = pd.read_csv(OUT)
        done = set(map(tuple, prev[["dataset", "rho", "noise", "seed"]]
                       .drop_duplicates().itertuples(index=False)))
    jobs = [j for j in jobs if (j[0], float(j[1]), j[2], j[3]) not in done]
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    chunk = max(a.workers * 2, 1)
    for s in range(0, len(jobs), chunk):
        out = Parallel(n_jobs=a.workers)(delayed(_safe)(j, dict(FIXED))
                                         for j in jobs[s:s + chunk])
        df = pd.DataFrame([r for o in out for r in o]).reindex(columns=COLS)
        if a.smoke:
            print(df.drop(columns=["seed"], errors="ignore").to_string())
        else:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        el = time.time() - t0
        n = min(s + chunk, len(jobs))
        print(f"{n}/{len(jobs)}  {el/60:.1f} min  ETA {(el/n)*(len(jobs)-n)/60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
