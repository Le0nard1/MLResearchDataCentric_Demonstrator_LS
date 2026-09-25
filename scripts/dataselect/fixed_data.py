"""
Weakspot-guided resampling of a fixed dataset (paper Section 4.5, restricted data).

No candidate pool, no new labels, nothing discarded. One fixed labelled dataset D
is split once into a training part (80%) and a validation part (20%, the diagnosis
sample, as any practitioner would hold out). The initial model is trained on the
training part; the weakspot is located on the validation errors; training then
*continues* on the same training part for a fixed number of iterations under a
per-example weight w_i (mean 1), i.e. equal data and equal compute for every arm:

    uniform   w = 1                                   (keep training on all data)
    weakspot  w = (1-a) + a K(x)/mean K,  K Gaussian around the detected centre (ours)
    gated     weakspot if the weakspot is significant (permutation test, p < 0.05),
              otherwise uniform
    loss      w = (1-a) + a r/mean r, r the example's own training residual
    jtt       Just Train Twice: the top-q residual set weighted lam, the rest 1
    density   w = (1-a) + a d/mean d, d the distance to the 5th nearest training
              neighbour (coverage: up-weights sparsely sampled regions)
    smoothed_loss  loss weighting on kNN-smoothed residuals (k = 10): regional
              averaging without a single centre (ablation of the weakspot arm)
    random_centre  the weakspot kernel (same alpha, sigma) around a random training
              point instead of the detected centre, averaged over N_RAND draws:
              the control for whether the detected location carries information
    oracle    the weakspot kernel around the true region centre (synthetic only)

Every row also records the effective sample size of its weights,
ESS = (sum w)^2 / (N sum w^2) in (0, 1], the strength of the intervention.

Tasks
    synthetic   the Gaussian-bump target on [0,1]^2; the region R (centre (0.25,0.75),
                radius 0.2) is sampled at a relative density rho (0 = hole,
                0.15 = sparse, 1 = none); label noise gauss / outlier / hetero
    california  California housing, inputs (longitude, latitude) scaled to [0,1]^2,
                target median house value / 1e5; D is a random subset of n points,
                the test set the remaining districts; no induced weakspot

Protocol: every arm's hyperparameters are chosen on pilot seeds (3000-3019) in the
reference condition of each task (``--stage pilot``), frozen in ``fixed_data_hp.json``
and then reported on fresh seeds 4000-4049 (``--stage main``).

    python -m scripts.dataselect.fixed_data --stage pilot --workers 12
    python -m scripts.dataselect.fixed_data --stage main  --workers 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor

from scripts.weakspot.models import build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import extract_weakspot
from scripts.dataselect import pipeline as P

RES = Path("data/experiment_results/data_selective_training")
HP_FILE = RES / "fixed_data_hp.json"
CAL_RAW = Path.home() / "scikit_learn_data" / "cal_housing_raw.npy"


def _cal_raw():
    """Raw California housing table (longitude, latitude, ..., median house value).

    Uses the local cache if present. Otherwise the needed columns are rebuilt from the
    OpenML copy of the same data (id 44138, the version of the Grinsztajn et al.
    benchmark used in Section 4.3), which holds the same rows in the same order with the
    target stored as log(value + 1); the result is identical to the cached table.
    """
    if CAL_RAW.exists():
        return np.load(CAL_RAW)
    from scripts.dataselect.budget import _openml
    X, y = _openml("houses")                  # columns: ..., latitude, longitude
    a = np.zeros((len(y), 9))
    a[:, 0], a[:, 1] = X[:, 7], X[:, 6]
    a[:, 8] = np.round(np.expm1(y))
    return a

FIXED = dict(n_data=1000, val_frac=0.2, n_bumps=5, noise_std=0.05,
             centre=(0.25, 0.75), radius=0.2, hetero_centre=(0.75, 0.25),
             hetero_std=0.5, ripple_amp=3.0, ripple_freq=3.0, diag="all", outlier_frac=0.05, outlier_std=2.0,
             iters_initial=400, iters_cont=200, complexity=0.5,
             grid_res=35, extract_q=0.85, support_dist=0.05, detector="kNN Performance Mapping",
             n_perm=199, gate_level=0.05, n_eval=4000, map_res=60)

PILOT_SEEDS = list(range(3000, 3020))
MAIN_SEEDS = list(range(4000, 4050))

# Reference condition per task on which the pilot tunes every arm.
REFERENCE = {"synthetic": dict(region="hard", noise="gauss"),
             "california": dict(region="real", noise="real")}
CONDITIONS = {
    "synthetic": [dict(region=r, noise=n) for r in ("hard", "hole", "none")
                  for n in ("gauss", "outlier", "hetero")],
    "california": [dict(region="real", noise="real")],
}

# Pilot grids (every arm tuned with a comparable budget of settings).
HP_GRID = {
    "weakspot": [dict(alpha=a, sigma=s) for a in (0.2, 0.5, 0.8, 1.0)
                 for s in (0.05, 0.1, 0.2)],
    "loss": [dict(alpha=a) for a in (0.2, 0.5, 0.8, 1.0)],
    "jtt": [dict(q=q, lam=l) for q in (0.1, 0.2) for l in (2, 5, 10, 20)],
    "density": [dict(alpha=a) for a in (0.2, 0.5, 0.8, 1.0)],
    "smoothed_loss": [dict(alpha=a) for a in (0.2, 0.5, 0.8, 1.0)],
}
# Controls that reuse the weakspot arm's frozen (alpha, sigma) and are not tuned.
CONTROLS = {"synthetic": ["random_centre", "oracle"], "california": ["random_centre"]}
N_RAND = 5
K_SMOOTH = 10


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
def _sample_region_density(n, rng, centre, radius, rho):
    """Uniform on [0,1]^2 except inside the region, kept with probability rho."""
    out = []
    while sum(len(o) for o in out) < n:
        X = rng.uniform(0, 1, size=(2 * n, 2))
        inside = np.linalg.norm(X - centre, axis=1) <= radius
        keep = ~inside | (rng.uniform(size=len(X)) < rho)
        out.append(X[keep])
    return np.vstack(out)[:n]


def _noise(X, rng, kind):
    f = FIXED
    e = rng.normal(0, f["noise_std"], size=len(X))
    if kind == "outlier":
        m = rng.uniform(size=len(X)) < f["outlier_frac"]
        e[m] += rng.normal(0, f["outlier_std"], size=m.sum())
    elif kind == "hetero":
        m = np.linalg.norm(X - np.asarray(f["hetero_centre"]), axis=1) <= f["radius"]
        e[m] = rng.normal(0, f["hetero_std"], size=m.sum())
    return e


def target(X, region):
    """Bump target; ``hard`` adds a local ripple (envelope radius/2) inside R."""
    y = P.true_function(X, FIXED["n_bumps"])
    if region == "hard":
        c = np.asarray(FIXED["centre"])
        d = X - c
        env = np.exp(-(d ** 2).sum(1) / (2 * (FIXED["radius"] / 2) ** 2))
        k = 2 * np.pi * FIXED["ripple_freq"]
        y = y + FIXED["ripple_amp"] * np.sin(k * d[:, 0]) * np.cos(k * d[:, 1]) * env
    return y


def make_synthetic(region, noise, seed):
    """region: 'hard' (local ripple, uniform density), 'hole' (R empty), 'none'."""
    rng = np.random.RandomState(seed)
    c = np.asarray(FIXED["centre"])
    rho = 0.0 if region == "hole" else 1.0
    X = _sample_region_density(FIXED["n_data"], rng, c, FIXED["radius"], rho)
    y = target(X, region) + _noise(X, rng, noise)
    X_ev = rng.uniform(0, 1, size=(FIXED["n_eval"], 2))
    y_ev = target(X_ev, region)
    return X, y, X_ev, y_ev, rng


_CAL = None


def make_california(seed):
    global _CAL
    if _CAL is None:
        a = _cal_raw()
        X = a[:, :2].copy()
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        _CAL = (X, a[:, 8] / 1e5)
    X_all, y_all = _CAL
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_all))
    d, t = perm[: FIXED["n_data"]], perm[FIXED["n_data"]:][: FIXED["n_eval"]]
    return X_all[d], y_all[d], X_all[t], y_all[t], rng


# ─────────────────────────────────────────────────────────────
# Diagnosis and significance gate
# ─────────────────────────────────────────────────────────────
def diagnose(X_val, err_val, rng, support=None):
    """Detected centre and extent on the validation errors, plus the gate p-value.

    Gate: the peak of the kNN error map against the peaks obtained after permuting
    the errors over the validation locations (no spatial structure under H0).
    """
    xx, yy, grid = create_grid(resolution=FIXED["grid_res"])
    raw = DETECTION_METHODS[FIXED["detector"]](X_val, err_val, grid)
    # Non-square input domain (California): grid cells farther than support_dist from
    # every data point lie outside the domain; they get the map's minimum.
    far = (cKDTree(support).query(grid)[0] > FIXED["support_dist"]
           if support is not None else np.zeros(len(grid), bool))
    raw = np.where(far, raw[~far].min(), raw)
    surf = P.normalize_surface(raw)
    ext = extract_weakspot(surf, xx, yy, threshold_quantile=FIXED["extract_q"])
    c_hat = (np.asarray(ext["center"]) if ext is not None
             else grid[int(np.argmax(surf))])
    stat = float(raw.max())
    knn = KNeighborsRegressor(n_neighbors=min(10, len(X_val) - 1), weights="distance")
    null = np.empty(FIXED["n_perm"])
    for b in range(FIXED["n_perm"]):
        null[b] = knn.fit(X_val, rng.permutation(err_val)).predict(grid[~far]).max()
    p = (1 + int((null >= stat).sum())) / (1 + FIXED["n_perm"])
    return c_hat, ext, p


def effect_ratio(X_d, err_d, ext):
    """Effect-size gate: mean diagnosis error inside the detected 2-sigma ellipse over
    the mean outside. The threshold 2 is the weakspot criterion of pilot_setup.py."""
    from scripts.weakspot.extraction import ellipse_mask
    if ext is None:
        return np.nan
    inside = ellipse_mask(X_d[:, 0], X_d[:, 1], ext)
    if inside.sum() < 3 or (~inside).sum() < 3:
        return np.nan
    return float(err_d[inside].mean() / err_d[~inside].mean())


def _ratio_one(task, cond, seed, fixed):
    """Recompute the initial model and diagnosis of run_one; return p and ratio."""
    FIXED.update(fixed)
    if task == "synthetic":
        X, y, *_ = make_synthetic(cond["region"], cond["noise"], seed)
    else:
        X, y, *_ = make_california(seed)
    n_val = int(round(FIXED["val_frac"] * len(X)))
    m0 = build_model("mlp", complexity=FIXED["complexity"],
                     iterations=FIXED["iters_initial"], warm_start=True,
                     early_stopping=True)
    m0.fit(X[n_val:], y[n_val:])
    err = np.abs(y - m0.predict(X))
    c_hat, ext, p = diagnose(X, err, np.random.RandomState(seed + 7),
                             support=X if task == "california" else None)
    return dict(task=task, region=cond["region"], noise=cond["noise"], seed=seed,
                p_check=p, ratio=effect_ratio(X, err, ext))


def ratio_stage(workers):
    from joblib import Parallel, delayed
    jobs = [(t, c, s) for t, cs in CONDITIONS.items() for c in cs for s in MAIN_SEEDS]
    out = Parallel(n_jobs=workers)(delayed(_ratio_one)(t, c, s, dict(FIXED))
                                   for t, c, s in jobs)
    pd.DataFrame(out).to_csv(RES / "fixed_data_ratio.csv", index=False)


# ─────────────────────────────────────────────────────────────
# Weights (mean 1)
# ─────────────────────────────────────────────────────────────
def _mix(score, alpha):
    s = np.asarray(score, float) + 1e-12
    w = (1 - alpha) + alpha * s / s.mean()
    return w / w.mean()


def ess(w):
    """Effective sample size as a fraction of N (1 = uniform)."""
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def smoothed(X, r, k=K_SMOOTH):
    """Mean residual over each point's k nearest neighbours (itself included)."""
    idx = cKDTree(X).query(X, k=min(k, len(X)))[1]
    return r[idx].mean(1)


def weights(arm, hp, X_tr, res_tr, c_hat):
    """``c_hat`` is the kernel centre: detected, random or true, per arm."""
    if arm == "uniform":
        return np.ones(len(X_tr))
    if arm in ("weakspot", "random_centre", "oracle"):
        return _mix(P.gaussian_weights(X_tr, c_hat, hp["sigma"]), hp["alpha"])
    if arm == "smoothed_loss":
        return _mix(smoothed(X_tr, res_tr), hp["alpha"])
    if arm == "loss":
        return _mix(res_tr, hp["alpha"])
    if arm == "jtt":
        w = np.ones(len(X_tr))
        k = max(1, int(round(hp["q"] * len(X_tr))))
        w[np.argsort(res_tr)[-k:]] = hp["lam"]
        return w / w.mean()
    if arm == "density":
        d = cKDTree(X_tr).query(X_tr, k=6)[0][:, -1]
        return _mix(d, hp["alpha"])
    raise ValueError(arm)


# ─────────────────────────────────────────────────────────────
# One (task, condition, seed)
# ─────────────────────────────────────────────────────────────
def run_one(task, cond, seed, arm_hps, keep_maps=False):
    if task == "synthetic":
        X, y, X_ev, y_ev, rng = make_synthetic(cond["region"], cond["noise"], seed)
    else:
        X, y, X_ev, y_ev, rng = make_california(seed)
    n_val = int(round(FIXED["val_frac"] * len(X)))
    X_tr, y_tr, X_val, y_val = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    m0 = build_model("mlp", complexity=FIXED["complexity"],
                     iterations=FIXED["iters_initial"], warm_start=True,
                     early_stopping=True)
    m0.fit(X_tr, y_tr)
    res_tr = np.abs(y_tr - m0.predict(X_tr))
    # Diagnosis on every labelled example (training residuals + held-out errors):
    # the setup pilot showed a held-out 20% alone too sparse to localise the region.
    X_d = X if FIXED["diag"] == "all" else X_val
    err_d = np.abs((y if FIXED["diag"] == "all" else y_val) - m0.predict(X_d))
    c_hat, ext, p_gate = diagnose(X_d, err_d, np.random.RandomState(seed + 7),
                                  support=X if task == "california" else None)

    c = np.asarray(FIXED["centre"])
    # California has no ground-truth weakspot: "inside" is the disc around the
    # detected centre.
    in_R = (np.linalg.norm(X_ev - (c if task == "synthetic" else c_hat), axis=1)
            <= FIXED["radius"])
    grid_map = None
    if keep_maps and task == "synthetic":
        lin = (np.arange(FIXED["map_res"]) + 0.5) / FIXED["map_res"]
        gx, gy = np.meshgrid(lin, lin)
        grid_map = np.column_stack([gx.ravel(), gy.ravel()])
        f_map = target(grid_map, cond["region"])

    def score(model):
        e = np.abs(y_ev - model.predict(X_ev))
        out = dict(mae=float(e.mean()))
        if in_R is not None:
            out.update(err_in=float(e[in_R].mean()) if in_R.any() else np.nan,
                       err_out=float(e[~in_R].mean()))
        return out, e

    s0, e0 = score(m0)
    base = dict(task=task, region=cond["region"], noise=cond["noise"], seed=seed,
                init_mae=s0["mae"], init_in=s0.get("err_in", np.nan),
                init_out=s0.get("err_out", np.nan), p_gate=p_gate,
                cx=c_hat[0], cy=c_hat[1],
                detect_dist=(float(np.hypot(*(c_hat - c))) if task == "synthetic"
                             else np.nan))
    rows, maps = [], {}
    if grid_map is not None:
        maps["initial"] = np.abs(f_map - m0.predict(grid_map))
    if task == "california":
        maps["initial_ev"] = e0

    def cont(w):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]      # fixed compute: no early or convergence stop
        mlp.set_params(max_iter=FIXED["iters_cont"], early_stopping=False,
                       n_iter_no_change=FIXED["iters_cont"] + 1)
        mlp.best_loss_ = np.inf
        m.fit(X_tr, y_tr, model__sample_weight=w)
        return m

    rng_c = np.random.RandomState(seed + 11)
    rand_centres = X_tr[rng_c.choice(len(X_tr), N_RAND, replace=False)]
    for arm, hp in arm_hps:
        centres = {"random_centre": rand_centres, "oracle": [c]}.get(arm, [c_hat])
        fits = [(cont(w), w) for w in (weights(arm, hp, X_tr, res_tr, cc)
                                       for cc in centres)]
        scored = [score(m) for m, _ in fits]
        s = {k: float(np.mean([sc[0][k] for sc in scored])) for k in scored[0][0]}
        e = np.mean([sc[1] for sc in scored], axis=0)
        tag = arm + ("" if not hp else "|" + ",".join(f"{k}={v}" for k, v in hp.items()))
        rows.append({**base, "arm": arm, "hp": tag, **s,
                     "ess": float(np.mean([ess(w) for _, w in fits]))})
        if grid_map is not None:
            maps[arm] = np.mean([np.abs(f_map - m.predict(grid_map)) for m, _ in fits], 0)
        if task == "california":
            maps[arm + "_ev"] = e
    if task == "california" and keep_maps:
        maps["X_ev"] = X_ev
    return rows, maps


def _safe(task, cond, seed, arm_hps, keep_maps, fixed):
    FIXED.update(fixed)
    try:
        return run_one(task, cond, seed, arm_hps, keep_maps)
    except Exception as e:
        return [dict(task=task, seed=seed, arm="ERROR", hp=str(e), **cond)], {}


# ─────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────
def _setup_one(n_data, iters, amp, freq, seed):
    FIXED.update(n_data=n_data, iters_initial=iters, ripple_amp=amp, ripple_freq=freq)
    out = []
    for region in ("hard", "none"):
        X, y, X_ev, y_ev, _ = make_synthetic(region, "gauss", seed)
        n_val = int(round(FIXED["val_frac"] * len(X)))
        m0 = build_model("mlp", complexity=FIXED["complexity"], iterations=iters,
                         warm_start=True, early_stopping=True)
        m0.fit(X[n_val:], y[n_val:])
        e = np.abs(y_ev - m0.predict(X_ev))
        inR = np.linalg.norm(X_ev - np.asarray(FIXED["centre"]), axis=1) <= FIXED["radius"]
        Xd, yd = (X, y) if FIXED.get("diag") == "all" else (X[:n_val], y[:n_val])
        c_hat, _, p = diagnose(Xd, np.abs(yd - m0.predict(Xd)),
                               np.random.RandomState(seed + 7))
        out.append(dict(n_data=n_data, iters=iters, amp=amp, freq=freq, seed=seed,
                        region=region, init_mae=e.mean(),
                        const_mae=np.abs(y_ev - y_ev.mean()).mean(),
                        ratio=e[inR].mean() / e[~inR].mean(), p_gate=p,
                        dist=float(np.hypot(*(c_hat - np.asarray(FIXED["centre"]))))))
    return out


def _setup_one_f(j, fixed):
    FIXED.update(fixed)
    return _setup_one(*j)


def setup(workers):
    """Choose n_data, initial iterations and the ripple from the diagnosis alone.

    Rule, fixed before running (as pilot_setup.py): among settings with
    init_mae <= 0.5 const_mae, in/out error ratio >= 2 and detection distance <= 0.1
    with the ripple, and distance >= 0.2 without it, take the smallest dataset, then
    the weakest ripple (amplitude, then frequency), then the most initial iterations.
    """
    from joblib import Parallel, delayed
    FIXED['diag'] = 'all'
    jobs = list(itertools.product([500, 1000], [200, 400], [1.0, 2.0, 3.0], [3.0, 5.0],
                                  PILOT_SEEDS))
    out = Parallel(n_jobs=workers)(delayed(_setup_one_f)(j, dict(FIXED)) for j in jobs)
    df = pd.DataFrame([r for o in out for r in o])
    df.to_csv(RES / "fixed_data_setup.csv", index=False)
    g = df.groupby(["n_data", "iters", "amp", "freq", "region"])[
        ["init_mae", "const_mae", "ratio", "dist", "p_gate"]].mean().unstack("region")
    g["ok"] = ((g[("init_mae", "hard")] <= 0.5 * g[("const_mae", "hard")])
               & (g[("ratio", "hard")] >= 2) & (g[("dist", "hard")] <= 0.1)
               & (g[("dist", "none")] >= 0.2))
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())


def pilot(workers, tasks, arms=None):
    from joblib import Parallel, delayed
    arms = arms or list(HP_GRID)
    arm_hps = [("uniform", {})] + [(a, hp) for a in arms for hp in HP_GRID[a]]
    jobs = [(t, REFERENCE[t], s) for t in tasks for s in PILOT_SEEDS]
    out = Parallel(n_jobs=workers)(delayed(_safe)(t, c, s, arm_hps, False, dict(FIXED))
                                   for t, c, s in jobs)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    f = RES / "fixed_data_pilot.csv"
    if f.exists():                    # keep other tasks' and other arms' pilot rows
        old = pd.read_csv(f)
        old = old[~(old.task.isin(tasks) & old.arm.isin(arms + ["uniform"]))]
        df = pd.concat([old, df])
    df.to_csv(f, index=False)
    chosen = json.loads(HP_FILE.read_text()) if HP_FILE.exists() else {}
    for task, g in df[(df.arm != "ERROR") & df.task.isin(tasks)].groupby("task"):
        m = g.groupby(["arm", "hp"])["mae"].mean()
        chosen.setdefault(task, {})
        for arm in arms:
            best = m.loc[arm].idxmin()
            chosen[task][arm] = next(hp for hp in HP_GRID[arm] if arm + "|" + ",".join(
                f"{k}={v}" for k, v in hp.items()) == best)
        print(task, "\n", m.round(4).to_string(), "\n", chosen[task])
    HP_FILE.write_text(json.dumps(chosen, indent=2))


def frontier(workers, tasks):
    """Every arm over its whole pilot grid on the reported seeds, reference condition:
    the in-weakspot / outside trade-off each signal traces (descriptive, no selection)."""
    from joblib import Parallel, delayed
    arm_hps = ([("uniform", {})] + [(a, hp) for a, g in HP_GRID.items() for hp in g]
               + [("random_centre", hp) for hp in HP_GRID["weakspot"]])
    jobs = [(t, REFERENCE[t], s) for t in tasks for s in MAIN_SEEDS]
    out = Parallel(n_jobs=workers)(delayed(_safe)(t, c, s, arm_hps, False, dict(FIXED))
                                   for t, c, s in jobs)
    pd.DataFrame([r for rr, _ in out for r in rr]).to_csv(
        RES / "fixed_data_frontier.csv", index=False)


def main_stage(workers, tasks):
    from joblib import Parallel, delayed
    chosen = json.loads(HP_FILE.read_text())
    rows_all, maps_all = [], {}
    for task, conds in CONDITIONS.items():
        if task not in tasks:
            continue
        arm_hps = ([("uniform", {})] + [(a, chosen[task][a]) for a in HP_GRID]
                   + [(a, chosen[task]["weakspot"]) for a in CONTROLS[task]])
        jobs = [(c, s) for c in conds for s in MAIN_SEEDS]
        t0 = time.time()
        out = Parallel(n_jobs=workers)(
            delayed(_safe)(task, c, s, arm_hps,
                           c == REFERENCE[task] or task == "california", dict(FIXED))
            for c, s in jobs)
        print(f"{task}: {len(jobs)} jobs, {(time.time()-t0)/60:.1f} min", flush=True)
        for (c, s), (rows, maps) in zip(jobs, out):
            rows_all += rows
            for k, v in maps.items():
                maps_all.setdefault(f"{task}__{k}", []).append(v)
    df = pd.DataFrame(rows_all)
    maps_all = {k: np.asarray(v) for k, v in maps_all.items()}
    if (RES / "fixed_data.csv").exists():             # keep the other tasks' results
        old = pd.read_csv(RES / "fixed_data.csv")
        df = pd.concat([old[~old.task.isin(tasks)], df])
        old_maps = dict(np.load(RES / "fixed_data_maps.npz"))
        maps_all = {**{k: v for k, v in old_maps.items()
                       if k.split("__")[0] not in tasks}, **maps_all}
    df.to_csv(RES / "fixed_data.csv", index=False)
    np.savez_compressed(RES / "fixed_data_maps.npz", **maps_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["setup", "pilot", "main", "smoke", "ratio", "frontier"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tasks", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--arms", nargs="+", default=None, help="pilot: arms to tune")
    a = ap.parse_args()
    if a.stage == "smoke":
        t0 = time.time()
        for task in REFERENCE:
            rows, _ = run_one(task, REFERENCE[task], 1, [("uniform", {}),
                              ("weakspot", dict(alpha=0.5, sigma=0.1)),
                              ("jtt", dict(q=0.2, lam=5))])
            print(pd.DataFrame(rows)[["task", "arm", "init_mae", "mae", "p_gate",
                                      "detect_dist"]])
        print(f"{time.time()-t0:.1f}s")
    elif a.stage == "frontier":
        frontier(a.workers, a.tasks)
    elif a.stage == "ratio":
        ratio_stage(a.workers)
    elif a.stage == "setup":
        setup(a.workers)
    elif a.stage == "pilot":
        pilot(a.workers, a.tasks, a.arms)
    else:
        main_stage(a.workers, a.tasks)


if __name__ == "__main__":
    main()
