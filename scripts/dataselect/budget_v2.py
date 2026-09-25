"""
Budgeted curation, second design (paper Section 4.3): pilot of the revised protocol.

Differences to ``budget.py`` (the first design, kept unchanged for reproducibility):

* Region conditions (the region is the q_region-fraction of a large candidate pool
  nearest a random anchor, in standardised inputs):
    sparse_init  the first design: region thinned to rho in the initial set only, so
                 the reserve is enriched in it
    sparse_all   region thinned to rho in the whole fixed dataset (initial set and
                 reserve alike): the data are rare where the model is weak, and stay
                 rare when more are sought
    hard         no thinning; the target carries a local ripple inside the region
                 (under-learnt: data present, function locally harder)
    hard_sparse  both
* Surface-consuming selection (the ensemble study's central finding: detectors that
  use the whole error surface beat those that use its argmax):
    ws_kernel_<det>   first design: rank kernel around the surface maximum
    ws_surface_<det>  sample in proportion to the (min-max scaled) error surface
    ws_region_<det>   uniform among the reserve points in the surface's top 15%
* Detectors, evaluated at the reserve points:
    knn     leave-one-out distance-weighted kNN mean (k = 10), as before
    evtgpr  geometric mean of peaks-over-threshold (EVT) and GPR, the best detector
            of the ensemble study, on coordinates scaled like the unit square; the
            EVT bandwidth grows with sqrt(d / 2) beyond two dimensions (our extension)
* rho_landscape uses the EVT x GPR surface.

Single pass, fixed guidance fraction alpha = 0.2 for every guided arm (tuning of the
mix and of the headroom is a separate experiment). Detection diagnostics per job:
initial inside/outside error ratio and, per detector, the share of the surface's top
15% of reserve points that lie in the region (localisation precision; region share
of the reserve is the chance level).

    python -m scripts.dataselect.budget_v2 --stage pilot --workers 19

Tuning stage (``--stage tune``, its own experiment, seeds 6100-6109): how to apply the
method and when it pays. Region selection (the pilot's winner) with both detectors,
under a grid of guidance fraction alpha x headroom (initial set size, initial epochs)
and, for the kNN region arm, the region size (top quantile of the surface). Every
competitor with a guidance fraction can run over the same alpha grid and headroom
(tune_ours_only=False); by default only the weakspot arms and the random reference
run, so the tuning is a guide to applying our method, and the main comparison must
state that competitors keep their default settings. Conditions: the two pure
mechanisms, hard (under-learnt) and sparse_all (under-represented everywhere).

    python -m scripts.dataselect.budget_v2 --stage tune --workers 19
    python -m scripts.dataselect.budget_v2 --stage tune_pilot   # 3 seeds, a first look

Main stage (``--stage main``, the baseline comparison, fresh seeds 7000-7049): every
competitor at its standard setting (alpha = 0.2) and our method twice, ``ws_default``
(region selection, kNN, alpha = 0.2, top 15%, the a-priori setting of the design
rules) and ``ws_tuned`` (the setting chosen in the tuning experiment, read from
budget_v2_tuned.json), over all four region conditions.

    python -m scripts.dataselect.budget_v2 --stage main --workers 19
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
from sklearn.neighbors import KernelDensity

from scripts.weakspot.models import build_model
from scripts.weakspot.detection import detect_gpr
from scripts.dataselect import pipeline as P
from scripts.dataselect.baselines import pick_kcenter, with_rehearsal
from scripts.dataselect.budget import (_openml, irreducible_loss, landscape,
                                       pick_stratified, rank_kernel, top)

RES = Path("data/experiment_results/data_selective_training")

FIXED = dict(n_data=2000, n_init=500, n_select=200, n_eval=4000, q_region=0.1,
             rho=0.25, noise_std=0.05, n_bumps=5, outlier_frac=0.05, outlier_std=2.0,
             ripple_amp=2.0, ripple_periods=3.0, complexity=0.5, iters_initial=400,
             iters_retrain=200, k_diag=10, q_kernel=0.3, top_q=0.85, alpha=0.2,
             tune_ours_only=True)
DATASETS = ["synth2d", "houses", "medical_charges"]
REGIONS = ["sparse_init", "sparse_all", "hard", "hard_sparse"]
NOISES = ["clean", "outlier"]
PILOT_SEEDS = list(range(6000, 6010))
TUNE_SEEDS = list(range(6100, 6110))
MAIN_SEEDS = list(range(7000, 7050))
BENCH_DATASETS = ["diamonds", "sulfur", "brazilian_houses", "nyc_taxi"]
BENCH_SEEDS = list(range(7000, 7010))
HD_DATASETS = ["cpu_act", "pol", "ailerons", "yprop_4_1", "superconduct"]
HD_REGIONS = ["hard", "sparse_all"]
HD_SEEDS = list(range(7000, 7005))
TUNED_FILE = RES / "budget_v2_tuned.json"      # written by hand from the tuning results
TUNE_REGIONS = ["hard", "sparse_all"]
TUNE_ALPHAS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
TUNE_HEADROOM = [(250, 400), (500, 400), (1000, 400), (500, 50)]   # (n_init, epochs)
TUNE_TOPQ = [0.7, 0.85, 0.95]
TUNE_COLS = ["dataset", "region", "noise", "seed", "n_init", "iters_initial",
             "init_mae", "init_in", "init_out", "region_share_R", "prec_knn",
             "prec_evtgpr", "arm", "alpha", "top_q", "mae", "err_in", "err_out",
             "n_in_region", "error"]
DETECTORS = ["knn", "evtgpr"]
N_RANDOM = 5
COLS = ["dataset", "region", "noise", "seed", "init_mae", "init_in", "init_out",
        "region_share_R", "prec_knn", "prec_evtgpr", "arm", "mae", "err_in",
        "err_out", "n_in_region", "error"]


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
def _pool(name, n, rng):
    if name == "synth2d":
        X = P.sample_inputs(n, rng)
        return X, P.true_function(X, n_bumps=FIXED["n_bumps"]), True
    X_all, y_all = _openml(name)
    idx = rng.choice(len(X_all), min(n, len(X_all)), replace=False)
    return X_all[idx], y_all[idx], False


def make(name, region, noise, seed):
    """Fixed dataset D, split (I, R), test set T, region membership function."""
    f = FIXED
    rng = np.random.RandomState(seed)
    n, n_ev = f["n_data"], f["n_eval"]
    X, y0, synth = _pool(name, 4 * (n + n_ev), rng)
    xm, xs = X.mean(0), X.std(0) + 1e-12
    U = (X - xm) / xs / np.sqrt(12.0)          # unit-square-like spread per axis
    a = U[rng.randint(len(U))]
    dist = np.linalg.norm(U - a, axis=1)
    r_q = np.quantile(dist, f["q_region"])
    inside = dist <= r_q
    to_U = lambda Xq: (Xq - xm) / xs / np.sqrt(12.0)

    # Test set: unthinned, so the region's error can be measured.
    perm = rng.permutation(len(X))
    ev, rest = perm[:n_ev], perm[n_ev:]
    thin_all = region in ("sparse_all", "hard_sparse")
    keep = ~inside[rest] | (rng.rand(len(rest)) < (f["rho"] if thin_all else 1.0))
    d_idx = rest[keep][:n]

    mu, sd = y0[d_idx].mean(), y0[d_idx].std()
    y = (y0 - mu) / sd
    if synth:                                  # labels noisy, evaluation noiseless
        y_lab = y + rng.normal(0, f["noise_std"] / sd, len(y))
    else:
        y_lab = y.copy()
    if region in ("hard", "hard_sparse"):      # local ripple, amplitude in std(y)
        dim = U.shape[1]
        v = np.linalg.qr(rng.normal(size=(dim, 2)))[0][:, :2]
        D = (U - a) @ v
        k = 2 * np.pi * f["ripple_periods"] / (2 * r_q)
        env = np.exp(-((U - a) ** 2).sum(1) / (2 * (r_q / 2) ** 2))
        rip = f["ripple_amp"] * np.sin(k * D[:, 0]) * np.cos(k * D[:, 1]) * env
        y, y_lab = y + rip, y_lab + rip
    X_D, y_D = X[d_idx], y_lab[d_idx]
    if noise == "outlier":
        o = rng.rand(len(y_D)) < f["outlier_frac"]
        y_D = y_D + o * rng.normal(0, f["outlier_std"], len(y_D))

    # Initial set: uniform from D, except in sparse_init (region thinned in I only).
    p = np.ones(len(d_idx))
    if region == "sparse_init":
        p[inside[d_idx]] = f["rho"]
    I = rng.choice(len(d_idx), f["n_init"], replace=False, p=p / p.sum())
    R = np.setdiff1d(np.arange(len(d_idx)), I)
    region_fn = lambda Xq: np.linalg.norm(to_U(Xq) - a, axis=1) <= r_q
    return X_D, y_D, I, R, X[ev], y[ev], region_fn, to_U, rng


# ─────────────────────────────────────────────────────────────
# Detectors at the reserve points
# ─────────────────────────────────────────────────────────────
def _unit(s):
    s = np.asarray(s, float)
    return (s - s.min()) / (np.ptp(s) + 1e-12)


def surface(det, U_R, err_R):
    if det == "knn":
        return _unit(landscape(U_R, err_R, FIXED["k_diag"]))
    thr = np.quantile(err_R, FIXED["top_q"])
    Xe = U_R[err_R >= thr]
    bw = max(0.05, 0.5 / np.sqrt(len(Xe))) * np.sqrt(U_R.shape[1] / 2.0)
    kde = KernelDensity(bandwidth=bw).fit(Xe)
    ld = kde.score_samples(U_R)
    s_evt = _unit(np.exp(ld - ld.max()))
    s_gpr = _unit(detect_gpr(U_R, err_R, U_R))
    return np.sqrt(s_evt * s_gpr)


# ─────────────────────────────────────────────────────────────
# One job
# ─────────────────────────────────────────────────────────────
def run_one(name, region, noise, seed):
    f = FIXED
    X, y, I, R, X_ev, y_ev, region_fn, to_U, rng = make(name, region, noise, seed)
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

    def retrain(idx=None, X_fit=None, y_fit=None):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        if X_fit is None:
            X_fit, y_fit = np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]])
        m.fit(X_fit, y_fit)
        return score(m)

    err_R = np.abs(y_R - m0.predict(X_R))
    surf = {d: surface(d, U_R, err_R) for d in DETECTORS}
    top_mask = {d: s >= np.quantile(s, f["top_q"]) for d, s in surf.items()}
    reducible = err_R - irreducible_loss(X_I, y_I, X_R, y_R, seed)

    s0 = score(m0)
    base = dict(dataset=name, region=region, noise=noise, seed=seed,
                init_mae=s0["mae"], init_in=s0["err_in"], init_out=s0["err_out"],
                region_share_R=float(in_R.mean()),
                **{f"prec_{d}": float(in_R[top_mask[d]].mean()) for d in DETECTORS})
    rows = []
    n, nR = f["n_select"], len(R)
    k = int(round(f["alpha"] * n))

    def add(arm, idx=None, **kw):
        s = retrain(idx, **kw)
        rows.append({**base, "arm": arm, **s,
                     "n_in_region": int(in_R[idx].sum()) if idx is not None else np.nan})

    add("all_data", X_fit=X, y_fit=y)
    rand = [retrain(np.random.RandomState(seed * 100 + d).choice(nR, n, replace=False))
            for d in range(N_RANDOM)]
    rows.append({**base, "arm": "random",
                 **{m: float(np.mean([s[m] for s in rand])) for m in rand[0]}})
    add("stratified", pick_stratified(U_R, n, np.random.RandomState(seed * 100 + 50)))

    def guided(arm, g, salt):
        add(arm, with_rehearsal(g, n, nR, np.random.RandomState(seed * 100 + salt)))

    def sample(w, salt):
        w = np.asarray(w, float) + 1e-12
        return np.random.RandomState(seed * 100 + salt).choice(nR, k, replace=False,
                                                                p=w / w.sum())

    guided("kcenter", pick_kcenter(U_R, U_I, k), 60)
    guided("density", sample(cKDTree(U_I).query(U_R)[0], 61), 62)
    guided("loss", top(err_R, k), 63)
    guided("rho", top(reducible, k), 64)
    guided("rho_landscape", top(np.maximum(reducible, 0) * surf["evtgpr"], k), 65)
    if f.get("main"):                          # baseline comparison: default + tuned
        def region_pick(det, q, kk, salt):
            members = np.flatnonzero(surf[det] >= np.quantile(surf[det], q))
            return np.random.RandomState(seed * 100 + salt).choice(
                members, min(kk, len(members)), replace=False)
        guided("ws_default", region_pick("knn", f["top_q"], k, 90), 91)
        t = f["tuned"]
        kt = max(1, int(round(t["alpha"] * n)))
        guided("ws_tuned", region_pick(t["detector"], t["top_q"], kt, 92), 93)
        return rows
    for j, d in enumerate(DETECTORS):
        s = surf[d]
        c = U_R[int(np.argmax(s))]
        guided(f"ws_kernel_{d}", sample(rank_kernel(U_R, c, f["q_kernel"]), 70 + j), 71 + j)
        guided(f"ws_surface_{d}", sample(s, 74 + j), 75 + j)
        members = np.flatnonzero(top_mask[d])
        g = np.random.RandomState(seed * 100 + 78 + j).choice(
            members, min(k, len(members)), replace=False)
        guided(f"ws_region_{d}", g, 80 + j)
    return rows


JTT_LAMBDA = 2.0          # the frozen JTT up-weight of the fixed-data experiment
MAIN_COLS = ["dataset", "region", "noise", "seed", "init_mae", "init_in", "init_out",
             "region_share_R", "prec_knn", "arm", "mae", "err_in", "err_out",
             "n_in_region", "error"]


def run_main(name, region, noise, seed):
    """Baseline comparison (Section 4.4): every competitor at its standard setting
    (alpha = 0.2), our method at its default and its tuned setting."""
    f = FIXED
    t = f["tuned"]
    X, y, I, R, X_ev, y_ev, region_fn, to_U, rng = make(name, region, noise, seed)
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

    def retrain(idx=None, X_fit=None, y_fit=None, w_new=None):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        if X_fit is None:
            X_fit, y_fit = np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]])
        if w_new is None:
            m.fit(X_fit, y_fit)
        else:                                  # JTT: up-weight the chosen error set
            w = np.concatenate([np.ones(len(X_I)), w_new])
            m.fit(X_fit, y_fit, model__sample_weight=w / w.mean())
        return score(m)

    err_R = np.abs(y_R - m0.predict(X_R))
    land = surface("knn", U_R, err_R)          # approximated error landscape, [0, 1]
    reducible = err_R - irreducible_loss(X_I, y_I, X_R, y_R, seed)
    s0 = score(m0)
    base = dict(dataset=name, region=region, noise=noise, seed=seed,
                init_mae=s0["mae"], init_in=s0["err_in"], init_out=s0["err_out"],
                region_share_R=float(in_R.mean()),
                prec_knn=float(in_R[land >= np.quantile(land, f["top_q"])].mean()))
    rows = []
    n, nR = f["n_select"], len(R)
    k = int(round(f["alpha"] * n))

    def add(arm, idx=None, **kw):
        s = retrain(idx, **kw)
        rows.append({**base, "arm": arm, **s,
                     "n_in_region": int(in_R[idx].sum()) if idx is not None else np.nan})

    def rs(salt):
        return np.random.RandomState(seed * 100 + salt)

    def reh(g, salt):
        return with_rehearsal(g, n, nR, rs(salt))

    def region_members(q):
        return np.flatnonzero(land >= np.quantile(land, q))

    if f.get("bench"):
        # Benchmark extension: the strong baselines and competitors of Table 1 and all
        # four of our methods, same salts as the main comparison.
        add("all_data", X_fit=X, y_fit=y)
        rand = [retrain(rs(d).choice(nR, n, replace=False)) for d in range(N_RANDOM)]
        rows.append({**base, "arm": "random",
                     **{m: float(np.mean([s[m] for s in rand])) for m in rand[0]}})
        add("uniform", pick_stratified(U_R, n, rs(50)))
        add("rho", reh(top(reducible, k), 63))
        mem = region_members(f["top_q"])
        add("ws_default", reh(rs(66).choice(mem, min(k, len(mem)), replace=False), 67))
        kt = max(1, int(round(t["alpha"] * n)))
        mt = region_members(t["top_q"])
        add("ws_tuned", reh(rs(68).choice(mt, min(kt, len(mt)), replace=False), 69))
        k = n                                  # alpha = 1 arms
        add("kcenter_a1", reh(pick_kcenter(U_R, U_I, k), 60))
        add("rho_a1", reh(top(reducible, k), 63))
        add("rho_landscape_a1", reh(top(np.maximum(reducible, 0) * land, k), 64))
        add("rho_filter_a1", reh(mem[top(reducible[mem], min(k, len(mem)))], 65))
        return rows
    if f.get("a1_only"):
        # Fairness check: every competitor at the tuned method's guidance fraction.
        # Runs are deterministic (seeded data, draws and MLP), so these rows pair with
        # budget_v2_main.csv; uniform is recomputed as a check of that pairing.
        k = n
        add("uniform", pick_stratified(U_R, n, rs(50)))
        add("kcenter_a1", reh(pick_kcenter(U_R, U_I, k), 60))
        add("loss_a1", reh(top(err_R, k), 61))
        g = top(err_R, k)
        idx = reh(g, 62)
        add("jtt_a1", idx, w_new=np.where(np.isin(idx, g), JTT_LAMBDA, 1.0))
        add("rho_a1", reh(top(reducible, k), 63))
        add("rho_landscape_a1", reh(top(np.maximum(reducible, 0) * land, k), 64))
        mem = region_members(f["top_q"])
        add("rho_filter_a1", reh(mem[top(reducible[mem], min(k, len(mem)))], 65))
        return rows
    add("all_data", X_fit=X, y_fit=y)
    rand = [retrain(rs(d).choice(nR, n, replace=False)) for d in range(N_RANDOM)]
    rows.append({**base, "arm": "random",
                 **{m: float(np.mean([s[m] for s in rand])) for m in rand[0]}})
    add("uniform", pick_stratified(U_R, n, rs(50)))
    add("kcenter", reh(pick_kcenter(U_R, U_I, k), 60))
    add("loss", reh(top(err_R, k), 61))
    g = top(err_R, k)                          # JTT: error set, up-weighted by lambda
    idx = reh(g, 62)
    add("jtt", idx, w_new=np.where(np.isin(idx, g), JTT_LAMBDA, 1.0))
    add("rho", reh(top(reducible, k), 63))
    add("rho_landscape", reh(top(np.maximum(reducible, 0) * land, k), 64))
    mem = region_members(f["top_q"])           # RHO-LOSS within the detected region
    add("rho_filter", reh(mem[top(reducible[mem], min(k, len(mem)))], 65))
    add("ws_default", reh(rs(66).choice(mem, min(k, len(mem)), replace=False), 67))
    kt = max(1, int(round(t["alpha"] * n)))
    mt = region_members(t["top_q"])
    add("ws_tuned", reh(rs(68).choice(mt, min(kt, len(mt)), replace=False), 69))
    return rows


def _safe_main(args, fixed):
    FIXED.update(fixed)
    try:
        return run_main(*args)
    except Exception as e:
        return [dict(zip(["dataset", "region", "noise", "seed"], args), arm="ERROR",
                     error=repr(e))]


def run_tune(name, region, noise, seed, n_init, iters):
    """All methods over the alpha grid at one headroom level (single pass)."""
    f = FIXED
    f.update(n_init=n_init, iters_initial=iters)
    X, y, I, R, X_ev, y_ev, region_fn, to_U, rng = make(name, region, noise, seed)
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    U_I, U_R = to_U(X_I), to_U(X_R)
    in_ev, in_R = region_fn(X_ev), region_fn(X_R)
    m0 = build_model("mlp", complexity=f["complexity"], iterations=iters,
                     warm_start=True, early_stopping=True)
    m0.fit(X_I, y_I)

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()),
                    err_out=float(e[~in_ev].mean()))

    def retrain(idx=None, X_fit=None, y_fit=None):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        if X_fit is None:
            X_fit, y_fit = np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]])
        m.fit(X_fit, y_fit)
        return score(m)

    err_R = np.abs(y_R - m0.predict(X_R))
    surf = {d: surface(d, U_R, err_R) for d in DETECTORS}
    ours_only = f["tune_ours_only"]
    if not ours_only:
        reducible = err_R - irreducible_loss(X_I, y_I, X_R, y_R, seed)
    s0 = score(m0)
    base = dict(dataset=name, region=region, noise=noise, seed=seed, n_init=n_init,
                iters_initial=iters, init_mae=s0["mae"], init_in=s0["err_in"],
                init_out=s0["err_out"], region_share_R=float(in_R.mean()),
                **{f"prec_{d}": float(in_R[surf[d] >= np.quantile(surf[d], f["top_q"])]
                                      .mean()) for d in DETECTORS})
    rows = []
    n, nR = f["n_select"], len(R)

    def add(arm, alpha=np.nan, top_q=np.nan, idx=None, **kw):
        s = retrain(idx, **kw)
        rows.append({**base, "arm": arm, "alpha": alpha, "top_q": top_q, **s,
                     "n_in_region": int(in_R[idx].sum()) if idx is not None else np.nan})

    if not ours_only:
        add("all_data", X_fit=X, y_fit=y)
    rand = [retrain(np.random.RandomState(seed * 100 + d).choice(nR, n, replace=False))
            for d in range(N_RANDOM)]
    rows.append({**base, "arm": "random", "alpha": 0.0,
                 **{m: float(np.mean([s[m] for s in rand])) for m in rand[0]}})
    if not ours_only:
        add("stratified", 0.0, idx=pick_stratified(
            U_R, n, np.random.RandomState(seed * 100 + 50)))
    dens = cKDTree(U_I).query(U_R)[0] + 1e-12
    for ai, alpha in enumerate(TUNE_ALPHAS):
        k = max(1, int(round(alpha * n)))

        def r(salt):
            return np.random.RandomState(seed * 1000 + 10 * ai + salt)

        def reh(g, salt):
            return with_rehearsal(g, n, nR, r(salt))

        if not ours_only:
            add("kcenter", alpha, idx=reh(pick_kcenter(U_R, U_I, k), 1))
            add("density", alpha, idx=reh(r(2).choice(nR, k, replace=False,
                                                       p=dens / dens.sum()), 3))
            add("loss", alpha, idx=reh(top(err_R, k), 4))
            add("rho", alpha, idx=reh(top(reducible, k), 5))
            add("rho_landscape", alpha,
                idx=reh(top(np.maximum(reducible, 0) * surf["evtgpr"], k), 6))
        for d in DETECTORS:
            for q in (TUNE_TOPQ if d == "knn" else [f["top_q"]]):
                members = np.flatnonzero(surf[d] >= np.quantile(surf[d], q))
                g = r(7).choice(members, min(k, len(members)), replace=False)
                add(f"ws_region_{d}", alpha, q, idx=reh(g, 8))
    return rows


def _safe_tune(args, fixed):
    FIXED.update(fixed)
    try:
        return run_tune(*args)
    except Exception as e:
        return [dict(zip(["dataset", "region", "noise", "seed", "n_init",
                          "iters_initial"], args), arm="ERROR", error=repr(e))]


def _safe(args, fixed=None):
    if fixed is not None:
        FIXED.update(fixed)
    try:
        return run_one(*args)
    except Exception as e:
        return [dict(zip(["dataset", "region", "noise", "seed"], args), arm="ERROR",
                     error=repr(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "smoke", "tune", "tune_pilot", "tune_smoke", "main", "main_a1", "bench", "bench_smoke", "hdpilot",
                             "main_smoke"],
                    required=True)
    ap.add_argument("--workers", type=int, default=19)
    a = ap.parse_args()
    out = RES / f"budget_v2_{a.stage}.csv"
    for name in DATASETS:
        if name != "synth2d":
            _openml(name)
    tune = a.stage.startswith("tune")
    worker, cols, keys = ((_safe_tune, TUNE_COLS, ["dataset", "region", "noise", "seed",
                                                    "n_init", "iters_initial"])
                          if tune else (_safe, COLS, ["dataset", "region", "noise",
                                                      "seed"]))
    if a.stage in ("main", "main_smoke", "main_a1", "bench", "bench_smoke", "hdpilot"):
        worker, cols = _safe_main, MAIN_COLS
    jobs = list(itertools.product(DATASETS, REGIONS, NOISES, PILOT_SEEDS))
    if tune:
        jobs = [(d, r, nz, s, ni, it) for d in DATASETS for r in TUNE_REGIONS
                for nz in NOISES for (ni, it) in TUNE_HEADROOM
                for s in (TUNE_SEEDS[:3] if a.stage == "tune_pilot" else TUNE_SEEDS)]
    if a.stage in ("main", "main_smoke", "main_a1", "bench", "bench_smoke", "hdpilot"):
        import json
        if not TUNED_FILE.exists():
            raise SystemExit(f"{TUNED_FILE} missing: fix the tuned setting from the "
                             "tuning experiment first, e.g. "
                             '{"alpha": 0.35, "top_q": 0.85, "detector": "knn"}')
        FIXED.update(main=True, tuned=json.loads(TUNED_FILE.read_text()),
                     a1_only=(a.stage == "main_a1"),
                     bench=a.stage.startswith("bench") or a.stage == "hdpilot")
        jobs = list(itertools.product(DATASETS, REGIONS, NOISES, MAIN_SEEDS))
        if a.stage.startswith("bench"):
            for name in BENCH_DATASETS:        # download once, before the workers start
                _openml(name)
            jobs = list(itertools.product(BENCH_DATASETS, REGIONS, NOISES, BENCH_SEEDS))
        if a.stage == "hdpilot":             # pilot: high-dimensional datasets
            for name in HD_DATASETS:
                _openml(name)
            jobs = list(itertools.product(HD_DATASETS, HD_REGIONS, NOISES, HD_SEEDS))
        if a.stage == "bench_smoke":
            jobs = [(d, "hard", "outlier", 7000) for d in BENCH_DATASETS]
            out.unlink(missing_ok=True)
        if a.stage == "main_smoke":
            jobs = [(d, "hard", "outlier", 7000) for d in DATASETS]
            out.unlink(missing_ok=True)
    if a.stage == "tune_smoke":
        jobs = [("medical_charges", "hard", "outlier", 6100, 500, 400)]
        out.unlink(missing_ok=True)
    elif a.stage == "smoke":
        jobs = [(d, r, "outlier", 6000) for d in DATASETS for r in ("sparse_all", "hard")]
        out.unlink(missing_ok=True)
    elif out.exists():
        prev = pd.read_csv(out)
        done = set(map(tuple, prev[keys].drop_duplicates().itertuples(index=False)))
        jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    chunk = max(a.workers * 2, 1)
    for s in range(0, len(jobs), chunk):
        res = Parallel(n_jobs=a.workers)(delayed(worker)(j, dict(FIXED))
                                         for j in jobs[s:s + chunk])
        df = pd.DataFrame([r for o in res for r in o]).reindex(columns=cols)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        el = time.time() - t0
        m = min(s + chunk, len(jobs))
        print(f"{m}/{len(jobs)}  {el/60:.1f} min  ETA {(el/m)*(len(jobs)-m)/60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
