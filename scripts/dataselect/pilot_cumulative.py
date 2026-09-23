"""
Pilot: does weakspot-guided selection help once retraining keeps the old data?

Same task, model, gap and diagnosis as ``baselines.py`` (revised setup), but each
arm is retrained two ways from the same warm-started initial model:

    new     fine-tune on the n_select new points only (the protocol used so far)
    cum     fine-tune on the initial training set plus the n_select new points

Arms: random (5 draws, averaged), stratified, k-center and per-point loss at
alpha 0.2 / 1, and weakspot at alpha {0.2, 0.5, 1} x sigma {0.1, 0.5, 1}.
Pilot seeds 2000-2019 are disjoint from the fresh evaluation seeds 1000-1049.

    python -m scripts.dataselect.pilot_cumulative --workers 12
"""
from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.dataselect import baselines as B
from scripts.dataselect import pipeline as P
from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import extract_weakspot

OUT = Path("data/experiment_results/data_selective_training/pilot_cumulative.csv")
SEEDS = list(range(2000, 2020))
GRID = dict(n_train=[100, 400], radius=[0.2], n_select=[100])
WS_ALPHAS, WS_SIGMAS = [0.2, 0.5, 1.0], [0.1, 0.5, 1.0]
MODES = ["new", "cum"]


def run_one(n_train, radius, n_select, seed):
    F = B.FIXED
    rng = np.random.RandomState(seed)
    nb, noise = F["n_bumps"], F["noise_std"]
    centre = np.asarray(F["center"])

    X_all = P.sample_inputs(F["n_pool_total"], rng)
    y_all = P.label(X_all, rng, n_bumps=nb, noise_std=noise)
    X_keep, y_keep, _, _ = P.induce_weakspot(X_all, y_all, centre, radius)
    tr = rng.choice(len(X_keep), size=min(n_train, len(X_keep)), replace=False)
    X_tr, y_tr = X_keep[tr], y_keep[tr]

    X_eval = P.sample_inputs(F["n_eval"], rng)
    y_eval = P.true_function(X_eval, n_bumps=nb)
    X_cand = P.sample_inputs(F["n_candidate"], rng)
    y_cand = P.label(X_cand, rng, n_bumps=nb, noise_std=noise)

    model0 = build_model(AVAILABLE_MODELS[F["model_name"]], complexity=F["complexity"],
                         iterations=F["iters_initial"], warm_start=True,
                         early_stopping=True)
    model0.fit(X_tr, y_tr)
    _, _, m0 = P.evaluate(model0, X_eval, y_eval)

    err_cand = np.abs(y_cand - model0.predict(X_cand))
    xx, yy, grid_flat = create_grid(resolution=F["grid_res"])
    surf = P.normalize_surface(
        DETECTION_METHODS[F["detector"]](X_cand, err_cand, grid_flat))
    ext = extract_weakspot(surf, xx, yy, threshold_quantile=F["extract_q"])
    c_hat = (np.asarray(ext["center"]) if ext is not None
             else grid_flat[int(np.argmax(surf))])

    base = dict(n_train=n_train, radius=radius, n_select=n_select, seed=seed,
                init_mae=m0["MAE"], detect_dist=float(np.hypot(*(c_hat - centre))))
    rows = []

    def add(arm, alpha, sigma, draw, idx):
        for mode in MODES:
            m = copy.deepcopy(model0)
            m.named_steps["model"].max_iter = F["iters_retrain"]
            if mode == "new":
                m.fit(X_cand[idx], y_cand[idx])
            else:
                m.fit(np.vstack([X_tr, X_cand[idx]]), np.concatenate([y_tr, y_cand[idx]]))
            _, err, met = P.evaluate(m, X_eval, y_eval)
            ein, eout = P.region_error(X_eval, err, centre, radius)
            rows.append({**base, "mode": mode, "arm": arm, "alpha": alpha,
                         "sigma": sigma, "draw": draw, "mae": met["MAE"],
                         "err_in": ein, "err_out": eout})

    n_pool = len(X_cand)
    for d in range(B.N_RANDOM):
        r = np.random.RandomState(seed * 100 + d)
        add("random", 0.0, np.nan, d, r.choice(n_pool, size=n_select, replace=False))
    add("stratified", 0.0, np.nan, 0,
        B.pick_stratified(X_cand, n_select, np.random.RandomState(seed * 100 + 50)))
    for arm_i, arm in enumerate(["kcenter", "loss"]):
        for alpha in [0.2, 1.0]:
            r = np.random.RandomState(seed * 100 + 60 + 10 * arm_i + int(alpha * 5))
            k = int(round(alpha * n_select))
            g = B.pick_kcenter(X_cand, X_tr, k) if arm == "kcenter" else B.pick_loss(err_cand, k)
            add(arm, alpha, np.nan, 0, B.with_rehearsal(g, n_select, n_pool, r))
    for i, (alpha, sigma) in enumerate(itertools.product(WS_ALPHAS, WS_SIGMAS)):
        r = np.random.RandomState(seed * 100 + 80 + i)
        k = int(round(alpha * n_select))
        g = B.pick_weakspot(X_cand, c_hat, sigma, k, r)
        add("weakspot", alpha, sigma, 0, B.with_rehearsal(g, n_select, n_pool, r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = list(itertools.product(GRID["n_train"], GRID["radius"], GRID["n_select"], SEEDS))
    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=a.workers)(delayed(run_one)(*j) for j in jobs)
    df = pd.DataFrame([r for o in out for r in o])
    df.to_csv(OUT, index=False)
    print(f"{len(jobs)} jobs in {(time.time() - t0) / 60:.1f} min -> {OUT}")

    rnd = (df[df.arm == "random"].groupby(["n_train", "mode", "seed"])
           [["mae", "err_in", "err_out"]].mean().add_prefix("r_"))
    d = df[df.arm != "random"].join(rnd, on=["n_train", "mode", "seed"])
    d["adv"] = d.r_mae - d.mae                       # positive = arm beats random
    d["adv_in"] = d.r_err_in - d.err_in
    d["adv_out"] = d.r_err_out - d.err_out
    g = d.groupby(["n_train", "mode", "arm", "alpha", "sigma"], dropna=False)
    s = g.agg(adv=("adv", "mean"), adv_in=("adv_in", "mean"), adv_out=("adv_out", "mean"),
              win=("adv", lambda x: (x > 0).mean()), mae=("mae", "mean"))
    s["rel%"] = 100 * s.adv / g.r_mae.mean()
    pd.set_option("display.width", 200)
    print(df.groupby(["n_train", "mode"]).init_mae.mean().round(3))
    print(df[df.arm == "random"].groupby(["n_train", "mode"]).mae.mean().round(3))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
