"""
Iterated weakspot curation (Paper B, Section 4.2 onwards): the single-pass step of the
budget protocol (``scripts.dataselect.budget_v2``, Paper A Sections 4.3-4.4) repeated
over T rounds.

Per seed: fixed dataset (initial set I of 500, reserve R of 4000), a weak region (the
10% nearest a random anchor) in one of four conditions, clean or 5%-outlier labels, a
noiseless/unthinned test set. Each arm runs its own loop: round t detects on the
current model's errors over the remaining reserve, selects n points, and retrains the
current model warm-started for a fixed number of epochs on I plus every point selected
so far (accumulated regime).

Arms (stage ``static``, Section 4.2):
    random       uniform selection, N_RANDOM independent loops per seed
    ws_tuned     Paper A tuned step: kNN landscape, alpha = 1, top 30%
    ws_default   Paper A a-priori step: alpha = 0.2, top 15%

Per round and arm the detection on the model entering the round is diagnosed
(Section 3.3): cross-fitted severity and one-sided Mann-Whitney p-value, detected
centre (argmax of the landscape), its offset from the anchor and drift from the
previous round, and localisation precision against its chance level.

    python -m scripts.iterative.iter_budget --stage smoke --workers 7
    python -m scripts.iterative.iter_budget --stage static --workers 19
"""
from __future__ import annotations

import argparse
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
from scipy.stats import mannwhitneyu

from scripts.weakspot.models import build_model
from scripts.dataselect import budget_v2 as B
from scripts.dataselect.budget import _openml, landscape
from scripts.dataselect.baselines import with_rehearsal

RES = Path("data/experiment_results/iterative_weakspot_curation")

FIXED = dict(n_data=4500, n_round=200, T=8, iters_initial=400, iters_retrain=200,
             k_diag=10, q_diag=0.3, n_random=3, complexity=0.5)
ARMS = {"ws_tuned": dict(alpha=1.0, q=0.3), "ws_default": dict(alpha=0.2, q=0.15)}
DATASETS = ["synth2d", "houses", "medical_charges", "diamonds", "sulfur",
            "brazilian_houses", "nyc_taxi"]
REGIONS = ["sparse_init", "sparse_all", "hard", "hard_sparse"]
NOISES = ["clean", "outlier"]
SEEDS = list(range(8000, 8020))
COLS = ["dataset", "region", "noise", "seed", "arm", "traj", "round", "alpha_t",
        "mae", "err_in", "err_out", "n_in_region", "severity", "p_value", "prec",
        "chance", "offset", "drift", "c0", "c1", "error"]


def _idle():
    """Lowest scheduling priority, so that sweeps do not slow the machine."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        pass


def _closure(fn, name):
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


def diagnose(U, e, in_reg, anchor, prev_c, rs):
    """Detection on one model's reserve errors (Section 3.3)."""
    k, q = FIXED["k_diag"], FIXED["q_diag"]
    land = landscape(U, e, k)
    W = land >= np.quantile(land, 1 - q)
    c = U[int(np.argmax(land))]
    half = rs.permutation(len(U)) % 2 == 0
    A, Bh = np.flatnonzero(half), np.flatnonzero(~half)
    land_A = landscape(U[A], e[A], k)
    d, nb = cKDTree(U[A]).query(U[Bh], k=k)
    w = 1.0 / np.maximum(d, 1e-12)
    land_B = (w * e[A][nb]).sum(1) / w.sum(1)
    WB = land_B >= np.quantile(land_A, 1 - q)
    inn, out = e[Bh][WB], e[Bh][~WB]
    diag = dict(severity=float(inn.mean() / out.mean()),
                p_value=float(mannwhitneyu(inn, out, alternative="greater").pvalue),
                prec=float(in_reg[W].mean()), chance=float(in_reg.mean()),
                offset=float(np.linalg.norm(c - anchor)),
                drift=float(np.linalg.norm(c - prev_c)) if prev_c is not None else np.nan,
                c0=float(c[0]), c1=float(c[1]))
    return diag, land, c


def run_one(name, region, noise, seed, arms=("static",)):
    f = FIXED
    B.FIXED.update(n_data=f["n_data"])
    X, y, I, R, X_ev, y_ev, region_fn, to_U, _ = B.make(name, region, noise, seed)
    anchor = _closure(region_fn, "a")
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    U_R = to_U(X_R)
    in_ev, in_R = region_fn(X_ev), region_fn(X_R)
    nR, n = len(R), f["n_round"]

    m0 = build_model("mlp", complexity=f["complexity"], iterations=f["iters_initial"],
                     warm_start=True, early_stopping=True)
    m0.fit(X_I, y_I)

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()),
                    err_out=float(e[~in_ev].mean()))

    def continue_training(m, sel):
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        m.fit(np.vstack([X_I, X_R[sel]]), np.concatenate([y_I, y_R[sel]]))

    import copy
    base = dict(dataset=name, region=region, noise=noise, seed=seed)
    s0 = score(m0)
    rows = []

    def loop(arm, traj, alpha, q, salt):
        rs = np.random.RandomState(seed * 1000 + salt)
        m = copy.deepcopy(m0)
        avail = np.ones(nR, bool)
        chosen = []
        prev_c = None
        rows.append({**base, "arm": arm, "traj": traj, "round": 0, **s0})
        for t in range(1, f["T"] + 1):
            av = np.flatnonzero(avail)
            e = np.abs(y_R[av] - m.predict(X_R[av]))
            diag, land, prev_c = diagnose(U_R[av], e, in_R[av], anchor, prev_c, rs)
            kk = int(round(alpha * n))
            if kk > 0:
                members = np.flatnonzero(land >= np.quantile(land, 1 - q))
                g = rs.choice(members, min(kk, len(members)), replace=False)
                pick = with_rehearsal(g, n, len(av), rs)
            else:
                pick = rs.choice(len(av), n, replace=False)
            sel = av[pick]
            avail[sel] = False
            chosen.extend(sel.tolist())
            continue_training(m, np.asarray(chosen))
            rows.append({**base, "arm": arm, "traj": traj, "round": t, "alpha_t": alpha,
                         **score(m), "n_in_region": int(in_R[sel].sum()), **diag})

    for r in range(f["n_random"]):
        loop("random", r, 0.0, 0.0, 10 + r)
    for j, (arm, cfg) in enumerate(ARMS.items()):
        loop(arm, 0, cfg["alpha"], cfg["q"], 50 + j)
    return rows


def _safe(args, fixed):
    _idle()
    FIXED.update(fixed)
    try:
        return run_one(*args)
    except Exception as e:
        return [dict(zip(["dataset", "region", "noise", "seed"], args), arm="ERROR",
                     error=repr(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "static"], required=True)
    ap.add_argument("--workers", type=int, default=19)
    a = ap.parse_args()
    _idle()
    out = RES / f"iter_{a.stage}.csv"
    for name in DATASETS:
        if name != "synth2d":
            _openml(name)
    jobs = list(itertools.product(DATASETS, REGIONS, NOISES, SEEDS))
    keys = ["dataset", "region", "noise", "seed"]
    if a.stage == "smoke":
        jobs = [(d, "sparse_init", "clean", SEEDS[0]) for d in DATASETS]
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
        res = Parallel(n_jobs=a.workers)(delayed(_safe)(j, dict(FIXED))
                                         for j in jobs[s:s + chunk])
        df = pd.DataFrame([r for o in res for r in o]).reindex(columns=COLS)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        el = time.time() - t0
        m = min(s + chunk, len(jobs))
        print(f"{m}/{len(jobs)}  {el/60:.1f} min  ETA {(el/m)*(len(jobs)-m)/60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
