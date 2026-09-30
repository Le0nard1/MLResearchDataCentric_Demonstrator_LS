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
    python -m scripts.iterative.iter_budget --stage static --workers 12 \
        --datasets synth2d --n-seeds 50
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
             k_diag=10, q_diag=0.3, n_random=3, complexity=0.5, gamma=0.5)
ARMS = {"ws_tuned": dict(alpha=1.0, q=0.3), "ws_default": dict(alpha=0.2, q=0.15)}
DATASETS = ["synth2d", "houses", "medical_charges", "diamonds", "sulfur",
            "brazilian_houses", "nyc_taxi"]
REGIONS = ["sparse_init", "sparse_all", "hard", "hard_sparse"]
NOISES = ["clean", "outlier"]
SEEDS = list(range(8000, 8020))
COLS = ["dataset", "region", "noise", "seed", "arm", "traj", "round", "alpha_t",
        "mae", "err_in", "err_out", "n_in_region", "severity", "p_value", "prec",
        "chance", "offset", "drift", "c0", "c1", "error"]
PILOT_COLS = (COLS[:5] + ["method", "schedule", "regime"] + COLS[5:-1]
              + ["n_fit", "epochs", "sample_updates", "fit_sec", "select_sec", "error"])
# Pilot (storyline check): fresh seeds, clean labels, two repairable and one
# persistent condition, the synthetic task and two benchmark datasets.
PILOT = dict(datasets=["synth2d", "houses", "sulfur"],
             regions=["hard", "sparse_init", "sparse_all"], noises=["clean"],
             seeds=list(range(9000, 9010)))
# Second pilot: retraining on the new points only (the compute-limited setting), the
# guidance fraction that leaves room for rehearsal inside every batch, and the two
# schedules on alpha = 0.5 and alpha = 1.
PILOT2_ARMS = (
    [("random", "random", "static", "new", 0.0, 0.0, 20 + r) for r in range(2)]
    + [("ws_default", "region", "static", "new", 0.2, 0.15, 81),
       ("ws_half", "region", "static", "new", 0.5, 0.3, 82),
       ("ws_tuned", "region", "static", "new", 1.0, 0.3, 70),
       ("rho_filter", "rho_filter", "static", "new", 1.0, 0.3, 83),
       ("rho_landscape", "rho_landscape", "static", "new", 1.0, 0.3, 84),
       ("ws_half_dynamic", "region", "dynamic", "new", 0.5, 0.3, 85),
       ("ws_half_adaptive", "region", "adaptive", "new", 0.5, 0.3, 86),
       ("ws_dynamic", "region", "dynamic", "new", 1.0, 0.3, 72),
       ("ws_adaptive", "region", "adaptive", "new", 1.0, 0.3, 73)])


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


def alpha_schedule(schedule, alpha, t, S_t, S_1):
    """Guidance fraction of round t (Section 3.4)."""
    if schedule == "static":
        return alpha
    if schedule == "dynamic":                  # fixed decay, front-loads the focus
        return alpha * FIXED["gamma"] ** (t - 1)
    if schedule == "adaptive":                 # scaled by the remaining severity
        return alpha * float(np.clip((S_t - 1.0) / max(S_1 - 1.0, 1e-9), 0.0, 1.0))
    raise ValueError(schedule)


# Pilot arms: (arm, method, schedule, regime, alpha, q, salt). The static stage keeps
# its own arms and salts, so iter_static.csv stays reproducible.
PILOT_ARMS = (
    [("random", "random", "static", "acc", 0.0, 0.0, 10 + r) for r in range(2)]
    + [("ws_default", "region", "static", "acc", 0.2, 0.15, 51),
       ("ws_tuned", "region", "static", "acc", 1.0, 0.3, 50),
       ("rho_filter", "rho_filter", "static", "acc", 1.0, 0.3, 60),
       ("rho_landscape", "rho_landscape", "static", "acc", 1.0, 0.3, 61),
       ("ws_dynamic", "region", "dynamic", "acc", 1.0, 0.3, 62),
       ("ws_adaptive", "region", "adaptive", "acc", 1.0, 0.3, 63)]
    + [("random", "random", "static", "new", 0.0, 0.0, 20 + r) for r in range(2)]
    + [("ws_tuned", "region", "static", "new", 1.0, 0.3, 70),
       ("ws_dynamic", "region", "dynamic", "new", 1.0, 0.3, 72),
       ("ws_adaptive", "region", "adaptive", "new", 1.0, 0.3, 73)])


def run_one(name, region, noise, seed, stage="static"):
    f = FIXED
    pilot = stage.startswith("pilot")
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
    # Irreducible loss for the RHO-LOSS combinations: cross-fitted once on the reserve,
    # independent of the model being trained (Paper A, Section 3).
    il = B.irreducible_loss(X_I, y_I, X_R, y_R, seed) if pilot else None

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()),
                    err_out=float(e[~in_ev].mean()))

    def continue_training(m, X_fit, y_fit):
        """Retrain; returns the compute spent (points, epochs, CPU seconds)."""
        mlp = m.named_steps["model"]
        t0 = time.process_time()
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        if pilot:                              # input scaling frozen at the initial model
            mlp.fit(m.named_steps["scaler"].transform(X_fit), y_fit)
        else:
            m.fit(X_fit, y_fit)
        return len(X_fit), mlp.n_iter_, time.process_time() - t0   # n_iter_ resets per fit

    import copy
    base = dict(dataset=name, region=region, noise=noise, seed=seed)
    s0 = score(m0)
    rows = []

    def loop(arm, traj, alpha, q, salt, method="region", schedule="static",
             regime="acc"):
        rs = np.random.RandomState(seed * 1000 + salt)
        m = copy.deepcopy(m0)
        avail = np.ones(nR, bool)
        chosen = []
        prev_c, S_1 = None, None
        tag = dict(method=method, schedule=schedule, regime=regime) if pilot else {}
        rows.append({**base, "arm": arm, "traj": traj, "round": 0, **tag, **s0})
        for t in range(1, f["T"] + 1):
            ts = time.process_time()
            av = np.flatnonzero(avail)
            e = np.abs(y_R[av] - m.predict(X_R[av]))
            diag, land, prev_c = diagnose(U_R[av], e, in_R[av], anchor, prev_c, rs)
            S_1 = diag["severity"] if S_1 is None else S_1
            a_t = alpha_schedule(schedule, alpha, t, diag["severity"], S_1)
            kk = int(round(a_t * n))
            if method == "random" or kk == 0:
                pick = rs.choice(len(av), n, replace=False)
            else:
                members = np.flatnonzero(land >= np.quantile(land, 1 - q))
                if method == "region":
                    g = rs.choice(members, min(kk, len(members)), replace=False)
                elif method == "rho_filter":       # reducible loss within the region
                    red = e - il[av]
                    g = members[np.argsort(red[members])[::-1][:min(kk, len(members))]]
                elif method == "rho_landscape":    # reducible loss x error landscape
                    g = np.argsort(np.maximum(e - il[av], 0) * land)[::-1][:kk]
                pick = with_rehearsal(g, n, len(av), rs)
            sel = av[pick]
            select_sec = time.process_time() - ts
            avail[sel] = False
            chosen.extend(sel.tolist())
            if regime == "acc":                # initial set plus every selection so far
                c = np.asarray(chosen)
                cost = continue_training(m, np.vstack([X_I, X_R[c]]),
                                         np.concatenate([y_I, y_R[c]]))
            else:                              # this round's selection alone
                cost = continue_training(m, X_R[sel], y_R[sel])
            rows.append({**base, "arm": arm, "traj": traj, "round": t, **tag,
                         "alpha_t": a_t, **score(m),
                         "n_in_region": int(in_R[sel].sum()), **diag,
                         **(dict(n_fit=cost[0], epochs=cost[1],
                                 sample_updates=cost[0] * cost[1], fit_sec=cost[2],
                                 select_sec=select_sec) if pilot else {})})

    if pilot:
        arms = PILOT2_ARMS if stage.startswith("pilot2") else PILOT_ARMS
        for arm, method, schedule, regime, alpha, q, salt in arms:
            traj = salt % 10 if method == "random" else 0
            loop(arm, traj, alpha, q, salt, method, schedule, regime)
        return rows
    for r in range(f["n_random"]):
        loop("random", r, 0.0, 0.0, 10 + r)
    for j, (arm, cfg) in enumerate(ARMS.items()):
        loop(arm, 0, cfg["alpha"], cfg["q"], 50 + j)
    return rows


def _safe(args, fixed, stage="static"):
    _idle()
    FIXED.update(fixed)
    try:
        return run_one(*args, stage=stage)
    except Exception as e:
        return [dict(zip(["dataset", "region", "noise", "seed"], args), arm="ERROR",
                     error=repr(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "static", "pilot", "pilot_smoke", "pilot2", "pilot2_smoke"], required=True)
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--n-seeds", type=int, default=len(SEEDS),
                    help="seeds 8000, 8001, ...; existing rows are kept and skipped")
    a = ap.parse_args()
    _idle()
    out = RES / f"iter_{a.stage}.csv"
    for name in a.datasets:
        if name != "synth2d":
            _openml(name)
    seeds = list(range(SEEDS[0], SEEDS[0] + a.n_seeds))
    # Seed-major order: every dataset and condition gains seeds together, so a partial
    # run is balanced and can already be summarised.
    jobs = [(d, r, nz, s) for s in seeds
            for d, r, nz in itertools.product(a.datasets, REGIONS, NOISES)]
    keys = ["dataset", "region", "noise", "seed"]
    cols = PILOT_COLS if a.stage.startswith("pilot") else COLS
    if a.stage.startswith("pilot"):
        out = RES / f"iter_{a.stage}.csv"
        jobs = [(d, r, nz, s) for s in PILOT["seeds"] for d in PILOT["datasets"]
                for r in PILOT["regions"] for nz in PILOT["noises"]]
        for name in PILOT["datasets"]:
            if name != "synth2d":
                _openml(name)
    if a.stage.endswith("pilot_smoke") or a.stage == "pilot2_smoke":
        jobs = [("sulfur", "hard", "clean", 9000)]
        out.unlink(missing_ok=True)
    elif a.stage == "smoke":
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
        res = Parallel(n_jobs=a.workers)(delayed(_safe)(j, dict(FIXED), a.stage)
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
