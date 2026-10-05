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
from scripts.dataselect.baselines import pick_kcenter, with_rehearsal

RES = Path("data/experiment_results/iterative_weakspot_curation")

FIXED = dict(n_data=4500, n_round=200, T=8, iters_initial=400, iters_retrain=200,
             k_diag=10, q_diag=0.3, n_random=3, complexity=0.5, gamma=0.5,
             kappa=1.5, switch_k=10.0, switch_mid=0.5)
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
              + ["size", "det_gain", "det_share", "n_fit", "epochs", "sample_updates",
                 "fit_sec", "select_sec",
                 "error"])
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
# Full Experiments 1-2 (Sections 4.2-4.3): every task, condition and label noise on
# fresh seeds; the arms of the second pilot plus random selection retrained on the
# accumulated set as the compute reference, on the first FULL["n_acc"] seeds only.
FULL = dict(seeds=list(range(8100, 8120)), n_acc=10)
ACC_REF = ("random_acc", "random", "static", "acc", 0.0, 0.0, 30)
# Third pilot: retraining on the new points plus a replay of n_replay earlier points
# (bounded compute, 2x the new-points-only budget), scarcity of the weak region and
# the initial training state. Conditions are "region:rho:init", e.g.
# "sparse_all:0.03:short"; random new-points-only is kept as the degradation check.
PILOT3 = dict(datasets=["synth2d", "houses", "sulfur"], noises=["clean"],
              regions=["sparse_init:0.25:std", "sparse_all:0.25:std",
                       "sparse_all:0.1:std", "sparse_all:0.03:std",
                       "sparse_init:0.25:short", "sparse_all:0.1:short"],
              seeds=list(range(9100, 9110)), n_replay=200, iters_short=30)
PILOT4 = dict(datasets=["synth2d", "houses", "sulfur"], noises=["clean"],
              regions=["sparse_init:0.25:std", "sparse_all:0.1:std"],
              seeds=list(range(9200, 9210)))
PILOT4_ARMS = (
    [("random", "random", "static", "replay", 0.0, 0.0, 20 + r) for r in range(2)]
    + [("ws_half", "region", "static", "replay", 0.5, 0.3, 82),
       ("ws_tuned", "region", "static", "replay", 1.0, 0.3, 70),
       ("ws_half_adaptive", "region", "adaptive", "replay", 0.5, 0.3, 86),
       ("ws_adaptive", "region", "adaptive", "replay", 1.0, 0.3, 73)]
    + [(f"ws{'_half' if a < 1 else ''}_{sch}", "region_adapt", sch, "replay", a, 0.3,
        100 + 10 * j + (a < 1))
       for j, sch in enumerate(["size_pow05", "size_pow2", "size_switch"])
       for a in (0.5, 1.0)])
# Full run of the replay design (pilots 3 and 4 at scale): every task, the scarcity and
# initial-training conditions of pilot 3, clean labels, fresh seeds; arms of pilot 3
# plus the six size-based schedules of pilot 4, so all comparisons share one control.
FULL2 = dict(regions=["sparse_init:0.25:std", "sparse_all:0.25:std",
                      "sparse_all:0.1:std", "sparse_all:0.03:std",
                      "sparse_init:0.25:short", "sparse_all:0.1:short"],
             noises=["clean"], seeds=list(range(8200, 8220)))
PILOT3_ARMS = (
    [("random", "random", "static", "replay", 0.0, 0.0, 20 + r) for r in range(2)]
    + [("random_new", "random", "static", "new", 0.0, 0.0, 29),
       ("ws_default", "region", "static", "replay", 0.2, 0.15, 81),
       ("ws_half", "region", "static", "replay", 0.5, 0.3, 82),
       ("ws_tuned", "region", "static", "replay", 1.0, 0.3, 70),
       ("ws_dynamic", "region", "dynamic", "replay", 1.0, 0.3, 72),
       ("ws_adaptive", "region", "adaptive", "replay", 1.0, 0.3, 73),
       ("ws_half_adaptive", "region", "adaptive", "replay", 0.5, 0.3, 86)])


# Per-round traces for figures, filled only when FIXED["record"] is set (in-process).
TRACE = []


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
                c0=float(c[0]), c1=float(c[1]),
                # weakspot size: share of the held-out half whose landscape exceeds
                # kappa x the median of the landscape fitted on the other half
                size=float((land_B > FIXED['kappa'] * np.median(land_A)).mean()))
    return diag, land, c


def alpha_schedule(schedule, alpha, t, S_t, S_1, z_t=None, z_1=None):
    """Guidance fraction of round t (Section 3.4)."""
    if schedule == "static":
        return alpha
    if schedule == "dynamic":                  # fixed decay, front-loads the focus
        return alpha * FIXED["gamma"] ** (t - 1)
    if schedule == "adaptive":                 # scaled by the remaining severity
        return alpha * float(np.clip((S_t - 1.0) / max(S_1 - 1.0, 1e-9), 0.0, 1.0))
    if schedule.startswith("size"):        # scaled by the remaining weakspot size
        r = float(np.clip(z_t / max(z_1, 1e-9), 0.0, 1.0)) if z_1 else 0.0
        if schedule == "size_pow05":         # holds the focus until nearly gone
            return alpha * r ** 0.5
        if schedule == "size_pow2":          # releases as soon as it shrinks
            return alpha * r ** 2
        if schedule == "size_switch":        # full focus above half the size
            k, mid = FIXED["switch_k"], FIXED["switch_mid"]
            return alpha / (1.0 + np.exp(-k * (r - mid)))
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
    pilot = stage.startswith(("pilot", "full"))
    base_region, rho, init = (region.split(":") + [None, None])[:3]
    B.FIXED.update(n_data=f["n_data"], rho=float(rho) if rho else 0.25)
    X, y, I, R, X_ev, y_ev, region_fn, to_U, _ = B.make(name, base_region, noise, seed)
    anchor = _closure(region_fn, "a")
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    U_R = to_U(X_R)
    in_ev, in_R = region_fn(X_ev), region_fn(X_R)
    nR, n = len(R), f["n_round"]

    iters0 = PILOT3["iters_short"] if init == "short" else f["iters_initial"]
    m0 = build_model("mlp", complexity=f["complexity"], iterations=iters0,
                     warm_start=True, early_stopping=True)
    m0.fit(X_I, y_I)
    # Irreducible loss for the RHO-LOSS combinations: cross-fitted once on the reserve,
    # independent of the model being trained (Paper A, Section 3).
    needs_il = pilot and not stage.startswith(("pilot3", "pilot4", "full2"))
    il = B.irreducible_loss(X_I, y_I, X_R, y_R, seed) if needs_il else None

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

    U_ev = to_U(X_ev)
    RPRED = {}                                 # control predictions per round

    def loop(arm, traj, alpha, q, salt, method="region", schedule="static",
             regime="acc"):
        rs = np.random.RandomState(seed * 1000 + salt)
        m = copy.deepcopy(m0)
        avail = np.ones(nR, bool)
        chosen = []
        prev_c, S_1, z_1 = None, None, None
        tag = dict(method=method, schedule=schedule, regime=regime) if pilot else {}
        rows.append({**base, "arm": arm, "traj": traj, "round": 0, **tag, **s0})
        for t in range(1, f["T"] + 1):
            ts = time.process_time()
            av = np.flatnonzero(avail)
            e = np.abs(y_R[av] - m.predict(X_R[av]))
            diag, land, prev_c = diagnose(U_R[av], e, in_R[av], anchor, prev_c, rs)
            S_1 = diag["severity"] if S_1 is None else S_1
            z_1 = diag["size"] if t == 1 else z_1
            a_t = alpha_schedule(schedule, alpha, t, diag["severity"], S_1,
                                 diag["size"], z_1)
            kk = int(round(a_t * n))
            if method == "kcenter_part":       # k-center share + uniform rest
                used = np.vstack([to_U(X_I), U_R[np.asarray(chosen, dtype=int)]]) \
                    if chosen else to_U(X_I)
                g = pick_kcenter(U_R[av], used, kk)
                pick = with_rehearsal(g, n, len(av), rs)
            elif method == "kcenter":
                used = np.vstack([to_U(X_I), U_R[np.asarray(chosen, dtype=int)]]) \
                    if chosen else to_U(X_I)
                pick = pick_kcenter(U_R[av], used, n)
            elif method == "random" or kk == 0:
                pick = rs.choice(len(av), n, replace=False)
            else:
                members = np.flatnonzero(land >= np.quantile(land, 1 - q))
                if method in ("region_adapt", "region_adapt_kc"):  # elevated area, capped
                    elev = np.flatnonzero(land > f["kappa"] * np.median(land))
                    if len(elev) < len(members):
                        members = elev
                if len(members) == 0:
                    members = np.arange(len(av))
                if method in ("region", "region_adapt", "region_adapt_kc", "region_kc"):
                    g = rs.choice(members, min(kk, len(members)), replace=False)
                elif method == "rho_filter":       # reducible loss within the region
                    red = e - il[av]
                    g = members[np.argsort(red[members])[::-1][:min(kk, len(members))]]
                elif method == "rho_landscape":    # reducible loss x error landscape
                    g = np.argsort(np.maximum(e - il[av], 0) * land)[::-1][:kk]
                if method in ("region_adapt_kc", "region_kc"):  # rest covers the space
                    rest = np.setdiff1d(np.arange(len(av)), g)
                    used = np.vstack([to_U(X_I), U_R[np.asarray(chosen, dtype=int)]]) \
                        if chosen else to_U(X_I)
                    kc = pick_kcenter(U_R[av][rest], np.vstack([used, U_R[av][g]]),
                                      n - len(g))
                    pick = np.concatenate([g, rest[kc]]).astype(int)
                else:
                    pick = with_rehearsal(g, n, len(av), rs)
            # Detected region on the test set (for the error reduction inside it)
            if stage.startswith("full2_abl"):
                dk, nk = cKDTree(U_R[av]).query(U_ev, k=f["k_diag"])
                wk = 1.0 / np.maximum(dk, 1e-12)
                land_ev = (wk * land[nk]).sum(1) / wk.sum(1)
                det_mask = land_ev >= np.quantile(land, 1 - f["q_diag"])
            sel = av[pick]
            select_sec = time.process_time() - ts
            if f.get("record"):
                TRACE.append(dict(arm=arm, traj=traj, regime=regime, round=t,
                                  X_av=X_R[av], U_av=U_R[av], land=land,
                                  centre_X=X_R[av][int(np.argmax(land))],
                                  centre_U=U_R[av][int(np.argmax(land))],
                                  sel_X=X_R[sel], sel_U=U_R[sel], region_fn=region_fn,
                                  to_U=to_U, U_R=U_R, in_R=in_R, anchor=anchor))
            avail[sel] = False
            chosen.extend(sel.tolist())
            if regime == "acc":                # initial set plus every selection so far
                c = np.asarray(chosen)
                cost = continue_training(m, np.vstack([X_I, X_R[c]]),
                                         np.concatenate([y_I, y_R[c]]))
            elif regime == "replay":           # new points + replay of earlier data
                prev = np.asarray(chosen[:-len(sel)], dtype=int)
                X_old = np.vstack([X_I, X_R[prev]]) if len(prev) else X_I
                y_old = np.concatenate([y_I, y_R[prev]]) if len(prev) else y_I
                k = rs.choice(len(X_old), min(PILOT3["n_replay"], len(X_old)),
                              replace=False)
                cost = continue_training(m, np.vstack([X_R[sel], X_old[k]]),
                                         np.concatenate([y_R[sel], y_old[k]]))
            else:                              # this round's selection alone
                cost = continue_training(m, X_R[sel], y_R[sel])
            det = {}
            if stage.startswith("full2_abl"):
                pred = m.predict(X_ev)
                if arm == "random":
                    RPRED[t] = pred
                elif t in RPRED:
                    e_ws = np.abs(y_ev - pred)[det_mask]
                    e_rn = np.abs(y_ev - RPRED[t])[det_mask]
                    det = dict(det_gain=float(100 * (e_rn.mean() - e_ws.mean())
                                              / e_rn.mean()),
                               det_share=float(det_mask.mean()))
            rows.append({**base, "arm": arm, "traj": traj, "round": t, **tag,
                         "alpha_t": a_t, **score(m), **det,
                         "n_in_region": int(in_R[sel].sum()), **diag,
                         **(dict(n_fit=cost[0], epochs=cost[1],
                                 sample_updates=cost[0] * cost[1], fit_sec=cost[2],
                                 select_sec=select_sec) if pilot else {})})

    if pilot:
        arms = PILOT2_ARMS if stage.startswith(("pilot2", "full")) else PILOT_ARMS
        if stage.startswith("pilot3"):
            arms = PILOT3_ARMS
        if stage.startswith("pilot4"):
            arms = PILOT4_ARMS
        if stage.startswith("full2"):
            arms = list(PILOT3_ARMS) + list(a for a in PILOT4_ARMS
                                       if a[2].startswith("size"))
        if stage.startswith("full2_abl"):    # ablation + hybrid; pairs with full2
            arms = [("random", "random", "static", "replay", 0.0, 0.0, 20),
                    ("ws_half", "region", "static", "replay", 0.5, 0.3, 82),
                    ("ws_half_size_pow2", "region_adapt", "size_pow2", "replay", 0.5,
                     0.3, 111),
                    ("ws_half_size_pow2_fixedreg", "region", "size_pow2", "replay",
                     0.5, 0.3, 130),
                    ("ws_half_adaptreg", "region_adapt", "static", "replay", 0.5, 0.3,
                     131),
                    ("hybrid_size_pow2_kc", "region_adapt_kc", "size_pow2", "replay",
                     0.5, 0.3, 132)]
        if stage.startswith("full2_ctl"):    # controls for focus + coverage
            arms = [("kcenter_half_uniform", "kcenter_part", "static", "replay", 0.5, 0.0,
                     140),
                    ("hybrid_fixedreg", "region_kc", "size_pow2", "replay", 0.5, 0.3, 141)]
        if stage.startswith("full2_kc"):     # baseline only; pairs with iter_full2.csv
            arms = [("kcenter", "kcenter", "static", "replay", 1.0, 0.0, 90)]
        if stage.startswith("full") and seed < FULL["seeds"][0] + FULL["n_acc"]:
            arms = arms + [ACC_REF]
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
    ap.add_argument("--stage", choices=["smoke", "static", "pilot", "pilot_smoke", "pilot2", "pilot2_smoke",
                             "pilot3", "pilot3_smoke", "pilot4", "pilot4_smoke",
                             "full2", "full2_smoke", "full2_kc", "full2_kc_smoke",
                             "full2_abl", "full2_abl_smoke", "full2_ctl", "full2_ctl_smoke",
                             "full", "full_smoke"], required=True)
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
    cols = PILOT_COLS if a.stage.startswith(("pilot", "full")) else COLS
    if a.stage.startswith("full"):
        out = RES / f"iter_{a.stage}.csv"
        jobs = [(d, r, nz, s) for s in FULL["seeds"]
                for d, r, nz in itertools.product(DATASETS, REGIONS, NOISES)]
        if a.stage == "full_smoke":
            jobs = [(d, "hard", "outlier", FULL["seeds"][0]) for d in DATASETS]
            out.unlink(missing_ok=True)
    if a.stage.startswith("pilot"):
        out = RES / f"iter_{a.stage}.csv"
        jobs = [(d, r, nz, s) for s in PILOT["seeds"] for d in PILOT["datasets"]
                for r in PILOT["regions"] for nz in PILOT["noises"]]
        for name in PILOT["datasets"]:
            if name != "synth2d":
                _openml(name)
    if a.stage.startswith("pilot3"):
        jobs = [(d, r, nz, s) for s in PILOT3["seeds"] for d in PILOT3["datasets"]
                for r in PILOT3["regions"] for nz in PILOT3["noises"]]
    if a.stage.startswith("full2"):
        out = RES / f"iter_{a.stage}.csv"
        jobs = [(d, r, nz, s) for s in FULL2["seeds"] for d in DATASETS
                for r in FULL2["regions"] for nz in FULL2["noises"]]
        for name in DATASETS:
            if name != "synth2d":
                _openml(name)
    if a.stage in ("full2_smoke", "full2_kc_smoke", "full2_abl_smoke", "full2_ctl_smoke"):
        jobs = [("nyc_taxi", "sparse_all:0.03:std", "clean", 8200)]
        out.unlink(missing_ok=True)
    if a.stage.startswith("pilot4"):
        jobs = [(d, r, nz, s) for s in PILOT4["seeds"] for d in PILOT4["datasets"]
                for r in PILOT4["regions"] for nz in PILOT4["noises"]]
    if a.stage == "pilot4_smoke":
        jobs = [("synth2d", "sparse_init:0.25:std", "clean", 9200)]
        out.unlink(missing_ok=True)
    elif a.stage == "pilot3_smoke":
        jobs = [("sulfur", "sparse_all:0.03:short", "clean", 9100)]
        out.unlink(missing_ok=True)
    elif a.stage.endswith("pilot_smoke") or a.stage == "pilot2_smoke":
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
    out.parent.mkdir(parents=True, exist_ok=True)
    # Each job is appended as soon as it finishes, so an interruption loses at most
    # the jobs still running; a restart skips every job already in the file.
    res = Parallel(n_jobs=a.workers, return_as="generator_unordered")(
        delayed(_safe)(j, dict(FIXED), a.stage) for j in jobs)
    for m, rows in enumerate(res, 1):
        df = pd.DataFrame(rows).reindex(columns=cols)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        el = time.time() - t0
        print(f"{m}/{len(jobs)}  {el/60:.1f} min  ETA {(el/m)*(len(jobs)-m)/60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
