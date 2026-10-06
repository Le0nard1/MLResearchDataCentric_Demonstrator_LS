"""
Pilot: gradient-based subset selection (CRAIG, GLISTER) in the training-budget protocol
of the baseline comparison (paper Section 4.4, ``budget_v2 --stage main``).

Data, initial model, reserve, budget and retraining are exactly those of ``run_main``
(same seeds, same RNG use), so the rows pair with budget_v2_main.csv and
budget_v2_main_a1.csv; ``uniform`` is recomputed as a check of that pairing.

Both methods are adapted like RHO-LOSS: one selection per pass from the reserve instead
of a selection every few epochs, on last-layer gradients of the initial model (the usual
approximation of both papers). For squared loss the per-example gradient with respect
to the output layer is  g_i = (f(x_i) - y_i) * [h(x_i), 1],  h the last hidden layer.

    craig        facility-location core-set of the reserve in gradient space
                 (Mirzasoleiman et al., 2020), greedy, unweighted
    craig_w      the same subset with CRAIG's cluster-size weights, rescaled to mean 1
                 so the selected points carry the data volume of the budget
    glister      greedy selection maximising the first-order decrease of the validation
                 loss, with the output layer updated after every pick (Killamsetty et
                 al., 2021); the validation data are the other half of the reserve
                 (two folds, half of the picks from each), as RHO-LOSS's cross-fitting

Each runs at alpha = 0.2 (rest random rehearsal, as every competitor) and alpha = 1.

    python -m scripts.dataselect.pilot_craig_glister --seeds 50 --workers 18
    python -m scripts.dataselect.pilot_craig_glister --bench --workers 18   # further datasets
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
from scripts.dataselect.baselines import with_rehearsal
from scripts.dataselect.budget import pick_stratified
from scripts.dataselect import budget_v2 as B

OUT = B.RES / "pilot_craig_glister.csv"
OUT_BENCH = B.RES / "pilot_craig_glister_bench.csv"
GLISTER_ETA = 0.5          # total first-order step over all picks, relative (see glister)


def last_layer(m, X):
    """Last hidden layer h(x) and prediction of the fitted MLP pipeline."""
    sc, mlp = m.named_steps["scaler"], m.named_steps["model"]
    a = sc.transform(X)
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        a = np.maximum(a @ W + b, 0.0)
    return a, mlp.coefs_[-1][:, 0], float(mlp.intercepts_[-1][0])


def grads(H, w, b, y):
    r = H @ w + b - y
    return r[:, None] * np.hstack([H, np.ones((len(H), 1))])


def craig(G, k):
    """Greedy facility location on similarities C - ||g_i - g_j||; returns the picks and
    the number of reserve points each pick represents."""
    sq = (G ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G @ G.T, 0.0))
    S = D.max() - D
    cur = np.zeros(len(G))
    picks = []
    for _ in range(k):
        gain = np.maximum(S - cur[:, None], 0.0).sum(0)
        gain[picks] = -np.inf
        j = int(np.argmax(gain))
        picks.append(j)
        cur = np.maximum(cur, S[:, j])
    picks = np.asarray(picks)
    owner = picks[np.argmax(S[:, picks], axis=1)]
    weights = np.array([(owner == p).sum() for p in picks], float)
    return picks, weights


def glister(H_c, y_c, H_v, y_v, w, b, k):
    """Greedy one-step-lookahead selection: pick the candidate whose gradient step most
    decreases the validation loss, then take that step on the output layer."""
    w, b = w.copy(), b
    G_c = grads(H_c, w, b, y_c)
    eta = GLISTER_ETA / (k * np.mean((G_c ** 2).sum(1)) + 1e-12)
    picks = []
    avail = np.ones(len(H_c), bool)
    for _ in range(k):
        g_val = grads(H_v, w, b, y_v).mean(0)
        G_c = grads(H_c, w, b, y_c)
        score = np.where(avail, G_c @ g_val, -np.inf)
        j = int(np.argmax(score))
        picks.append(j)
        avail[j] = False
        w = w - eta * G_c[j, :-1]
        b = b - eta * G_c[j, -1]
    return np.asarray(picks)


def run(name, region, noise, seed):
    f = B.FIXED
    X, y, I, R, X_ev, y_ev, region_fn, to_U, rng = B.make(name, region, noise, seed)
    X_I, y_I, X_R, y_R = X[I], y[I], X[R], y[R]
    U_R = to_U(X_R)
    in_ev, in_R = region_fn(X_ev), region_fn(X_R)
    m0 = build_model("mlp", complexity=f["complexity"], iterations=f["iters_initial"],
                     warm_start=True, early_stopping=True)
    m0.fit(X_I, y_I)

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=float(e.mean()), err_in=float(e[in_ev].mean()),
                    err_out=float(e[~in_ev].mean()))

    def retrain(idx, w_new=None):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=f["iters_retrain"], early_stopping=False,
                       n_iter_no_change=f["iters_retrain"] + 1)
        mlp.best_loss_ = np.inf
        X_fit, y_fit = np.vstack([X_I, X_R[idx]]), np.concatenate([y_I, y_R[idx]])
        if w_new is None:
            m.fit(X_fit, y_fit)
        else:
            w = np.concatenate([np.ones(len(X_I)), w_new])
            m.fit(X_fit, y_fit, model__sample_weight=w / w.mean())
        return score(m)

    s0 = score(m0)
    base = dict(dataset=name, region=region, noise=noise, seed=seed,
                init_mae=s0["mae"], init_in=s0["err_in"], init_out=s0["err_out"])
    rows = []
    n, nR = f["n_select"], len(R)

    def add(arm, idx, w_new=None):
        rows.append({**base, "arm": arm, **retrain(idx, w_new),
                     "n_in_region": int(in_R[idx].sum())})

    def rs(salt):
        return np.random.RandomState(seed * 100 + salt)

    add("uniform", pick_stratified(U_R, n, rs(50)))      # pairing check

    H_R, w, b = last_layer(m0, X_R)
    G_R = grads(H_R, w, b, y_R)
    half = rs(120).permutation(nR)
    folds = (half[: nR // 2], half[nR // 2:])

    for alpha, tag in ((0.2, ""), (1.0, "_a1")):
        k = int(round(alpha * n))
        p, cw = craig(G_R, k)
        add("craig" + tag, with_rehearsal(p, n, nR, rs(110)))
        if alpha == 1.0:
            add("craig_w" + tag, p, w_new=cw / cw.mean())
        g = []
        for a_idx, v_idx in (folds, folds[::-1]):
            kk = k // 2 if not g else k - len(g)
            sel = glister(H_R[a_idx], y_R[a_idx], H_R[v_idx], y_R[v_idx], w, b, kk)
            g.extend(a_idx[sel].tolist())
        add("glister" + tag, with_rehearsal(np.asarray(g), n, nR, rs(111)))
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
    a = ap.parse_args()
    if a.bench:
        jobs = list(itertools.product(B.BENCH_DATASETS, B.REGIONS, B.NOISES, B.BENCH_SEEDS))
        out_file = OUT_BENCH
    else:
        jobs = list(itertools.product(B.DATASETS, B.REGIONS, B.NOISES,
                                      B.MAIN_SEEDS[: a.seeds]))
        out_file = OUT
    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=a.workers, verbose=5)(delayed(_safe)(j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(out_file, index=False)
    print(f"{len(jobs)} jobs, {time.time() - t0:.0f}s -> {out_file}")


if __name__ == "__main__":
    main()
