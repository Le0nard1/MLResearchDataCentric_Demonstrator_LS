"""
Fixed-data reweighting beyond two dimensions (paper Section 5.3).

The protocol of ``fixed_data.py`` (1000 labelled points, 80% train, diagnosis on
the residuals of all 1000, continuation for a fixed number of epochs under
per-example weights of mean 1, compared with uniform continuation) with the one
2-D-specific step replaced: detection on a regular grid. Here the error is
smoothed at the data points themselves (kNN, k = 10, distance-weighted, the point
itself left out) and the centre is the point of largest smoothed error. This works
in any dimension and in any feature space.

Two spaces are used for detection and kernel:
    input      the standardised inputs (the model's own scaler)
    embedding  the initial MLP's last hidden layer (standardised, dead units dropped),
               a learnt representation
A Gaussian kernel on distances does not transfer across dimensions: distances
concentrate, so a fixed width puts all weight on the centre point in 20-D, and a
width set to a distance quantile is almost flat (ESS >= 0.9 even at alpha = 1 for
d >= 10; the pilot, fixed_data_hd_pilot.csv, shows it). The kernel therefore acts on
neighbour ranks, K_i = exp(-rank_i / (q N)), rank 0 the centre's nearest training
point: the q-fraction of points closest to the centre carries the focus in any
dimension (in 2-D this is close to a disc kernel; q = 0.03, 0.1, 0.3 correspond to
radii of about 0.1, 0.18, 0.31 on the unit square).

Tasks
    synth, d in {2, 5, 10, 20}   the bump target and hard region of fixed_data.py on
        (x1, x2), plus a smooth term g over the remaining coordinates, inputs uniform
        on [0,1]^d; the dataset size n(d) and g follow the setup rule (--stage setup). The weakspot is the cylinder
        ||(x1,x2) - (0.25,0.75)|| <= 0.2 (ground truth). Conditions: hard/gauss,
        hard/outlier, none/gauss.
    cal8  California housing with all eight features (income, age, log rooms,
        log bedrooms, log population, log occupancy, latitude, longitude).
        "Inside" = the 10% of test districts nearest to the detected centre.

Arms: uniform, weakspot (input), weakspot_emb (embedding), random_emb (the embedding
kernel around a random training point, 2 draws; random_centre is its input-space twin),
oracle (synthetic: kernel on the true (x1,x2) centre), smoothed_loss, loss, jtt.

Reported stage (``--stage frontier``): every arm over a fixed grid on seeds
4000-4019 (4000-4009 at d = 10, 20), no tuning, as the 2-D trade-off figure. Tuning on whole-area MAE
(``--stage pilot``) was abandoned: it selects near-uniform weights (ESS ~ 1) for every
arm, since focus on fixed data reallocates error rather than removing it.

    python -m scripts.dataselect.fixed_data_hd --stage frontier --workers 19
"""
from __future__ import annotations

import argparse
import copy
import json
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

from scripts.weakspot.models import build_model
from scripts.dataselect import fixed_data as F

RES = F.RES
HP_FILE = RES / "fixed_data_hd_hp.json"

HD = dict(n_data=1000, g="cos", early_stopping=False, val_frac=0.2, iters_initial=400, iters_cont=200, complexity=0.5,
          n_eval=4000, k_diag=10, inside_frac=0.1, n_rand=5)
DIMS = (2, 5, 10, 20)
PILOT_SEEDS, MAIN_SEEDS = F.PILOT_SEEDS, F.MAIN_SEEDS
TASKS = [f"synth{d}" for d in DIMS] + ["cal8"]
CONDITIONS = {**{f"synth{d}": [dict(region="hard", noise="gauss"),
                               dict(region="hard", noise="outlier"),
                               dict(region="none", noise="gauss")] for d in DIMS},
              "cal8": [dict(region="real", noise="real")]}
# compute: the no-weakspot control runs up to d = 5 (16000 points at d = 20)
for _d in (10, 20):
    CONDITIONS[f"synth{_d}"] = CONDITIONS[f"synth{_d}"][:2]
REFERENCE = {t: c[0] for t, c in CONDITIONS.items()}

Q_GRID = (0.03, 0.1, 0.3)
HP_GRID = {
    "weakspot": [dict(alpha=a, q=q) for a in (0.2, 0.5, 0.8, 1.0) for q in Q_GRID],
    "weakspot_emb": [dict(alpha=a, q=q) for a in (0.2, 0.5, 0.8, 1.0) for q in Q_GRID],
    "loss": [dict(alpha=a) for a in (0.2, 0.5, 0.8, 1.0)],
    "smoothed_loss": [dict(alpha=a) for a in (0.2, 0.5, 0.8, 1.0)],
    "jtt": [dict(q=q, lam=l) for q in (0.1, 0.2) for l in (2, 5, 10, 20)],
    "density": [dict(alpha=a) for a in (0.2, 0.5, 0.8, 1.0)],
}


def controls(task):
    return ["random_centre", "oracle"] if task.startswith("synth") else ["random_centre"]


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
N_BY_DIM = {2: 1000, 5: 4000, 10: 8000, 20: 16000}    # set by the setup rule


def target_hd(X, region, g=None):
    y = F.target(X[:, :2], region)
    if X.shape[1] > 2:
        m, Xr = X.shape[1] - 2, X[:, 2:]
        if (g or HD["g"]) == "cos":
            y = y + np.cos(np.pi * Xr).sum(1) / np.sqrt(m)
        else:                                               # "lin"
            y = y + 2 * (Xr - 0.5).sum(1) / np.sqrt(m)
    return y


def make_synth(d, region, noise, seed, n=None):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n or N_BY_DIM[d], d))
    y = target_hd(X, region) + F._noise(X[:, :2], rng, noise)
    X_ev = rng.uniform(0, 1, size=(HD["n_eval"], d))
    return X, y, X_ev, target_hd(X_ev, region)


_CAL8 = None


def make_cal8(seed):
    global _CAL8
    if _CAL8 is None:
        a = np.load(F.CAL_RAW)
        lon, lat, age, rooms, beds, pop, hh, inc, val = a.T
        X = np.column_stack([inc, age, np.log(rooms / hh), np.log(beds / hh),
                             np.log(pop), np.log(pop / hh), lat, lon])
        _CAL8 = (X, val / 1e5)
    X_all, y_all = _CAL8
    perm = np.random.RandomState(seed).permutation(len(X_all))
    d, t = perm[: HD["n_data"]], perm[HD["n_data"]:][: HD["n_eval"]]
    return X_all[d], y_all[d], X_all[t], y_all[t]


def make(task, cond, seed):
    if task == "cal8":
        return make_cal8(seed)
    return make_synth(int(task[5:]), cond["region"], cond["noise"], seed)


# ─────────────────────────────────────────────────────────────
# Spaces, detection, weights
# ─────────────────────────────────────────────────────────────
def embedder(m0, X_tr):
    """Last hidden layer of the fitted MLP pipeline, standardised on the training set."""
    sc, mlp = m0.named_steps["scaler"], m0.named_steps["model"]

    def h(X):
        a = sc.transform(X)
        for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
            a = np.maximum(a @ W + b, 0.0)
        return a

    H = h(X_tr)
    mu, sd = H.mean(0), H.std(0)
    live = sd > 1e-8
    return lambda X: (h(X)[:, live] - mu[live]) / sd[live]


def detect(Z, err, k):
    """Index of the point of largest leave-one-out kNN-smoothed error."""
    dist, idx = cKDTree(Z).query(Z, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    w = 1.0 / np.maximum(dist, 1e-12)
    s = (w * err[idx]).sum(1) / w.sum(1)
    return int(np.argmax(s))


def kernel(Z, c, q):
    """Rank kernel: exp(-rank / (q N)) over the distance ranks to the centre."""
    rank = np.argsort(np.argsort(((Z - c) ** 2).sum(1)))
    return np.exp(-rank / (q * len(Z)))


def nearest_frac(Z, c, frac):
    d = np.linalg.norm(Z - c, axis=1)
    return d <= np.quantile(d, frac)


# ─────────────────────────────────────────────────────────────
# One (task, condition, seed)
# ─────────────────────────────────────────────────────────────
def run_one(task, cond, seed, arm_hps):
    X, y, X_ev, y_ev = make(task, cond, seed)
    synth = task.startswith("synth")
    n_val = int(round(HD["val_frac"] * len(X)))
    X_tr, y_tr = X[n_val:], y[n_val:]
    m0 = build_model("mlp", complexity=HD["complexity"], iterations=HD["iters_initial"],
                     warm_start=True, early_stopping=HD["early_stopping"])
    m0.fit(X_tr, y_tr)
    res_tr = np.abs(y_tr - m0.predict(X_tr))
    err_d = np.abs(y - m0.predict(X))

    sc = m0.named_steps["scaler"]
    emb = embedder(m0, X_tr)
    Z, Z_tr, Z_ev = sc.transform(X), sc.transform(X_tr), sc.transform(X_ev)
    E, E_tr, E_ev = emb(X), emb(X_tr), emb(X_ev)
    i_in, i_emb = detect(Z, err_d, HD["k_diag"]), detect(E, err_d, HD["k_diag"])
    c_in, c_emb = Z[i_in], E[i_emb]
    c_true = np.asarray(F.FIXED["centre"])

    if synth:
        inside = np.linalg.norm(X_ev[:, :2] - c_true, axis=1) <= F.FIXED["radius"]
    else:
        inside = nearest_frac(Z_ev, c_in, HD["inside_frac"])
    inside_emb = nearest_frac(E_ev, c_emb, HD["inside_frac"])

    def score(m):
        e = np.abs(y_ev - m.predict(X_ev))
        return dict(mae=e.mean(), err_in=e[inside].mean(), err_out=e[~inside].mean(),
                    err_in_emb=e[inside_emb].mean())

    s0 = score(m0)
    base = dict(task=task, region=cond["region"], noise=cond["noise"], seed=seed,
                dim=X.shape[1], emb_dim=E.shape[1],
                init_mae=s0["mae"], init_in=s0["err_in"], init_out=s0["err_out"],
                const_mae=float(np.abs(y_ev - y_tr.mean()).mean()),
                detect_dist=(float(np.linalg.norm(X[i_in, :2] - c_true)) if synth else np.nan),
                detect_dist_emb=(float(np.linalg.norm(X[i_emb, :2] - c_true))
                                 if synth else np.nan))

    rng_c = np.random.RandomState(seed + 11)
    rand_idx = rng_c.choice(len(Z_tr), HD["n_rand"], replace=False)

    def weight_sets(arm, hp):
        if arm == "uniform":
            return [np.ones(len(X_tr))]
        if arm == "weakspot":
            return [F._mix(kernel(Z_tr, c_in, hp["q"]), hp["alpha"])]
        if arm == "weakspot_emb":
            return [F._mix(kernel(E_tr, c_emb, hp["q"]), hp["alpha"])]
        if arm == "random_centre":
            return [F._mix(kernel(Z_tr, Z_tr[i], hp["q"]), hp["alpha"]) for i in rand_idx]
        if arm == "random_emb":
            return [F._mix(kernel(E_tr, E_tr[i], hp["q"]), hp["alpha"]) for i in rand_idx]
        if arm == "oracle":                 # the true centre on the (x1, x2) plane
            return [F._mix(kernel(X_tr[:, :2], c_true, hp["q"]), hp["alpha"])]
        if arm == "smoothed_loss":
            return [F._mix(F.smoothed(Z_tr, res_tr), hp["alpha"])]
        if arm == "density":
            d5 = cKDTree(Z_tr).query(Z_tr, k=6)[0][:, -1]
            return [F._mix(d5, hp["alpha"])]
        return [F.weights(arm, hp, Z_tr, res_tr, None)]      # loss, jtt

    def cont(w):
        m = copy.deepcopy(m0)
        mlp = m.named_steps["model"]
        mlp.set_params(max_iter=HD["iters_cont"], early_stopping=False,
                       n_iter_no_change=HD["iters_cont"] + 1)
        mlp.best_loss_ = np.inf
        m.fit(X_tr, y_tr, model__sample_weight=w)
        return m

    rows = []
    for arm, hp in arm_hps:
        ws = weight_sets(arm, hp)
        sc_ = [score(cont(w)) for w in ws]
        s = {k: float(np.mean([x[k] for x in sc_])) for k in sc_[0]}
        tag = arm + ("" if not hp else "|" + ",".join(f"{k}={v}" for k, v in hp.items()))
        rows.append({**base, "arm": arm, "hp": tag, **s,
                     "ess": float(np.mean([F.ess(w) for w in ws]))})
    return rows


def _safe(task, cond, seed, arm_hps):
    try:
        return run_one(task, cond, seed, arm_hps)
    except Exception as e:                                  # noqa: BLE001
        return [dict(task=task, seed=seed, arm="ERROR", hp=repr(e), **cond)]


# ─────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────
def _setup_one(d, n, g, seed):
    HD["g"] = g
    X, y, X_ev, y_ev = make_synth(d, "hard", "gauss", seed, n=n)
    n_val = int(round(HD["val_frac"] * len(X)))
    m0 = build_model("mlp", complexity=HD["complexity"], iterations=HD["iters_initial"],
                     warm_start=True, early_stopping=HD["early_stopping"])
    m0.fit(X[n_val:], y[n_val:])
    e = np.abs(y_ev - m0.predict(X_ev))
    inR = np.linalg.norm(X_ev[:, :2] - np.asarray(F.FIXED["centre"]), axis=1) <= F.FIXED["radius"]
    err_d = np.abs(y - m0.predict(X))
    Z, E = m0.named_steps["scaler"].transform(X), embedder(m0, X[n_val:])(X)
    c = np.asarray(F.FIXED["centre"])
    return dict(d=d, n=n, g=g, seed=seed, init_mae=e.mean(),
                const_mae=np.abs(y_ev - y_ev.mean()).mean(),
                ratio=e[inR].mean() / e[~inR].mean(),
                dist=np.linalg.norm(X[detect(Z, err_d, HD["k_diag"]), :2] - c),
                dist_emb=np.linalg.norm(X[detect(E, err_d, HD["k_diag"]), :2] - c))


def setup(workers):
    """Dataset size and extra-coordinate term per dimension, from the initial model alone.

    Rule (as the 2-D setup): per d, keep the dataset sizes whose initial model reaches
    at most half the error of the constant predictor with in/out error ratio >= 2, and
    take the smallest. Detection is reported, not used in the rule. The initial model
    trains the full 400 epochs: with early stopping on its internal split it halts after
    a few epochs for d >= 10 and stays barely fitted at any n. With g = cos, 6 pilot
    seeds: n = 1000, 4000, 8000, 16000 for d = 2, 5, 10, 20 (2000 at d = 5, 4000 at
    d = 10 and 8000 at d = 20 fail the ratio).
    """
    from joblib import Parallel, delayed
    import itertools
    cands = [(2, 1000), (5, 2000), (5, 4000), (10, 4000), (10, 8000), (20, 8000),
             (20, 16000)]
    jobs = [(d, n, "cos", s) for d, n in cands for s in PILOT_SEEDS[:6]]
    t0 = time.time()
    out = Parallel(n_jobs=workers)(delayed(_setup_one)(*j) for j in jobs)
    print(f"setup: {len(jobs)} jobs, {(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(RES / "fixed_data_hd_setup.csv", index=False)
    g = df.groupby(["d", "n"])[["init_mae", "const_mae", "ratio", "dist", "dist_emb"]].mean()
    g["ok"] = (g.init_mae <= 0.5 * g.const_mae) & (g.ratio >= 2)
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())


def pilot(workers, tasks):
    from joblib import Parallel, delayed
    arm_hps = [("uniform", {})] + [(a, hp) for a, g in HP_GRID.items() for hp in g]
    jobs = [(t, REFERENCE[t], s) for t in tasks for s in PILOT_SEEDS[:10]]
    t0 = time.time()
    out = Parallel(n_jobs=workers)(delayed(_safe)(t, c, s, arm_hps) for t, c, s in jobs[::-1])
    print(f"pilot: {len(jobs)} jobs, {(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(RES / "fixed_data_hd_pilot.csv", index=False)
    chosen = json.loads(HP_FILE.read_text()) if HP_FILE.exists() else {}
    for task, g in df[df.arm != "ERROR"].groupby("task"):
        m = g.groupby(["arm", "hp"])["mae"].mean()
        chosen[task] = {}
        for arm in HP_GRID:
            best = m.loc[arm].idxmin()
            chosen[task][arm] = next(hp for hp in HP_GRID[arm] if arm + "|" + ",".join(
                f"{k}={v}" for k, v in hp.items()) == best)
        print(task, chosen[task], flush=True)
    HP_FILE.write_text(json.dumps(chosen, indent=2))


def main_stage(workers, tasks):
    from joblib import Parallel, delayed
    chosen = json.loads(HP_FILE.read_text())
    jobs = []
    for t in tasks:
        arm_hps = ([("uniform", {})] + [(a, chosen[t][a]) for a in HP_GRID]
                   + [(a, chosen[t]["weakspot"]) for a in controls(t)])
        jobs += [(t, c, s, arm_hps) for c in CONDITIONS[t] for s in MAIN_SEEDS]
    t0 = time.time()
    jobs.sort(key=lambda j: -N_BY_DIM.get(int(j[0][5:]) if j[0] != "cal8" else 2, 0))
    out = Parallel(n_jobs=workers)(delayed(_safe)(*j) for j in jobs)
    print(f"main: {len(jobs)} jobs, {(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(RES / "fixed_data_hd.csv",
                                                           index=False)


# Compute (14 cores; one d = 20 job is ~45 min): 20 seeds up to d = 5 and California,
# 10 seeds at d = 10 and d = 20, the outlier condition up to d = 10.
FR_SEEDS = {"synth2": MAIN_SEEDS[:20], "synth5": MAIN_SEEDS[:20], "synth10": MAIN_SEEDS[:10],
            "synth20": MAIN_SEEDS[:10], "cal8": MAIN_SEEDS[:20]}
FR_CONDITIONS = {t: [c for c in cs if c["region"] != "none"] for t, cs in CONDITIONS.items()}
FR_CONDITIONS["synth20"] = FR_CONDITIONS["synth20"][:1]
FR_KERNEL = [dict(alpha=a, q=q) for a in (0.5, 1.0) for q in Q_GRID]
FR_GRID = ([("uniform", {})]
           + [(a, hp) for a in ("weakspot", "weakspot_emb", "random_emb")
              for hp in FR_KERNEL]
           + [(a, dict(alpha=x)) for a in ("loss", "smoothed_loss") for x in (0.5, 1.0)]
           + [("jtt", dict(q=0.2, lam=l)) for l in (2, 5, 10)])


def frontier(workers, tasks):
    """Every arm over a fixed grid on the reported seeds (descriptive, no selection)."""
    from joblib import Parallel, delayed
    HD["n_rand"] = 2
    jobs = []
    for t in tasks:
        grid = FR_GRID + ([("oracle", hp) for hp in FR_KERNEL] if t != "cal8" else [])
        jobs += [(t, c, s, grid) for c in FR_CONDITIONS[t] for s in FR_SEEDS[t]]
    jobs.sort(key=lambda j: -N_BY_DIM.get(int(j[0][5:]) if j[0] != "cal8" else 2, 0))
    t0 = time.time()
    f = RES / "fixed_data_hd_frontier.csv"
    f.unlink(missing_ok=True)
    gen = Parallel(n_jobs=workers, return_as="generator_unordered")(
        delayed(_safe)(*j) for j in jobs)
    for i, rows in enumerate(gen, 1):        # written as jobs finish
        pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"{i}/{len(jobs)} {rows[0]['task']} {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"frontier: {len(jobs)} jobs, {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "setup", "pilot", "main", "frontier"], required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    a = ap.parse_args()
    if a.stage == "smoke":
        t0 = time.time()
        for t in a.tasks:
            rows = run_one(t, REFERENCE[t], 1, [("uniform", {}),
                           ("weakspot", dict(alpha=0.5, q=0.1)),
                           ("weakspot_emb", dict(alpha=0.5, q=0.1)),
                           ("random_centre", dict(alpha=0.5, q=0.1))]
                           + ([("oracle", dict(alpha=0.5, q=0.1))] if t != "cal8" else []))
            print(pd.DataFrame(rows)[["task", "arm", "init_mae", "const_mae", "mae", "err_in",
                                      "err_out", "detect_dist", "detect_dist_emb", "ess"]]
                  .round(3).to_string(), flush=True)
        print(f"{time.time()-t0:.1f}s")
    elif a.stage == "frontier":
        frontier(a.workers, a.tasks)
    elif a.stage == "setup":
        setup(a.workers)
    elif a.stage == "pilot":
        pilot(a.workers, a.tasks)
    else:
        main_stage(a.workers, a.tasks)


if __name__ == "__main__":
    main()
