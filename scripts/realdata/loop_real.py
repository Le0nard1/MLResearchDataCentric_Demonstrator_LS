"""The iterative data-selective loop on a real, fixed tabular dataset.

Everything the synthetic study assumed away is real here: there is no ground
truth beyond the labels, no induced gap, and no fresh candidates --- the
dataset IS the pool, consumed without replacement, which is the fixed-dataset
industrial premise the papers argue from. The loop is the same as
``scripts.iterative.loop``: train, evaluate, identify the weakspot, select
around it, retrain on the new points alone (warm start, fixed epoch budget),
against a random baseline drawing the same counts from the same pool.

What must change on real data:

* **Detection.** The 2-D grid detectors do not apply in twenty dimensions.
  The weakspot is found directly on the evaluation set: each eval point's
  error is smoothed over its k nearest neighbours in standardised feature
  space, and the detected centre is the point where that local error peaks ---
  the kNN-performance-mapping idea without the grid. Severity is the
  kernel-weighted mean error around the centre over the global mean error,
  the same ground-truth-free signal the adaptive policy uses in the papers.

* **Distance with categorical columns.** Mixed-type dissimilarity is a solved
  problem — Gower's similarity coefficient (Biometrics 27, 1971) and the
  heterogeneous Euclidean/overlap distance functions of instance-based
  learning (Wilson & Martinez, JAIR 6, 1997) — so the encodings here follow
  that literature rather than reinvent it.
  ``numeric``    drops the categorical columns everywhere (model + distance):
                 the naive baseline.
  ``onehot``     one-hot encodes them into the model features AND the
                 standardised distance space — the Euclidean-overlap (HEOM)
                 construction: a candidate in the wrong category is simply
                 *farther away* and "closest data point" stays well defined.
  ``stratified`` one-hot for the model, numeric-only distance, but the guided
                 share is drawn to per-category quotas matching the weakspot's
                 own categorical distribution (classical stratified sampling,
                 Neyman 1934, applied to the guided draw).

Mix policies mirror the paper: static (constant alpha), dynamic (alpha x0.1
per round) and adaptive (alpha scaled by remaining severity).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_CSV = Path("data/raw/iris_extended.csv")
ENCODINGS = ("numeric", "onehot", "stratified")
POLICIES = ("static", "dynamic", "adaptive")

DEFAULTS = dict(
    seed=42,
    target="leaf_area_cm2",
    encoding="onehot",
    policy="adaptive",
    n_eval=300,            # held-out evaluation rows
    n_train=50,            # initial training rows
    n_select=40,           # rows added per round (each arm)
    n_iterations=8,
    mix_ratio=0.5,         # alpha_0
    sel_sigma=0.5,         # kernel width, in units of the median pool distance
    knn_k=15,              # neighbours for the local-error surface
    top_q=0.85,            # eval-error quantile defining the weakspot members
    iters_initial=30, iters_retrain=200,
    hidden=(64, 64), lr_init=1e-3, alpha_l2=1e-4,
    # Real-data analogue of the papers' induced weakspot + scarce pool: rows
    # whose ``deficit_col`` equals ``deficit_val`` are withheld from the
    # initial training set entirely and thinned out of the candidate pool with
    # probability ``deficit_frac`` (0 = no deficit; the evaluation set is
    # untouched, so the deficit shows up as error there).
    deficit_col=None, deficit_val=None, deficit_frac=0.0,
)


def load_dataset(csv_path=DATA_CSV):
    df = pd.read_csv(csv_path)
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return df, cat_cols


def _mlp(cfg, epochs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=tuple(cfg["hidden"]), activation="relu",
            solver="adam", alpha=float(cfg["alpha_l2"]),
            learning_rate_init=float(cfg["lr_init"]), max_iter=int(epochs),
            random_state=42, early_stopping=False, warm_start=True))])


def _continue_fit(m, X, y, epochs):
    est = m.named_steps["model"]
    est.max_iter = int(epochs)
    if hasattr(est, "_no_improvement_count"):     # per-round stopping state
        est._no_improvement_count = 0
        est.best_loss_ = np.inf
    m.named_steps["scaler"].fit(X) if False else None  # scaler frozen: no-op
    est.fit(m.named_steps["scaler"].transform(X), y)
    return m


def prepare(cfg, df, cat_cols):
    """Feature matrices, distance space and split for one seeded run."""
    rng = np.random.RandomState(int(cfg["seed"]))
    target = cfg["target"]
    y = df[target].to_numpy(dtype=float)
    feat = df.drop(columns=[target])
    num_cols = [c for c in feat.columns if c not in cat_cols]

    X_num = feat[num_cols].to_numpy(dtype=float)
    X_cat = pd.get_dummies(feat[cat_cols]).to_numpy(dtype=float) if cat_cols else \
        np.zeros((len(feat), 0))
    cats = (feat[cat_cols].astype(str).agg(" / ".join, axis=1).to_numpy()
            if cat_cols else np.array(["-"] * len(feat)))

    if cfg["encoding"] == "numeric":
        X_model = X_num
    else:                                          # onehot & stratified
        X_model = np.hstack([X_num, X_cat])
    # Distance space: standardised; one-hot dimensions participate only in the
    # onehot encoding, which is exactly the "vector distance over categories".
    X_dist = np.hstack([X_num, X_cat]) if cfg["encoding"] == "onehot" else X_num
    X_dist = StandardScaler().fit_transform(X_dist)

    idx = rng.permutation(len(df))
    ev = idx[:int(cfg["n_eval"])]
    rest = idx[int(cfg["n_eval"]):]
    col, val = cfg.get("deficit_col"), cfg.get("deficit_val")
    frac = float(cfg.get("deficit_frac") or 0.0)
    if col and val and frac > 0:
        in_def = df[col].astype(str).to_numpy()[rest] == str(val)
        # initial training: deficit rows fully withheld
        tr = rest[~in_def][:int(cfg["n_train"])]
        # pool: deficit rows kept only with probability 1 - frac
        keep = ~in_def | (rng.random_sample(len(rest)) >= frac)
        pool = np.setdiff1d(rest[keep], tr, assume_unique=False)
    else:
        tr = rest[:int(cfg["n_train"])]
        pool = rest[int(cfg["n_train"]):]
    med = np.median(np.linalg.norm(
        X_dist[rng.choice(len(df), 400)][:, None, :]
        - X_dist[rng.choice(len(df), 40)][None, :, :], axis=2))
    return dict(X_model=X_model, X_dist=X_dist, y=y, cats=cats,
                ev=ev, tr=tr, pool=pool, med_dist=float(med), rng=rng)


def _detect(D, cfg, err):
    """Local-error peak on the eval set: centre, per-eval kernel, severity,
    and the categorical distribution of the weakspot members."""
    Xe = D["X_dist"][D["ev"]]
    k = min(int(cfg["knn_k"]), len(Xe) - 1)
    d2 = np.linalg.norm(Xe[:, None, :] - Xe[None, :, :], axis=2)
    nn = np.argsort(d2, axis=1)[:, :k]
    local = err[nn].mean(axis=1)
    c_i = int(np.argmax(local))
    center = Xe[c_i]
    sigma = float(cfg["sel_sigma"]) * D["med_dist"]
    w_ev = np.exp(-0.5 * (np.linalg.norm(Xe - center, axis=1) / sigma) ** 2)
    sev = float((w_ev @ err) / max(w_ev.sum(), 1e-9) / max(err.mean(), 1e-12))
    members = local >= np.quantile(local, float(cfg["top_q"]))
    cat_dist = pd.Series(D["cats"][D["ev"]][members]).value_counts(normalize=True)
    return center, sigma, w_ev, sev, cat_dist


def _select_guided(D, cfg, center, sigma, cat_dist, avail, mix, rng):
    """Guided pick from the remaining pool: kernel share + uniform rehearsal,
    with per-category quotas under the stratified encoding."""
    cand = D["pool"][avail]
    n_sel = min(int(cfg["n_select"]), len(cand))
    n_g = int(round(np.clip(mix, 0, 1) * n_sel))
    w = np.exp(-0.5 * (np.linalg.norm(
        D["X_dist"][cand] - center, axis=1) / sigma) ** 2) + 1e-12

    picked: list[int] = []
    if cfg["encoding"] == "stratified" and n_g > 0 and len(cat_dist):
        for cat, frac in cat_dist.items():          # quotas from the weakspot
            want = int(round(frac * n_g))
            sub = np.flatnonzero(D["cats"][cand] == cat)
            sub = [s for s in sub if s not in picked]
            if want and len(sub):
                p = w[sub] / w[sub].sum()
                take = rng.choice(sub, size=min(want, len(sub)),
                                  replace=False, p=p)
                picked.extend(int(t) for t in take)
    short = n_g - len(picked)
    if short > 0:
        rest = np.setdiff1d(np.arange(len(cand)), picked)
        p = w[rest] / w[rest].sum()
        picked.extend(int(t) for t in
                      rng.choice(rest, size=min(short, len(rest)),
                                 replace=False, p=p))
    rest = np.setdiff1d(np.arange(len(cand)), picked)
    n_u = min(n_sel - len(picked), len(rest))
    if n_u > 0:
        picked.extend(int(t) for t in rng.choice(rest, size=n_u, replace=False))
    return cand[np.asarray(picked, dtype=int)]


def run_real(cfg: dict, keep_rounds: bool = False) -> dict:
    """One seeded guided-vs-random run. Returns per-iteration series and the
    per-round categorical distributions (weakspot members and guided picks).

    ``keep_rounds`` additionally stores what a landscape visualisation needs:
    the guided arm's per-eval-point error at every iteration (including the
    initial model), the detected centre per round, the guided picks' positions,
    and the eval set's distance-space coordinates and category labels."""
    c = {**DEFAULTS, **cfg}
    df, cat_cols = load_dataset()
    D = prepare(c, df, cat_cols)
    rng = D["rng"]
    Xm, y = D["X_model"], D["y"]
    Xe, ye = Xm[D["ev"]], y[D["ev"]]

    m0 = _mlp(c, c["iters_initial"])
    m0.fit(Xm[D["tr"]], y[D["tr"]])
    mg, mr = copy.deepcopy(m0), copy.deepcopy(m0)
    # Freeze both scalers on the full feature table so continued fits share one
    # representation (the dataset is known upfront in the fixed-data setting).
    for m in (m0, mg, mr):
        m.named_steps["scaler"].fit(Xm)

    avail_g = np.ones(len(D["pool"]), dtype=bool)
    avail_r = np.ones(len(D["pool"]), dtype=bool)
    out = dict(mae_g=[], mae_r=[], sev=[], mix=[], n_pool=[],
               cat_weak=[], cat_pick=[])
    e0 = np.abs(mg.predict(Xe) - ye)
    out["mae_g"].append(float(e0.mean())); out["mae_r"].append(float(e0.mean()))
    sev0 = None
    if keep_rounds:
        out["rounds"] = dict(
            err_eval=[e0.copy()], centers=[], picks_dist=[],
            X_eval_dist=D["X_dist"][D["ev"]].copy(),
            cats_eval=D["cats"][D["ev"]].copy(),
            med_dist=D["med_dist"])

    for it in range(int(c["n_iterations"])):
        err = np.abs(mg.predict(Xe) - ye)
        center, sigma, _, sev, cat_dist = _detect(D, c, err)
        if sev0 is None:
            sev0 = sev
        if c["policy"] == "dynamic":
            mix = float(c["mix_ratio"]) * (0.1 ** it)
        elif c["policy"] == "adaptive":
            excess = max(sev - 1.0, 0.0) / max(sev0 - 1.0, 1e-9)
            mix = float(c["mix_ratio"]) * float(np.clip(excess, 0.0, 1.0))
        else:
            mix = float(c["mix_ratio"])

        sel_g = _select_guided(D, c, center, sigma, cat_dist, avail_g, mix, rng)
        cand_r = D["pool"][avail_r]
        sel_r = rng.choice(cand_r, size=min(int(c["n_select"]), len(cand_r)),
                           replace=False)
        avail_g[np.isin(D["pool"], sel_g)] = False
        avail_r[np.isin(D["pool"], sel_r)] = False

        if len(sel_g):
            _continue_fit(mg, Xm[sel_g], y[sel_g], c["iters_retrain"])
        if len(sel_r):
            _continue_fit(mr, Xm[sel_r], y[sel_r], c["iters_retrain"])

        out["mae_g"].append(float(np.abs(mg.predict(Xe) - ye).mean()))
        out["mae_r"].append(float(np.abs(mr.predict(Xe) - ye).mean()))
        out["sev"].append(sev); out["mix"].append(mix)
        out["n_pool"].append(int(avail_g.sum()))
        out["cat_weak"].append(cat_dist.to_dict())
        out["cat_pick"].append(
            pd.Series(D["cats"][sel_g]).value_counts(normalize=True).to_dict()
            if len(sel_g) else {})
        if keep_rounds:
            out["rounds"]["err_eval"].append(np.abs(mg.predict(Xe) - ye))
            out["rounds"]["centers"].append(center.copy())
            out["rounds"]["picks_dist"].append(D["X_dist"][sel_g].copy())

    out["gap"] = [g - r for g, r in zip(out["mae_g"], out["mae_r"])]
    out["cfg"] = c
    return out


def summarise(res: dict) -> dict:
    g = np.asarray(res["mae_g"][1:]); r = np.asarray(res["mae_r"][1:])
    return dict(auc_gap=float(np.mean(g - r)), final_gap=float(g[-1] - r[-1]),
                init_mae=res["mae_g"][0], final_mae_g=float(g[-1]),
                final_mae_r=float(r[-1]))
