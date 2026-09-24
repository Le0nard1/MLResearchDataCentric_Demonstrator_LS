"""
Significance gate for the Section 4.3 baseline regimes (well-trained model).

Recomputes, for every (condition, seed) of baselines_v2*.csv, exactly the initial
model and pool diagnosis of ``baselines.run_one`` and adds a permutation p-value
for the weakspot (``fixed_data.diagnose``: peak of the kNN error map against the
peaks after permuting the pool errors). The gated arm uses the weakspot selection
when p < 0.05 and the random baseline otherwise, so it needs no further retraining.

    python -m scripts.dataselect.gate --workers 12
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.dataselect import pipeline as P
from scripts.dataselect.baselines import FIXED
from scripts.dataselect.fixed_data import diagnose, effect_ratio, FIXED as FD

RES = "data/experiment_results/data_selective_training/"
CONDS = [(400, 0.0), (400, 0.2)]            # the Table 2 well-trained regimes
N_SELECT = 100
SEEDS = list(range(1000, 1050))


def p_value(n_train, radius, seed):
    """Replicates baselines.run_one up to the diagnosis (same RNG sequence)."""
    rng = np.random.RandomState(seed)
    nb, noise = FIXED["n_bumps"], FIXED["noise_std"]
    centre = np.asarray(FIXED["center"])
    X_all = P.sample_inputs(FIXED["n_pool_total"], rng)
    y_all = P.label(X_all, rng, n_bumps=nb, noise_std=noise)
    X_keep, y_keep = ((P.induce_weakspot(X_all, y_all, centre, radius)[:2])
                      if radius > 0 else (X_all, y_all))
    tr = rng.choice(len(X_keep), size=min(n_train, len(X_keep)), replace=False)
    X_tr, y_tr = X_keep[tr], y_keep[tr]
    X_eval = P.sample_inputs(FIXED["n_eval"], rng)
    X_cand = P.sample_inputs(FIXED["n_candidate"], rng)
    y_cand = P.label(X_cand, rng, n_bumps=nb, noise_std=noise)
    m0 = build_model(AVAILABLE_MODELS[FIXED["model_name"]], complexity=FIXED["complexity"],
                     iterations=FIXED["iters_initial"], warm_start=True,
                     early_stopping=True)
    m0.fit(X_tr, y_tr)
    err = np.abs(y_cand - m0.predict(X_cand))
    FD.update(grid_res=FIXED["grid_res"], extract_q=FIXED["extract_q"],
              detector=FIXED["detector"])
    c_hat, ext, p = diagnose(X_cand, err, np.random.RandomState(seed + 7))
    ratio = effect_ratio(X_cand, err, ext)
    init = float(np.abs(P.true_function(X_eval, nb) - m0.predict(X_eval)).mean())
    return dict(n_train=n_train, radius=radius, seed=seed, p_gate=p, ratio=ratio, init_check=init,
                gate_cx=c_hat[0], gate_cy=c_hat[1])


def summary(csv, gates):
    df = pd.read_csv(RES + csv)
    df = df[(df.n_select == N_SELECT) & df.n_train.isin([c[0] for c in CONDS])]
    key = ["n_train", "radius", "seed"]
    rnd = df[df.arm == "random"].groupby(key)[["mae", "err_in"]].mean()
    out = []
    for alpha in (0.2, 1.0):
        ws = df[(df.arm == "weakspot") & (df.alpha == alpha)].set_index(key)[["mae", "err_in"]]
        j = ws.join(rnd, rsuffix="_r").join(gates.set_index(key))
        ok = (j.p_gate < 0.05) & (j.ratio >= 2)
        j["mae_g"] = np.where(ok, j.mae, j.mae_r)
        j["in_g"] = np.where(ok, j.err_in, j.err_in_r)
        for (n, r), g in j.groupby(level=[0, 1]):
            row = dict(file=csv, n_train=n, radius=r, alpha=alpha,
                       gate_pass=float(((g.p_gate < 0.05) & (g.ratio >= 2)).mean()),
                       ratio=float(g.ratio.mean()))
            for tag, col, ref in (("ungated", "mae", "mae_r"), ("gated", "mae_g", "mae_r"),
                                  ("ungated_in", "err_in", "err_in_r"),
                                  ("gated_in", "in_g", "err_in_r")):
                if g[col].isna().all():
                    continue
                d = g[ref] - g[col]
                row[tag] = 100 * d.mean() / g[ref].mean()
                nz = d[d != 0]
                row[tag + "_p"] = (stats.wilcoxon(nz).pvalue if len(nz) > 5 else np.nan)
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    gates = pd.DataFrame(Parallel(n_jobs=a.workers)(
        delayed(p_value)(n, r, s) for n, r in CONDS for s in SEEDS))
    gates.to_csv(RES + "gate_v2.csv", index=False)
    tab = pd.concat([summary(f, gates) for f in ("baselines_v2.csv", "baselines_v2_cum.csv")])
    tab.to_csv(RES + "gate_v2_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
