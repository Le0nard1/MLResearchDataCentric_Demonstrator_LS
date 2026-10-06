"""Sampling of sparse areas by iterated k-center (Section 5). k-center is model-free, so its
eight rounds can be replayed exactly. Run from the code root:
    python -m scripts.iterative.analysis_kcenter_sparsity
 Measures, per dataset (seeds 8200-8204, repairable condition):
  sparse10 : share of selected points among the 10% sparsest reserve points (10-NN distance)
  tail     : share of selected points with any input outside the reserve's 1-99% range
  test_nn  : mean distance of selected points to their nearest test point, / random
  dim      : input dimension
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.dataselect import budget_v2 as B
from scripts.dataselect.baselines import pick_kcenter

rows = []
for name in ["synth2d", "brazilian_houses", "diamonds", "houses", "medical_charges",
             "nyc_taxi", "sulfur"]:
    for seed in range(8200, 8205):
        B.FIXED.update(n_data=4500, rho=0.25)
        X, y, I, R, X_ev, y_ev, region_fn, to_U, _ = B.make(name, "sparse_init", "clean", seed)
        U_I, U_R = to_U(X[I]), to_U(X[R])
        U_ev = to_U(X_ev)
        dk, _ = cKDTree(U_R).query(U_R, k=11)
        sparse = dk[:, -1] >= np.quantile(dk[:, -1], 0.9)
        lo, hi = np.quantile(U_R, 0.01, 0), np.quantile(U_R, 0.99, 0)
        tail = ((U_R < lo) | (U_R > hi)).any(1)
        tree_ev = cKDTree(U_ev)
        rs = np.random.RandomState(seed)
        for arm in ["kcenter", "random"]:
            avail = np.ones(len(R), bool)
            chosen = []
            for t in range(8):
                av = np.flatnonzero(avail)
                if arm == "kcenter":
                    used = np.vstack([U_I, U_R[chosen]]) if chosen else U_I
                    pick = pick_kcenter(U_R[av], used, 200)
                else:
                    pick = rs.choice(len(av), 200, replace=False)
                sel = av[pick]
                chosen += list(sel)
                avail[sel] = False
            c = np.asarray(chosen)
            rows.append(dict(dataset=name, seed=seed, arm=arm, dim=U_R.shape[1],
                             sparse10=sparse[c].mean(), tail=tail[c].mean(),
                             test_nn=tree_ev.query(U_R[c])[0].mean(),
                             y_abs=np.abs(y[R][c] - np.median(y[R])).mean()
                             / np.abs(y[R] - np.median(y[R])).mean()))
        print(name, seed, flush=True)

d = pd.DataFrame(rows)
m = d.groupby(["dataset", "arm"])[["dim", "sparse10", "tail", "test_nn", "y_abs"]].mean()
print(m.round(3).to_string())
