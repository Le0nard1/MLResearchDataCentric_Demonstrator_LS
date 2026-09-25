"""
Summary of the tuning experiment (budget_v2 --stage tune).

For every (dataset, region, noise, headroom) and method setting: % error reduction vs.
random selection over the whole test set and inside the region, mean over seeds.
Prints (1) the weakspot region arm's advantage over alpha per headroom, (2) the best
alpha per method, chosen on whole-test error pooled over datasets and noise, and the
advantage it reaches, and (3) how often each method beats random over the grid.

    python -m scripts.dataselect.summarise_tune [tune|tune_pilot]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("data/experiment_results/data_selective_training")
COND = ["dataset", "region", "noise", "n_init", "iters_initial"]


def advantages(df):
    df = df[df.arm != "ERROR"].copy()
    df["top_q"] = df.top_q.fillna(-1)
    ref = (df[df.arm == "random"].set_index(COND + ["seed"])[["mae", "err_in"]]
           .rename(columns=lambda c: c + "_rand"))
    d = df[df.arm != "random"].join(ref, on=COND + ["seed"])
    g = d.groupby(COND + ["arm", "alpha", "top_q"])
    out = g[["mae", "mae_rand", "err_in", "err_in_rand"]].mean()
    out["adv"] = 100 * (out.mae_rand - out.mae) / out.mae_rand
    out["adv_in"] = 100 * (out.err_in_rand - out.err_in) / out.err_in_rand
    return out.reset_index()


def main():
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "tune"
    a = advantages(pd.read_csv(RES / f"budget_v2_{stage}.csv"))
    a.to_csv(RES / f"budget_v2_{stage}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    a["headroom"] = a.n_init.astype(str) + "/" + a.iters_initial.astype(str)

    ws = a[(a.arm == "ws_region_knn") & (a.top_q == 0.85)]
    print("=== ws_region_knn (top 15%): whole-test advantage % over alpha, "
          "mean over noise; rows dataset/region/headroom")
    print(ws.pivot_table(index=["dataset", "region", "headroom"], columns="alpha",
                         values="adv").round(1).to_string())

    print("\n=== ws_region_knn: region size (top_q) x alpha, mean over all conditions")
    print(a[a.arm == "ws_region_knn"].pivot_table(index="top_q", columns="alpha",
                                                  values="adv").round(2).to_string())

    key = ["arm", "alpha", "top_q"]
    pooled = a.groupby(key)[["adv", "adv_in"]].mean().reset_index()
    best = pooled.loc[pooled.groupby("arm").adv.idxmax()]
    print("\n=== best setting per method (whole-test advantage pooled over everything)")
    print(best.round(2).to_string(index=False))

    print("\n=== share of grid cells (condition x alpha) with advantage > 0")
    print(a.groupby("arm").adv.apply(lambda s: (s > 0).mean()).round(2).to_string())

    print("\n=== best-alpha advantage per method, by dataset x region x headroom")
    sel = a.merge(best[key], on=key)
    print(sel.pivot_table(index=["dataset", "region", "headroom"], columns="arm",
                          values="adv").round(1).to_string())


if __name__ == "__main__":
    main()
