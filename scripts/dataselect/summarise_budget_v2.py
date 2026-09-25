"""
Summary of the budget_v2 pilot: % error reduction vs. random selection per
(dataset, region, noise) and arm, whole test set and inside the region, mean over
seeds, with the uncorrected paired Wilcoxon p-value (a pilot, read as direction only).

    python -m scripts.dataselect.summarise_budget_v2 [--stage pilot]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path("data/experiment_results/data_selective_training")
KEYS = ["dataset", "region", "noise"]


def summarise(df):
    rows = []
    for key, g in df.groupby(KEYS):
        ref = g[g.arm == "random"].set_index("seed")
        for arm, a in g[g.arm != "random"].groupby("arm"):
            a = a.set_index("seed").loc[ref.index]
            r = dict(zip(KEYS, key), arm=arm, n_in_region=a.n_in_region.mean())
            for m in ("mae", "err_in"):
                d = ref[m] - a[m]
                r[f"{m}_red"] = 100 * d.mean() / ref[m].mean()
                r[f"{m}_p"] = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    a = ap.parse_args()
    df = pd.read_csv(RES / f"budget_v2_{a.stage}.csv")
    print("errors:", int((df.arm == "ERROR").sum()))
    df = df[df.arm != "ERROR"]
    diag = (df.drop_duplicates(KEYS + ["seed"])
            .assign(ratio=lambda d: d.init_in / d.init_out)
            .groupby(KEYS)[["ratio", "region_share_R", "prec_knn", "prec_evtgpr"]]
            .mean())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(diag.round(3).to_string())
    s = summarise(df)
    s.to_csv(RES / f"budget_v2_{a.stage}_summary.csv", index=False)
    star = lambda p: "*" if p < 0.05 else ""
    s["cell"] = [f"{w:+.1f}{star(pw)} ({i:+.1f}{star(pi)})" for w, pw, i, pi in
                 zip(s.mae_red, s.mae_p, s.err_in_red, s.err_in_p)]
    for (ds,), g in s.groupby(["dataset"]):
        print(f"\n=== {ds}: % vs random, whole (inside region); * p<0.05 uncorrected")
        print(g.pivot_table(index="arm", columns=["region", "noise"], values="cell",
                            aggfunc="first").to_string())


if __name__ == "__main__":
    main()
