"""
Fairness check: the competitors at the tuned method's guidance fraction (alpha = 1).

Merges budget_v2_main.csv (competitors at alpha = 0.2, our methods) with
budget_v2_main_a1.csv (competitors at alpha = 1, same seeds; runs are deterministic, and
the recomputed uniform arm must match). For each competitor the better of alpha = 0.2
and alpha = 1 is chosen once, on the pooled whole-test advantage over all 24 conditions;
since that choice is made on the comparison seeds it favours the competitors, which makes
the comparison conservative for our method. Reports the advantage over random per
dataset and label condition, and the paired margin of tuned weakspot selection over each
competitor (Wilcoxon over all seed-level pairs).

    python -m scripts.dataselect.summarise_a1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path("data/experiment_results/data_selective_training")
KEYS = ["dataset", "region", "noise", "seed"]
COMP = ["kcenter", "loss", "jtt", "rho", "rho_landscape", "rho_filter"]


def main():
    m = pd.read_csv(RES / "budget_v2_main.csv")
    a = pd.read_csv(RES / "budget_v2_main_a1.csv")
    print("errors:", int((a.arm == "ERROR").sum()))
    a = a[a.arm != "ERROR"]
    chk = (a[a.arm == "uniform"].set_index(KEYS).mae
           - m[m.arm == "uniform"].set_index(KEYS).mae).abs().max()
    print(f"pairing check, max |uniform difference| = {chk:.2e}")
    d = pd.concat([m[m.arm != "ERROR"], a[a.arm != "uniform"]], ignore_index=True)
    ref = d[d.arm == "random"].set_index(KEYS).mae.rename("mae_r")
    d = d[d.arm != "random"].join(ref, on=KEYS)
    d["adv"] = 100 * (d.mae_r - d.mae) / d.mae_r
    w = d.pivot_table(index=KEYS, columns="arm", values="adv")

    best = {}
    for c in COMP:
        best[c] = c if w[c].mean() >= w[c + "_a1"].mean() else c + "_a1"
    print("\nchosen setting per competitor (pooled):",
          {c: ("alpha=1" if b.endswith("_a1") else "alpha=0.2") for c, b in best.items()})

    cols = ["uniform"] + [best[c] for c in COMP] + ["ws_default", "ws_tuned"]
    wr = w.reset_index()
    tab = wr.groupby(["dataset", "noise"])[cols].mean().T
    tab["all"] = w[cols].mean()
    pd.set_option("display.width", 220)
    print("\n=== % vs random, averaged over region conditions (competitors at their better alpha)")
    print(tab.round(1).to_string())
    print("\n=== both alphas per competitor, all conditions")
    print(pd.DataFrame({c: [w[c].mean(), w[c + "_a1"].mean()] for c in COMP},
                       index=["alpha=0.2", "alpha=1"]).round(2).to_string())

    print("\n=== tuned weakspot minus competitor (pp), paired over all seeds")
    for c in ["uniform"] + [best[x] for x in COMP]:
        diff = w.ws_tuned - w[c]
        per = wr.assign(diff=(wr.ws_tuned - wr[c])).groupby("dataset")["diff"].mean().round(2)
        print(f"{c:18s} mean {diff.mean():+.2f}  p={wilcoxon(diff).pvalue:.1e}  "
              f"by dataset {per.to_dict()}")
    tab.to_csv(RES / "budget_v2_a1_summary.csv")


if __name__ == "__main__":
    main()
