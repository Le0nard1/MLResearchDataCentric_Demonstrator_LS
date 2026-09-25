"""
Summary of the baseline comparison (budget_v2 --stage main, paper Section 4.4).

Per (dataset, region, noise) and method: error reduction vs. random selection in %,
whole test set and inside the region, paired Wilcoxon over seeds, Holm-corrected within
each condition and metric over the methods. Also the per-method average over all
conditions and the count of significant wins and losses.

    python -m scripts.dataselect.summarise_main
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.dataselect.summarise_budget import holm

RES = Path("data/experiment_results/data_selective_training")
KEYS = ["dataset", "region", "noise"]
ORDER = ["uniform", "kcenter", "loss", "jtt", "rho", "rho_landscape", "rho_filter",
         "ws_default", "ws_tuned", "all_data"]


def main():
    df = pd.read_csv(RES / "budget_v2_main.csv")
    print("errors:", int((df.arm == "ERROR").sum()))
    df = df[df.arm != "ERROR"]
    rows = []
    for key, g in df.groupby(KEYS):
        ref = g[g.arm == "random"].set_index("seed")
        block = []
        for arm, a in g[g.arm != "random"].groupby("arm"):
            a = a.set_index("seed").loc[ref.index]
            r = dict(zip(KEYS, key), arm=arm, n=len(ref))
            for m in ("mae", "err_in"):
                d = ref[m] - a[m]
                r[f"{m}_red"] = float(100 * d.mean() / ref[m].mean())
                r[f"{m}_p"] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            block.append(r)
        for m in ("mae", "err_in"):
            for r, q in zip(block, holm([r[f"{m}_p"] for r in block])):
                r[f"{m}_ph"] = float(q)
        rows += block
    s = pd.DataFrame(rows)
    s.to_csv(RES / "budget_v2_main_summary.csv", index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    agg = s.groupby("arm").agg(
        whole=("mae_red", "mean"), inside=("err_in_red", "mean"),
        sig_win=("mae_ph", lambda p: int(((p < 0.05) & (s.loc[p.index, "mae_red"] > 0)).sum())),
        sig_loss=("mae_ph", lambda p: int(((p < 0.05) & (s.loc[p.index, "mae_red"] < 0)).sum())),
        worst=("mae_red", "min")).reindex(ORDER)
    print("=== average over all 24 conditions (% vs random)")
    print(agg.round(2).to_string())

    star = lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else "")
    s["cell"] = [f"{w:+.1f}{star(pw)} ({i:+.1f}{star(pi)})" for w, pw, i, pi in
                 zip(s.mae_red, s.mae_ph, s.err_in_red, s.err_in_ph)]
    for (ds,), g in s.groupby(["dataset"]):
        print(f"\n=== {ds}: % vs random, whole (inside region), Holm per condition")
        print(g.pivot_table(index="arm", columns=["region", "noise"], values="cell",
                            aggfunc="first").reindex(ORDER).to_string())


if __name__ == "__main__":
    main()
