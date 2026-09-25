"""
Does the advantage of weakspot selection grow with the severity of the weakspot?

Severity of a seed = the initial model's test error inside the weak region divided by its
error outside (budget_v2_main.csv, init_in / init_out). Advantage = % error reduction vs.
random selection over the whole test set. Reports Spearman correlations per method (pooled
and per dataset) and the advantage per severity bin.

    python -m scripts.dataselect.severity_analysis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = Path("data/experiment_results/data_selective_training")
KEYS = ["dataset", "region", "noise", "seed"]
ARMS = ["uniform", "kcenter", "loss", "jtt", "rho", "rho_landscape", "rho_filter",
        "ws_default", "ws_tuned"]
BINS = [0, 1.0, 1.25, 1.5, 2.0, np.inf]


def main():
    df = pd.read_csv(RES / "budget_v2_main.csv")
    df = df[df.arm != "ERROR"]
    ref = df[df.arm == "random"].set_index(KEYS)[["mae"]].rename(columns={"mae": "mae_r"})
    d = df[df.arm.isin(ARMS)].join(ref, on=KEYS)
    d["adv"] = 100 * (d.mae_r - d.mae) / d.mae_r
    d["severity"] = d.init_in / d.init_out
    d["sev_bin"] = pd.cut(d.severity, BINS, right=False)
    d.to_csv(RES / "budget_v2_severity.csv", index=False)

    pd.set_option("display.width", 220)
    print("seeds per severity bin:",
          d[d.arm == "ws_tuned"].sev_bin.value_counts().sort_index().to_dict())
    print("\n=== mean advantage (% vs random) per severity bin (inside/outside error ratio)")
    print(d.pivot_table(index="arm", columns="sev_bin", values="adv", observed=False)
          .reindex(ARMS).round(1).to_string())

    rows = []
    for arm, g in d.groupby("arm"):
        r = dict(arm=arm, rho_all=spearmanr(g.severity, g.adv)[0],
                 p_all=spearmanr(g.severity, g.adv)[1])
        for ds, h in g.groupby("dataset"):
            r[f"rho_{ds[:5]}"] = spearmanr(h.severity, h.adv)[0]
        rows.append(r)
    print("\n=== Spearman correlation of advantage with severity (per seed)")
    print(pd.DataFrame(rows).set_index("arm").reindex(ARMS).round(3).to_string())

    # Margin of tuned weakspot selection over the best competitor, per bin.
    w = d.pivot_table(index=KEYS, columns="arm", values="adv")
    w["sev"] = d.drop_duplicates(KEYS).set_index(KEYS).severity
    comp = ["uniform", "kcenter", "loss", "jtt", "rho"]
    w["margin_best"] = w.ws_tuned - w[comp].mean(axis=0).pipe(lambda m: w[m.idxmax()])
    w["margin_rho"] = w.ws_tuned - w.rho
    w["margin_uniform"] = w.ws_tuned - w.uniform
    w["margin_kcenter"] = w.ws_tuned - w.kcenter
    w["bin"] = pd.cut(w.sev, BINS, right=False)
    print("\n=== tuned weakspot minus competitor (percentage points), per severity bin")
    print(w.groupby("bin", observed=False)[["margin_rho", "margin_uniform", "margin_kcenter"]]
          .mean().round(2).to_string())
    print("\n=== severity distribution per dataset x region (median ratio)")
    print(d[d.arm == "ws_tuned"].pivot_table(index="dataset", columns="region",
                                            values="severity", aggfunc="median")
          .round(2).to_string())


if __name__ == "__main__":
    main()
