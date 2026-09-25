"""
Benchmark extension (Section 4.4): the methods of Table 1 across all real datasets.

Combines budget_v2_bench.csv (diamonds, sulfur, Brazilian houses, nyc-taxi; seeds
7000-7009) with budget_v2_main.csv and budget_v2_main_a1.csv (synthetic task, California
housing, medical charges; seeds 7000-7049). Per dataset: error reduction vs. random
selection in % over the whole test set, averaged over the eight conditions (four region
conditions x clean/outlier labels; per condition 100 * mean paired difference / mean random
error). Across the six real datasets: average rank (1 = best), number of datasets won, and
a Friedman test. Paired margins over all seed-level pairs of the four new datasets.

    python -m scripts.dataselect.summarise_bench
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RES = Path("data/experiment_results/data_selective_training")
KEYS = ["dataset", "region", "noise", "seed"]
ARMS = ["uniform", "kcenter_a1", "rho", "rho_a1", "ws_default", "ws_tuned",
        "rho_landscape_a1", "rho_filter_a1"]
LABEL = {"uniform": "Uniform", "kcenter_a1": "k-center ($\\alpha=1$)",
         "rho": "RHO-LOSS", "rho_a1": "RHO-LOSS ($\\alpha=1$)",
         "ws_default": "Weakspot, default", "ws_tuned": "Weakspot, tuned",
         "rho_landscape_a1": "RHO-landscape ($\\alpha=1$)",
         "rho_filter_a1": "RHO-filter ($\\alpha=1$)", "all_data": "All data"}
ORDER = ["synth2d", "houses", "medical_charges", "brazilian_houses", "diamonds", "sulfur",
         "nyc_taxi"]
REAL = ORDER[1:]


def load():
    m = pd.read_csv(RES / "budget_v2_main.csv")
    a = pd.read_csv(RES / "budget_v2_main_a1.csv")
    b = pd.read_csv(RES / "budget_v2_bench.csv")
    d = pd.concat([m, a[a.arm != "uniform"], b], ignore_index=True)
    return d[d.arm != "ERROR"]


def per_condition(d):
    rows = []
    for key, g in d.groupby(["dataset", "region", "noise"]):
        ref = g[g.arm == "random"].set_index("seed")
        for arm in ARMS + ["all_data"]:
            x = g[g.arm == arm].set_index("seed")
            if x.empty:
                continue
            x = x.loc[ref.index]
            rows.append(dict(zip(["dataset", "region", "noise"], key), arm=arm,
                             adv=100 * (ref.mae - x.mae).mean() / ref.mae.mean(),
                             adv_in=100 * (ref.err_in - x.err_in).mean() / ref.err_in.mean()))
    return pd.DataFrame(rows)


def main():
    d = load()
    print("errors in bench:", int((pd.read_csv(RES / "budget_v2_bench.csv").arm == "ERROR").sum()))
    c = per_condition(d)
    t = c.pivot_table(index="arm", columns="dataset", values="adv").reindex(ARMS + ["all_data"])
    t = t[[x for x in ORDER if x in t.columns]]
    pd.set_option("display.width", 220)
    print("\n=== % vs random, whole test set, mean over the 8 conditions")
    print(t.round(1).to_string())

    real = t.loc[ARMS, [x for x in REAL if x in t.columns]]
    ranks = real.rank(ascending=False)
    summary = pd.DataFrame({"mean_real": real.mean(axis=1), "avg_rank": ranks.mean(axis=1),
                            "wins": (ranks == 1).sum(axis=1),
                            "synthetic": t.loc[ARMS, "synth2d"]})
    print("\n=== across the real datasets (rank 1 = best among the eight methods)")
    print(summary.round(2).to_string())
    print("Friedman over real datasets: p = %.3g" % friedmanchisquare(*[real.loc[a] for a in ARMS]).pvalue)

    new = [x for x in ["brazilian_houses", "diamonds", "sulfur", "nyc_taxi"] if x in t.columns]
    ref = d[d.arm == "random"].set_index(KEYS).mae.rename("r")
    e = d[d.dataset.isin(new) & (d.arm != "random")].join(ref, on=KEYS)
    e["adv"] = 100 * (e.r - e.mae) / e.r
    w = e.pivot_table(index=KEYS, columns="arm", values="adv").reset_index()
    print("\n=== paired margins over the four new datasets (pp)")
    for a_, b_ in [("ws_tuned", "rho"), ("ws_tuned", "kcenter_a1"), ("ws_tuned", "uniform"),
                   ("rho_filter_a1", "rho_a1"), ("rho_filter_a1", "rho"),
                   ("rho_landscape_a1", "rho_a1"), ("ws_tuned", "ws_default")]:
        x = w[a_] - w[b_]
        per = w.assign(x=x).groupby("dataset").x.mean().round(2).to_dict()
        print(f"{a_} - {b_}: {x.mean():+.2f} (p={wilcoxon(x).pvalue:.1e}) {per}")

    print("\n=== LaTeX rows (datasets in columns)")
    for arm in ARMS + ["all_data"]:
        vals = " & ".join(f"${v:.1f}$" for v in t.loc[arm])
        extra = (f" & ${summary.loc[arm, 'avg_rank']:.1f}$" if arm in summary.index else " & --")
        print(f"{LABEL[arm]:<28}& {vals}{extra} \\\\")
    t.to_csv(RES / "bench_summary.csv")


if __name__ == "__main__":
    main()
