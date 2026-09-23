"""
Summary statistics and figure for the baseline comparison (scripts.dataselect.baselines).

Per (condition, seed) the random arm is the mean of its independent draws; every
other arm is scored by the paired advantage  Δ = MAE_random − MAE_arm  (positive =
better than random). Per condition: mean Δ with a 95% t-interval over seeds, the
Wilcoxon signed-rank p-value, and Holm correction across all arms of the condition.

    python -m scripts.dataselect.plot_baselines
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

RESULTS = Path("data/experiment_results/data_selective_training")
CSV = RESULTS / "baselines_v2.csv"
FIG = RESULTS / "figures" / "fig_baselines.png"

COND = ["n_train", "radius", "n_select"]
ARMS = {  # (arm, alpha) -> label, colour, linestyle
    ("weakspot", 0.2): ("Weakspot (ours)", "#1f77b4", "-"),
    ("kcenter", 0.2): ("k-center", "#ff7f0e", "-"),
    ("density", 0.2): ("Density", "#9467bd", "-"),
    ("loss", 0.2): ("Per-point loss", "#8c564b", "-"),
    ("stratified", 0.0): ("Stratified coverage", "#7f7f7f", "--"),
    ("kcenter", 1.0): ("k-center (pure)", "#ff7f0e", ":"),
}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["arm"] != "ERROR"].copy()
    rnd = (df[df["arm"] == "random"].groupby(COND + ["seed"])
           .agg(r_mae=("mae", "mean"), r_in=("err_in", "mean"), r_out=("err_out", "mean")))
    oth = df[df["arm"] != "random"].set_index(COND + ["seed"]).join(rnd)
    oth["adv"] = oth["r_mae"] - oth["mae"]
    oth["adv_rel"] = oth["adv"] / oth["r_mae"]
    oth["adv_in"] = oth["r_in"] - oth["err_in"]
    rows = []
    for (cond, g) in oth.reset_index().groupby(COND + ["arm", "alpha"]):
        a = g["adv"].to_numpy()
        n = len(a)
        half = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
        p = stats.wilcoxon(a).pvalue if np.any(a != 0) else 1.0
        rows.append(dict(zip(COND + ["arm", "alpha"], cond), n=n, adv=a.mean(), ci=half,
                         rel=g["adv_rel"].mean(), adv_in=g["adv_in"].mean(),
                         win=(a > 0).mean(), p=p, r_mae=g["r_mae"].mean(),
                         n_in_gap=g["n_in_gap"].mean()))
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby(COND)["p"].transform(holm)
    return t


def plot(t: pd.DataFrame, n_select: int = 100):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, radius, title in zip(axes, [0.2, 0.0],
                                 ["(a) Induced gap (radius 0.2)",
                                  "(b) No induced gap"]):
        sub = t[(t["radius"] == radius) & (t["n_select"] == n_select)]
        for (arm, alpha), (lab, col, ls) in ARMS.items():
            s = sub[(sub["arm"] == arm) & (sub["alpha"] == alpha)].sort_values("n_train")
            if s.empty:
                continue
            ax.errorbar(s["n_train"], 100 * s["adv"] / s["r_mae"],
                        yerr=100 * s["ci"] / s["r_mae"], color=col, ls=ls, marker="o",
                        ms=4, capsize=2, lw=1.4, label=lab)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks([100, 200, 400, 800])
        ax.set_xticklabels(["100", "200", "400", "800"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Initial training-set size")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Error reduction vs. random (%)")
    axes[1].legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return FIG


def main():
    t = paired_table(pd.read_csv(CSV))
    t.to_csv(RESULTS / "baselines_v2_summary.csv", index=False)
    pd.set_option("display.width", 200)
    show = t.copy()
    show["rel%"] = (100 * show["adv"] / show["r_mae"]).round(1)
    for c in ["adv", "ci", "adv_in", "win", "p", "p_holm", "r_mae"]:
        show[c] = show[c].round(3)
    print(show[COND + ["arm", "alpha", "adv", "ci", "rel%", "adv_in", "win", "p_holm",
                       "n_in_gap"]].to_string(index=False))
    print(plot(t))


if __name__ == "__main__":
    main()
