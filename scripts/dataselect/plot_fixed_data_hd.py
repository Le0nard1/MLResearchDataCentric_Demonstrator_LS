"""
Summary and figure for fixed-data reweighting beyond two dimensions
(scripts.dataselect.fixed_data_hd, ``--stage frontier``).

Per (task, condition, arm, setting) the mean change against uniform continuation on
the same seed, in % of the uniform error (positive = better), over the whole area,
inside and outside the weakspot, with the Wilcoxon p of the inside change and the
mean ESS of the weights. Descriptive over fixed grids, as the 2-D trade-off figure.

    python -m scripts.dataselect.plot_fixed_data_hd
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("data/experiment_results/data_selective_training")


def table(df):
    df = df[df.arm != "ERROR"].copy()
    key = ["task", "region", "noise", "seed"]
    uni = df[df.arm == "uniform"].set_index(key)[["mae", "err_in", "err_out"]]
    j = df[df.arm != "uniform"].set_index(key).join(uni, rsuffix="_u").reset_index()
    rows = []
    for (t, r, n, arm, hp), a in j.groupby(["task", "region", "noise", "arm", "hp"]):
        d, din = a.mae_u - a.mae, a.err_in_u - a.err_in
        rows.append(dict(task=t, region=r, noise=n, arm=arm, hp=hp, dim=a.dim.iloc[0],
                         whole=100 * d.mean() / a.mae_u.mean(),
                         p=stats.wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 5 else 1.0,
                         inside=100 * din.mean() / a.err_in_u.mean(),
                         p_in=(stats.wilcoxon(din[din != 0]).pvalue
                               if (din != 0).sum() > 5 else 1.0),
                         outside=100 * (a.err_out_u - a.err_out).mean() / a.err_out_u.mean(),
                         ess=a.ess.mean(), detect_dist=a.detect_dist.mean(),
                         detect_dist_emb=a.detect_dist_emb.mean(),
                         init_mae=a.init_mae.mean(), const_mae=a.const_mae.mean(),
                         init_in=a.init_in.mean(), init_out=a.init_out.mean(),
                         uni_mae=a.mae_u.mean(), n=len(a)))
    return pd.DataFrame(rows)


STYLE = {"weakspot": ("Weakspot, input space", "#1f77b4", "o"),
         "weakspot_emb": ("Weakspot, learnt embedding", "#17becf", "D"),
         "oracle": ("Oracle centre", "#000000", "*"),
         "random_centre": ("Random centre, input", "#bdbdbd", "x"),
         "random_emb": ("Random centre, embedding", "#636363", "+"),
         "smoothed_loss": ("Smoothed loss", "#2ca02c", "v"),
         "loss": ("Per-point loss", "#8c564b", "s"),
         "jtt": ("JTT", "#d62728", "^")}
PANELS = [("synth2", "$d=2$"), ("synth5", "$d=5$"), ("synth10", "$d=10$"),
          ("synth20", "$d=20$"), ("cal8", "California, 8 features")]


def fig(tab, out, noise="gauss"):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(19, 4.3), constrained_layout=True)
    for ax, (task, title) in zip(axes, PANELS):
        nz = "real" if task == "cal8" else noise
        g = tab[(tab.task == task) & (tab.noise == nz)]
        for arm, (lab, col, mk) in STYLE.items():
            a = g[(g.arm == arm) & (g.outside > -60)]
            if len(a):
                ax.scatter(-a.outside, a.inside, c=col, marker=mk, s=30 if mk != "*" else 70,
                           label=lab, alpha=0.9)
        ax.axhline(0, c="k", lw=0.6)
        ax.axvline(0, c="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("error added outside (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("weakspot error removed (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), fontsize=9, frameon=False)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(RES / "fixed_data_hd_frontier.csv")
    tab = table(df)
    tab.to_csv(RES / "fixed_data_hd_summary.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["task", "noise", "arm", "hp", "whole", "inside", "p_in", "outside", "ess",
            "detect_dist", "detect_dist_emb", "init_mae", "const_mae", "init_in", "init_out"]
    print(tab[cols].round(3).to_string())
    (RES / "figures").mkdir(exist_ok=True)
    fig(tab, RES / "figures" / "fig_fixed_data_hd.png")
    fig(tab, RES / "figures" / "fig_fixed_data_hd_outliers.png", noise="outlier")


if __name__ == "__main__":
    main()
