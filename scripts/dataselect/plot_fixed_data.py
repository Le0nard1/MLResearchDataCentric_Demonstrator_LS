"""
Summary table and figures for the fixed-data experiment (scripts.dataselect.fixed_data).

Per (condition, seed) every arm is compared with uniform continuation on the same
data at equal compute:  rel = (MAE_uniform - MAE_arm) / MAE_uniform  (positive = the
arm is better). Mean over seeds, Wilcoxon signed-rank p, Holm across the arms of a
condition. "gated" = weakspot when the permutation gate passes (p < 0.05), else uniform.
For California the "inside" region is the disc of radius 0.2 around each seed's
detected centre (no ground-truth weakspot exists).

    python -m scripts.dataselect.plot_fixed_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.dataselect.plot_baselines import holm

RES = Path("data/experiment_results/data_selective_training")
ARMS = ["weakspot", "gated", "loss", "jtt", "density"]
R_CAL = 0.2


def add_region_cal(df, maps):
    """In/out-of-detected-region errors for California from the stored test errors."""
    cal = df[df.task == "california"].copy()
    if cal.empty or "california__X_ev" not in maps:
        return df
    Xs = maps["california__X_ev"]
    seeds = sorted(cal.seed.unique())
    for i, s in enumerate(seeds):
        row0 = cal[cal.seed == s].iloc[0]
        inside = np.hypot(Xs[i][:, 0] - row0.cx, Xs[i][:, 1] - row0.cy) <= R_CAL
        m = (df.task == "california") & (df.seed == s)
        df.loc[m, "init_in"] = maps["california__initial_ev"][i][inside].mean()
        for arm in df.loc[m, "arm"].unique():
            e = maps[f"california__{arm}_ev"][i]
            mm = m & (df.arm == arm)
            df.loc[mm, "err_in"] = e[inside].mean()
            df.loc[mm, "err_out"] = e[~inside].mean()
    return df


def table(df):
    df = df[df.arm != "ERROR"].copy()
    key = ["task", "region", "noise", "seed"]
    uni = df[df.arm == "uniform"].set_index(key)[["mae", "err_in", "err_out"]]
    ws = df[df.arm == "weakspot"].set_index(key)
    gated = ws.copy()
    fail = gated.p_gate >= 0.05
    gated.loc[fail, ["mae", "err_in", "err_out"]] = uni.loc[gated.index[fail]].values
    gated["arm"] = "gated"
    oth = pd.concat([df[df.arm != "uniform"].set_index(key), gated])
    j = oth.join(uni, rsuffix="_u")
    rows = []
    for (t, r, n), g in j.reset_index().groupby(["task", "region", "noise"]):
        sub = []
        for arm in ARMS:
            a = g[g.arm == arm]
            if a.empty:
                continue
            d = a.mae_u - a.mae
            din = a.err_in_u - a.err_in
            nz = d[d != 0]
            sub.append(dict(task=t, region=r, noise=n, arm=arm,
                            whole=100 * d.mean() / a.mae_u.mean(),
                            p=stats.wilcoxon(nz).pvalue if len(nz) > 5 else 1.0,
                            inside=100 * din.mean() / a.err_in_u.mean(),
                            p_in=(stats.wilcoxon(din[din != 0]).pvalue
                                  if (din != 0).sum() > 5 else 1.0),
                            outside=100 * (a.err_out_u - a.err_out).mean() / a.err_out_u.mean(),
                            gate_pass=float((a.p_gate < 0.05).mean()),
                            detect_dist=a.detect_dist.mean(),
                            uni_mae=a.mae_u.mean(), init_mae=a.init_mae.mean(),
                            uni_in=a.err_in_u.mean(), init_in=a.init_in.mean(),
                            hp=a.hp.iloc[0], n=len(a)))
        s = pd.DataFrame(sub)
        s["p_holm"] = holm(s.p.to_numpy())
        s["p_in_holm"] = holm(s.p_in.to_numpy())
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def fig_maps(maps, out):
    """Reference condition: initial error and error change, weakspot vs uniform."""
    k = "synthetic__"
    init, ws, uni = (maps[k + "initial"].mean(0), maps[k + "weakspot"].mean(0),
                     maps[k + "uniform"].mean(0))
    res = int(np.sqrt(init.size))
    ext = (0, 1, 0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.9), constrained_layout=True)
    im0 = ax[0].imshow(init.reshape(res, res), origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im0, ax=ax[0], shrink=0.85, label=r"absolute error $|\hat f - f|$")
    ax[0].set_title("(a) Initial model")
    v = np.abs(np.concatenate([ws - init, uni - init])).max()
    for i, (d, t) in enumerate([(uni - init, "(b) Uniform $-$ initial"),
                                (ws - init, "(c) Weakspot-weighted $-$ initial")]):
        im = ax[i + 1].imshow(d.reshape(res, res), origin="lower", extent=ext,
                              cmap="RdBu_r", vmin=-v, vmax=v)
        ax[i + 1].set_title(t)
    fig.colorbar(im, ax=ax[1:3], shrink=0.85, label="change in absolute error")
    d = ws - uni
    v2 = np.abs(d).max()
    im3 = ax[3].imshow(d.reshape(res, res), origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=-v2, vmax=v2)
    ax[3].set_title("(d) Weakspot-weighted $-$ uniform")
    fig.colorbar(im3, ax=ax[3], shrink=0.85)
    for a in ax:
        a.add_patch(plt.Circle((0.25, 0.75), 0.2, fill=False, ls="--", lw=1.5, color="k"))
        a.set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_california(maps, df, out):
    """Per-district error change (weakspot - uniform), pooled over seeds, hexbin."""
    X = np.concatenate(maps["california__X_ev"])
    d = np.concatenate(maps["california__weakspot_ev"] - maps["california__uniform_ev"])
    i0 = np.concatenate(maps["california__initial_ev"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    h0 = ax[0].hexbin(X[:, 0], X[:, 1], C=i0, gridsize=35, cmap="viridis",
                      reduce_C_function=np.mean)
    fig.colorbar(h0, ax=ax[0], label="initial absolute error")
    v = np.nanpercentile(np.abs(d), 99) / 3
    h1 = ax[1].hexbin(X[:, 0], X[:, 1], C=d, gridsize=35, cmap="RdBu_r",
                      reduce_C_function=np.mean, vmin=-v, vmax=v)
    fig.colorbar(h1, ax=ax[1], label="weakspot $-$ uniform")
    cal = df[(df.task == "california") & (df.arm == "uniform")]
    for a in ax:
        a.scatter(cal.cx, cal.cy, s=8, c="w", edgecolors="k", lw=0.5, label="detected centres")
        a.set_xlabel("longitude (scaled)")
    ax[0].set_ylabel("latitude (scaled)")
    ax[0].set_title("(a) Initial model")
    ax[1].set_title("(b) Error change, weakspot $-$ uniform")
    ax[0].legend(loc="upper right", fontsize=8)
    fig.savefig(out, dpi=200)
    plt.close(fig)


STYLE = {"weakspot": ("Weakspot (ours)", "#1f77b4", "o"), "loss": ("Per-point loss", "#8c564b", "s"),
         "jtt": ("JTT", "#d62728", "^"), "density": ("Density", "#9467bd", "D")}


def fig_frontier(out):
    """In-weakspot vs outside error over each arm's full grid (reported seeds)."""
    fr = pd.read_csv(RES / "fixed_data_frontier.csv")
    fr = fr[fr.arm != "ERROR"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    titles = {"synthetic": "(a) Synthetic, hard region", "california": "(b) California housing"}
    for ax, task in zip(axes, ["synthetic", "california"]):
        g = fr[fr.task == task].groupby(["arm", "hp"])[["err_in", "err_out", "mae"]].mean()
        u = g.loc["uniform"].iloc[0]
        for arm, (lab, col, mk) in STYLE.items():
            a = g.loc[arm]
            ok = a[a.err_out <= u.err_out * 1.6]   # drop collapsed settings off-scale
            ax.scatter(100 * (ok.err_out / u.err_out - 1), 100 * (ok.err_in / u.err_in - 1),
                       c=col, marker=mk, s=28, label=lab, alpha=0.85)
        ax.axhline(0, c="k", lw=0.6)
        ax.axvline(0, c="k", lw=0.6)
        ax.scatter([0], [0], c="k", marker="*", s=120, zorder=5, label="Uniform")
        ax.set_xlabel("error outside the weakspot vs uniform (%)")
        ax.set_title(titles[task])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error inside the weakspot vs uniform (%)")
    axes[0].legend(fontsize=8, loc="center right")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _frontier_panel(ax, fr, task, title, legend=False):
    g = fr[fr.task == task].groupby(["arm", "hp"])[["err_in", "err_out", "mae"]].mean()
    u = g.loc["uniform"].iloc[0]
    for arm, (lab, col, mk) in STYLE.items():
        a = g.loc[arm]
        ok = a[a.err_out <= u.err_out * 1.6]       # drop collapsed settings off-scale
        ax.scatter(100 * (ok.err_out / u.err_out - 1), 100 * (ok.err_in / u.err_in - 1),
                   c=col, marker=mk, s=22, label=lab, alpha=0.85)
    ax.axhline(0, c="k", lw=0.6)
    ax.axvline(0, c="k", lw=0.6)
    ax.scatter([0], [0], c="k", marker="*", s=100, zorder=5, label="Uniform")
    ax.set_xlabel("outside the weakspot (%)")
    ax.set_ylabel("inside the weakspot (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=7, loc="center right")


def fig_combined(maps, out):
    """One row: frontiers (synthetic, California) and the hard-region maps
    (initial error; weakspot-weighted minus uniform continuation)."""
    fr = pd.read_csv(RES / "fixed_data_frontier.csv")
    fr = fr[fr.arm != "ERROR"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1, 1, 1.05, 1.05]))
    _frontier_panel(ax[0], fr, "synthetic", "(a) Trade-off, hard region", legend=True)
    _frontier_panel(ax[1], fr, "california", "(b) Trade-off, California")
    k = "synthetic__"
    init = maps[k + "initial"].mean(0)
    d = maps[k + "weakspot"].mean(0) - maps[k + "uniform"].mean(0)
    res = int(np.sqrt(init.size))
    ext = (0, 1, 0, 1)
    im = ax[2].imshow(init.reshape(res, res), origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=ax[2], shrink=0.9, label="absolute error")
    ax[2].set_title("(c) Initial error, hard region")
    v = np.abs(d).max()
    im = ax[3].imshow(d.reshape(res, res), origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax[3], shrink=0.9, label="error change")
    ax[3].set_title("(d) Weakspot-weighted $-$ uniform")
    for a in ax[2:]:
        a.add_patch(plt.Circle((0.25, 0.75), 0.2, fill=False, ls="--", lw=1.3, color="k"))
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(RES / "fixed_data.csv")
    maps = dict(np.load(RES / "fixed_data_maps.npz"))
    df = add_region_cal(df, maps)
    tab = table(df)
    tab.to_csv(RES / "fixed_data_summary.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["task", "region", "noise", "arm", "whole", "p_holm", "inside", "p_in_holm",
            "outside", "gate_pass", "detect_dist", "uni_mae", "init_mae", "uni_in", "init_in"]
    print(tab[cols].round(3).to_string())
    print(json.loads((RES / "fixed_data_hp.json").read_text()))
    (RES / "figures").mkdir(exist_ok=True)
    fig_maps(maps, RES / "figures" / "fig_fixed_data_maps.png")
    fig_california(maps, df, RES / "figures" / "fig_fixed_data_california.png")
    fig_frontier(RES / "figures" / "fig_fixed_data_frontier.png")
    fig_combined(maps, RES / "figures" / "fig_fixed_data_combined.png")


if __name__ == "__main__":
    main()
