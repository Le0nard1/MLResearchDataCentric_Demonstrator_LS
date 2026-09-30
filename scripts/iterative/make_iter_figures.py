"""
Figures for Paper B, Sections 4.2-4.3 (iterated weakspot curation, new points only).

Reads the pilot results (iter_pilot2.csv: new points only; iter_pilot.csv: its random
loop retrained on the accumulated set serves as the compute reference) and re-runs one
seed in-process with the trace recorder for the weakspot maps. Until the full runs
exist, every figure is a placeholder built from the pilots (10 seeds, clean labels).

    python -m scripts.iterative.make_iter_figures --out <paper>/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.iterative import iter_budget as IB

RES = IB.RES
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# One colour per entity throughout the paper (default categorical order).
COL = {"random": INK2, "ws_default": "#2a78d6", "ws_tuned": "#eb6834",
       "rho_filter": "#1baf7a", "rho_landscape": "#eda100",
       "static": "#eb6834", "dynamic": "#e87ba4", "adaptive": "#4a3aa7"}
LABEL = {"random": "Random", "ws_default": r"Weakspot, default ($\alpha$=0.2)",
         "ws_tuned": r"Weakspot, tuned ($\alpha$=1)", "rho_filter": "RHO-filter",
         "rho_landscape": "RHO-landscape", "static": "Static", "dynamic": "Dynamic",
         "adaptive": "Adaptive"}
DS = {"synth2d": "Synthetic", "houses": "California housing", "sulfur": "Sulfur"}
ROUND_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42})


def _ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def _band(ax, df, col, key, **kw):
    g = df.groupby("round")[col]
    m = g.mean()
    h = g.apply(lambda s: _ci(s)[1])
    ax.plot(m.index, m.values, color=COL[key], label=LABEL[key], marker="o",
            markersize=3, **kw)
    ax.fill_between(m.index, m - h, m + h, color=COL[key], alpha=0.15, lw=0)


def gains(d, col="mae", arms=None):
    """Per-seed error reduction [%] against the mean of the random loops."""
    K = ["dataset", "region", "seed", "round"]
    r = d[d.arm == "random"].groupby(K)[col].mean()
    out = []
    for arm, x in d[d.arm != "random"].groupby("arm"):
        if arms and arm not in arms:
            continue
        w = x.set_index(K)[col]
        g = (100 * (r - w) / r).dropna().rename("G").reset_index()
        out.append(g.assign(arm=arm))
    g = pd.concat(out)
    return g[g["round"] > 0]


def _stamp(fig):
    fig.text(0.5, -0.09, "PLACEHOLDER - pilot data, 10 seeds, clean labels",
             ha="center", va="bottom", fontsize=6, color="#d03b3b")


def _save(fig, out, name):
    _stamp(fig)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Experiment 1
# ─────────────────────────────────────────────────────────────
def fig_exp1(p2, p1, out):
    methods = ["ws_default", "ws_tuned", "rho_filter", "rho_landscape"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    for j, ds in enumerate(DS):
        ax = axes[0, j]
        x = p2[p2.dataset == ds]
        acc = p1[(p1.dataset == ds) & (p1.arm == "random") & (p1.regime == "acc")]
        init = x[x["round"] == 0].mae.mean()
        ax.axhline(init, color=INK2, ls=":", lw=1.1, label="No retraining")
        m = acc.groupby("round").mae.mean()
        ax.plot(m.index, m.values, color=INK2, ls="--", lw=1.1,
                label="Random, accumulated (7x compute)")
        rnd = x[x.arm == "random"].groupby(["region", "seed", "round"]).mae.mean()
        _band(ax, rnd.reset_index(), "mae", "random")
        for a in methods:
            _band(ax, x[x.arm == a], "mae", a)
        ax.set_yscale("log")
        ax.set_title(DS[ds], color=INK)
        ax.set_xticks(range(0, 9))
        if j == 0:
            ax.set_ylabel("Test MAE (log)")
        ax = axes[1, j]
        g = gains(x, arms=["ws_default", "ws_tuned"])
        ax.axhline(0, color=INK2, lw=0.9)
        for a in ["ws_default", "ws_tuned"]:
            _band(ax, g[g.arm == a], "G", a)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Round $t$")
        if j == 0:
            ax.set_ylabel(r"Error reduction $G_t$ vs. random [%]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    _save(fig, out, "fig_exp1_iterating")


# ─────────────────────────────────────────────────────────────
# Experiment 2: dynamics, schedules, maps
# ─────────────────────────────────────────────────────────────
def fig_exp2_dynamics(p2, out):
    x = p2[(p2.dataset == "synth2d") & (p2["round"] > 0)]
    x = x[(x.arm != "random") | (x.traj == 0)]
    arms = {"random": "random", "ws_tuned": "static", "ws_dynamic": "dynamic",
            "ws_adaptive": "adaptive"}
    go = gains(p2[p2.dataset == "synth2d"], "err_out", list(arms))
    panels = [("severity", "Severity $S_t$"), ("drift", r"Drift $\delta_t$"),
              ("n_in_region", "Selected points in region"),
              ("G", r"Error reduction outside region [%]")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, (col, lab) in zip(axes, panels):
        for arm, key in arms.items():
            if col == "G":
                if arm == "random":
                    continue
                d = go[go.arm == arm]
            else:
                d = x[x.arm == arm]
            _band(ax, d, col, key)
        if col == "G":
            ax.axhline(0, color=INK2, lw=0.9)
        ax.set_title(lab, color=INK)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Round $t$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    _save(fig, out, "fig_exp2_dynamics")


def fig_exp2_schedules(p2, out):
    rows = [(r"Full focus ($\alpha$=1)",
             {"ws_tuned": "static", "ws_dynamic": "dynamic", "ws_adaptive": "adaptive"}),
            (r"Moderate focus ($\alpha$=0.5)",
             {"ws_half": "static", "ws_half_dynamic": "dynamic",
              "ws_half_adaptive": "adaptive"})]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.8), sharex=True)
    for i, (title, arms) in enumerate(rows):
        for j, ds in enumerate(DS):
            ax = axes[i, j]
            g = gains(p2[p2.dataset == ds], arms=list(arms))
            ax.axhline(0, color=INK2, lw=0.9)
            for arm, key in arms.items():
                _band(ax, g[g.arm == arm], "G", key)
            if i == 0:
                ax.set_title(DS[ds], color=INK)
            if j == 0:
                ax.set_ylabel(f"{title}\n$G_t$ vs. random [%]")
            if i == 1:
                ax.set_xlabel("Round $t$")
            ax.set_xticks(range(1, 9))
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    _save(fig, out, "fig_exp2_schedules")


def trace(name, region, seed):
    IB.TRACE.clear()
    IB.FIXED.update(record=True)
    IB.run_one(name, region, "clean", seed, stage="pilot2")
    IB.FIXED.pop("record")
    return [t for t in IB.TRACE if t["arm"] in
            ("random", "ws_tuned", "ws_dynamic", "ws_adaptive") and t["traj"] == 0]


def _round_cmap():
    cmap = ListedColormap(ROUND_RAMP)
    return cmap, BoundaryNorm(np.arange(0.5, 9.5), cmap.N)


def fig_exp2_maps(tr, out, name, pca=False, title=""):
    arms = [("random", "Random"), ("ws_tuned", r"Static ($\alpha$=1)"),
            ("ws_dynamic", "Dynamic"), ("ws_adaptive", "Adaptive")]
    t0 = tr[0]
    if pca:
        P = PCA(2).fit(t0["U_R"])
        proj = lambda U: P.transform(U)
        base = proj(t0["U_R"])
        inside = base[t0["in_R"]]
    cmap, norm = _round_cmap()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharex=True, sharey=True)
    for ax, (arm, lab) in zip(axes, arms):
        rs = sorted([t for t in tr if t["arm"] == arm], key=lambda t: t["round"])
        if pca:
            ax.scatter(base[:, 0], base[:, 1], s=1, color=GRID, lw=0, rasterized=True)
            ax.scatter(inside[:, 0], inside[:, 1], s=2, color="#8a8883", lw=0,
                       rasterized=True, label="Constructed region")
            sel = [proj(t["sel_U"]) for t in rs]
            cen = np.array([proj(t["centre_U"][None])[0] for t in rs])
        else:
            g = np.linspace(0, 1, 200)
            GX, GY = np.meshgrid(g, g)
            Z = rs[0]["region_fn"](np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
            ax.contourf(GX, GY, Z, levels=[0.5, 1.5], colors=["#efeeea"])
            ax.contour(GX, GY, Z, levels=[0.5], colors=[INK], linewidths=1.0)
            sel = [t["sel_X"] for t in rs]
            cen = np.array([t["centre_X"] for t in rs])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", "0.5", "1"])
        for t, S in zip(rs, sel):
            ax.scatter(S[:, 0], S[:, 1], s=1.5, color=cmap(norm(t["round"])), lw=0,
                       alpha=0.35, rasterized=True)
        ax.plot(cen[:, 0], cen[:, 1], color=INK2, lw=0.8, zorder=3)
        sc = ax.scatter(cen[:, 0], cen[:, 1], c=[t["round"] for t in rs], cmap=cmap,
                        norm=norm, s=34, edgecolor="#fcfcfb", linewidth=1.2, zorder=4,
                        marker="D")
        ax.set_title(lab, color=INK)
        ax.grid(False)
        ax.set_xlabel("PC 1" if pca else "$x_1$")
    axes[0].set_ylabel("PC 2" if pca else "$x_2$")
    cb = fig.colorbar(sc, ax=axes, ticks=range(1, 9), fraction=0.025, pad=0.01)
    cb.set_label("Round $t$")
    cb.outline.set_visible(False)
    if title:
        fig.suptitle(title, color=INK, y=1.02)
    _save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(RES / "iter_pilot2.csv")
    p1 = pd.read_csv(RES / "iter_pilot.csv")
    fig_exp1(p2, p1, out)
    fig_exp2_dynamics(p2, out)
    fig_exp2_schedules(p2, out)
    # Seed 9002 is typical of the under-learnt condition (static drift 0.71 per round
    # against 0.16 under random selection; median ratio over seeds about 4).
    fig_exp2_maps(trace("synth2d", "hard", 9002), out, "fig_exp2_maps_synthetic")
    fig_exp2_maps(trace("sulfur", "hard", 9002), out, "fig_exp2_maps_pca", pca=True)
    print("figures written to", out)


if __name__ == "__main__":
    main()
