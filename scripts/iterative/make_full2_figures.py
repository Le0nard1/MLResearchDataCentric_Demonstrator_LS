"""
Final figures for Paper B, Section 4 (replay regime, full run ``full2``).

    python -m scripts.iterative.make_full2_figures --out <paper>/figures

Figure 1: error reduction per round for the static guidance fractions (synthetic task with a
repairable and with a scarce weakspot, and the six real datasets).
Figure 2: weakspot dynamics on the synthetic task (inside-region error reduction, selected
points inside the region, weakspot size).
Figure 3: detected weakspot after every round of one synthetic run (traced in-process).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from scripts.iterative import iter_budget as IB
from scripts.iterative.make_iter_figures import INK, INK2, GRID, _ci  # noqa: F401  (rcParams)

RES = IB.RES
K = ["dataset", "region", "seed", "round"]
COL = {"random": INK2, "ws_default": "#2a78d6", "ws_half": "#1baf7a", "ws_tuned": "#eb6834",
       "ws_half_size_switch": "#4a3aa7"}
LAB = {"random": "Random", "ws_default": r"Static, $\alpha=0.2$", "ws_half": r"Static, $\alpha=0.5$",
       "ws_tuned": r"Static, $\alpha=1$", "ws_half_size_switch": r"Size-adaptive, $\alpha=0.5$"}
REP, S10 = "sparse_init:0.25:std", "sparse_all:0.1:std"


def gains(d, col):
    r = d[d.arm == "random"].groupby(K)[col].mean()
    out = []
    for arm, x in d[~d.arm.isin(["random", "random_new"])].groupby("arm"):
        w = x.set_index(K)[col]
        out.append((100 * (r - w) / r).dropna().rename("G").reset_index().assign(arm=arm))
    g = pd.concat(out)
    return g[g["round"] > 0]


def band(ax, df, col, arm, ls="-", label=None):
    g = df.groupby("round")[col]
    m = g.mean()
    h = g.apply(lambda s: _ci(s)[1])
    ax.plot(m.index, m.values, color=COL[arm], ls=ls, marker="o", markersize=3,
            label=label or LAB[arm])
    ax.fill_between(m.index, m - h, m + h, color=COL[arm], alpha=0.13, lw=0)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)


def fig_rounds(d, out):
    G = gains(d, "mae")
    panels = [("Synthetic, repairable weakspot", (G.dataset == "synth2d") & (G.region == REP)),
              ("Synthetic, scarce weakspot ($\\rho=0.1$)", (G.dataset == "synth2d") & (G.region == S10)),
              ("Real datasets (all conditions)", G.dataset != "synth2d")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, (title, m) in zip(axes, panels):
        ax.axhline(0, color=INK2, lw=0.9)
        for arm in ["ws_default", "ws_half", "ws_tuned"]:
            band(ax, G[m & (G.arm == arm)], "G", arm)
        ax.set_title(title, color=INK)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Round $t$")
    axes[0].set_ylabel(r"Error reduction $G_t$ [%]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, out, "fig_b_rounds")


def fig_dynamics(d, out):
    syn = d[d.dataset == "synth2d"]
    Gi = gains(syn, "err_in")
    x = syn[(syn["round"] > 0) & ((syn.arm != "random") | (syn.traj == 0))]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    ax = axes[0]
    ax.axhline(0, color=INK2, lw=0.9)
    band(ax, Gi[(Gi.region == REP) & (Gi.arm == "ws_half")], "G", "ws_half",
         label="Repairable")
    band(ax, Gi[(Gi.region == S10) & (Gi.arm == "ws_half")], "G", "ws_half", ls="--",
         label="Scarce ($\\rho=0.1$)")
    ax.set_title("Error reduction inside region [%]", color=INK)
    ax.legend(frameon=False, loc="lower left")
    for ax, col, title in [(axes[1], "n_in_region", "Selected points inside region"),
                           (axes[2], "size", "Weakspot size $z_t$")]:
        for arm in ["random", "ws_half"]:
            band(ax, x[(x.region == REP) & (x.arm == arm)], col, arm)
        ax.set_title(title, color=INK)
    for ax in axes:
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Round $t$")
    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(h, l, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig_b_dynamics")


def fig_maps(out, seed=8203):
    IB.TRACE.clear()
    IB.FIXED.update(record=True)
    IB.run_one("synth2d", REP, "clean", seed, stage="full2")
    IB.FIXED.pop("record")
    rows = [("random", "Random"), ("ws_half", "Static\n" r"$\alpha=0.5$"),
            ("ws_tuned", "Static\n" r"$\alpha=1$"),
            ("ws_half_size_switch", "Size-adaptive\n" r"$\alpha=0.5$")]
    tr = [t for t in IB.TRACE if t["arm"] in dict(rows) and t["traj"] == 0]
    seq = LinearSegmentedColormap.from_list(
        "seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    g = np.linspace(0, 1, 90)
    GX, GY = np.meshgrid(g, g)
    grid = np.c_[GX.ravel(), GY.ravel()]
    Z_true = tr[0]["region_fn"](grid).reshape(GX.shape)
    vmax = np.quantile([t for t in tr if t["round"] == 1][0]["land"], 0.99)
    fig, axes = plt.subplots(len(rows), 8, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for i, (arm, lab) in enumerate(rows):
        rs = sorted([t for t in tr if t["arm"] == arm], key=lambda t: t["round"])
        for j, t in enumerate(rs):
            ax = axes[i, j]
            Z = KNeighborsRegressor(10, weights="distance").fit(t["X_av"], t["land"]) \
                .predict(grid).reshape(GX.shape)
            im = ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), cmap=seq, vmin=0, vmax=vmax,
                           interpolation="bilinear", rasterized=True)
            ax.contour(GX, GY, Z, levels=[np.quantile(t["land"], 0.7)], colors=["#eb6834"],
                       linewidths=0.8)
            ax.contour(GX, GY, Z_true, levels=[0.5], colors=[INK], linewidths=0.8)
            c = t["centre_X"]
            ax.scatter([c[0]], [c[1]], marker="D", s=18, color="#eb6834", edgecolor="#fcfcfb",
                       linewidth=0.9, zorder=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(GRID)
            if i == 0:
                ax.set_title(f"Round {t['round']}", color=INK)
        axes[i, 0].set_ylabel(lab, color=INK, rotation=0, ha="right", va="center")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(r"Error landscape $\hat e$")
    cb.outline.set_visible(False)
    fig.legend([Line2D([], [], color=INK, lw=0.8), Line2D([], [], color="#eb6834", lw=0.8),
                Line2D([], [], marker="D", color="#eb6834", lw=0, markersize=5)],
               ["Constructed region", "Detected weakspot (top 30%)", "Detected centre"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.47, 1.03))
    save(fig, out, "fig_b_maps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RES / "iter_full2.csv")
    fig_rounds(d, out)
    fig_dynamics(d, out)
    fig_maps(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
