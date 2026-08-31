"""The three perspectives on the guided-advantage decay, for one experiment pair.

  persp1_initial_training<pair>.png   how the effect depends on the INITIAL TRAINING
        setup — the 15-cell grid (iters_initial x n_train). Reads the saved
        init_training_search.csv, so it is the same data for every pair.
  persp2_decay_iterations<pair>.png   decay over TRAINING ITERATIONS: the gap round
        by round, static against dynamic, with 95% CIs.
  persp3_decay_weakspot_size<pair>.png  decay against the WEAKSPOT SIZE that
        produced it: the same trajectory replotted with the detected region's area
        on the x-axis instead of the round number. This is the mechanism plot — if
        the advantage is really a function of how much weakspot is left rather than
        of elapsed rounds, the two arms should collapse onto one curve here even
        though they separate in perspective 2.

Usage:  python make_perspective_figures.py [pair]      (pair: i5n50 | i3n50 | default)
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

APP = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application")
sys.path.insert(0, str(APP))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = APP / "data/experiment_results/iterative_data_selective_training"
OUT = RES / "figures/Exploration"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "i5n50":   ("iter20_anchor_i5n50", "iter20_schedules_i5n50", "_i5n50"),
    "i3n50":   ("iter20_anchor_i3n50", "iter20_schedules_i3n50", "_i3n50"),
    "default": ("iter20_anchor", "iter20_schedules", ""),
}
PAIR = sys.argv[1] if len(sys.argv) > 1 else "i5n50"
STATIC, DYNAMIC, SUF = PAIRS[PAIR]
G, R = "gnew", "rnew"
C_ST, C_DY = "#7f7f7f", "#ff7f0e"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    d = pd.read_csv(RES / f"sweep__{name}.csv", low_memory=False)
    d = d[d.iteration >= 1].copy()
    d["gap"] = d[f"{G}_mae"] - d[f"{R}_mae"]
    return d


# ── perspective 1: the initial-training grid ─────────────────────────────
def persp_initial():
    s = pd.read_csv(RES / "init_training_search.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    marks = {50: "o", 100: "s", 200: "^"}
    for nt, m in marks.items():
        d = s[s.n_train == nt].sort_values("iters_initial")
        axes[0].plot(d.iters_initial, d.late_static, "-" + m, color=C_ST, ms=5,
                     lw=1.6, alpha=0.55 + 0.2 * (nt == 50),
                     label=f"static · n_train={nt}")
        axes[0].plot(d.iters_initial, d.late_dyn, "--" + m, color=C_DY, ms=5,
                     lw=1.6, alpha=0.55 + 0.2 * (nt == 50),
                     label=f"dynamic · n_train={nt}")
        axes[1].plot(d.iters_initial, d.rescue, "-" + m, color="#2ca02c", ms=5,
                     lw=1.6, alpha=0.55 + 0.2 * (nt == 50), label=f"n_train={nt}")
    for a in axes:
        a.axhline(0, color="black", lw=1)
        a.set_xlabel("initial training iterations")
        a.set_xticks(sorted(s.iters_initial.unique()))
    axes[0].set_ylabel("late-window gap (rounds 11–20)")
    axes[1].set_ylabel("rescue  (static late − dynamic late)")
    axes[0].legend(fontsize=6.5, ncol=2); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"persp1_initial_training.png"); plt.close(fig)


# ── perspective 2: decay over training iterations ────────────────────────
def persp_iterations(st, dy):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhline(0, color="black", lw=1)
    for d, c, lab in ((st, C_ST, "static  (σ, α held constant)"),
                      (dy, C_DY, "dynamic  (σ widens, α decays)")):
        g = d.groupby("iteration").gap.agg(["mean", "std", "count"])
        hw = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
        ax.fill_between(g.index, g["mean"] - hw, g["mean"] + hw, color=c,
                        alpha=0.15, lw=0)
        ax.plot(g.index, g["mean"], "-o", color=c, ms=4, lw=2, label=lab)
    ax.set_xlabel("training iteration"); ax.set_ylabel("guided − random MAE")
    ax.set_xticks(range(1, int(st.iteration.max()) + 1, 2))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / f"persp2_decay_iterations{SUF}.png"); plt.close(fig)


# ── perspective 3: decay against the weakspot size ───────────────────────
def persp_size(st, dy):
    """Same trajectories, but against the detected region's area rather than the
    round index. Points are per-round means coloured by round; the heavy line is
    a binned trend over the pooled arms."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    for d, c, lab in ((st, C_ST, "static"), (dy, C_DY, "dynamic")):
        g = d.groupby("iteration").agg(gap=("gap", "mean"), area=("det_area", "mean"))
        axes[0].plot(g.area, g.gap, "-o", color=c, ms=4, lw=1.2, alpha=0.75, label=lab)
        for it in (1, 2, 5, 10, 20):
            if it in g.index:
                axes[0].annotate(str(it), (g.area.loc[it], g.gap.loc[it]), fontsize=6.5,
                                 xytext=(0, 6), textcoords="offset points", ha="center",
                                 color=c)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("detected weakspot size  (area fraction of the input square)")
    axes[0].set_ylabel("guided − random MAE")
    axes[0].legend(fontsize=8, title="round labelled", title_fontsize=7)

    # binned trend across both arms and every seed: does size alone predict the gap?
    both = pd.concat([st.assign(arm="static"), dy.assign(arm="dynamic")])
    both = both.dropna(subset=["det_area", "gap"])
    q = pd.qcut(both.det_area, 8, duplicates="drop")
    b = both.groupby(q, observed=True).agg(
        area=("det_area", "mean"), gap=("gap", "mean"),
        sd=("gap", "std"), n=("gap", "count"))
    b["hw"] = 1.96 * b.sd / np.sqrt(b.n)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].fill_between(b.area, b.gap - b.hw, b.gap + b.hw, color="#1f77b4",
                         alpha=0.18, lw=0)
    axes[1].plot(b.area, b.gap, "-o", color="#1f77b4", ms=5, lw=2,
                 label="both arms pooled, binned by size")
    axes[1].set_xlabel("detected weakspot size  (area fraction)")
    axes[1].set_ylabel("guided − random MAE")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"persp3_decay_weakspot_size{SUF}.png"); plt.close(fig)


if __name__ == "__main__":
    st, dy = load(STATIC), load(DYNAMIC)
    persp_initial()
    persp_iterations(st, dy)
    persp_size(st, dy)
    corr = np.corrcoef(pd.concat([st, dy]).dropna(subset=["det_area", "gap"]).det_area,
                       pd.concat([st, dy]).dropna(subset=["det_area", "gap"]).gap)[0, 1]
    print(f"pair {PAIR}: corr(detected size, gap) = {corr:+.3f}")
    for p in sorted(OUT.glob("persp*")):
        print("  ", p.name)
