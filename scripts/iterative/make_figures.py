"""Figures for experiments A (anchor) and B (scheduled coverage), 20 iterations.

Three figures per experiment plus a direct A-vs-B comparison:

  fig1_<exp>_decay.png       the headline: guided-minus-random per iteration with a
                             95% CI, against the detector's own IoU with the induced
                             gap on a second axis, so the advantage and the target
                             that produced it rise and fall in the same frame.
  fig2_<exp>_curves.png      absolute MAE of both arms, plus in-gap and out-of-gap
                             error, so the "local win, global cost" split is visible.
  fig3_<exp>_migration.png   top-down path of the detected weakspot centre with a
                             dashed, transparent 2-sigma ellipse at each round.
  fig4_compare.png           A vs B side by side.
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
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import LineCollection

RES = APP / "data/experiment_results/iterative_data_selective_training"
# Figures sit beside the results that produced them, matching the single-round
# study's layout (data/experiment_results/<experiment>/figures/).
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Experiment pairs. ``python make_figures.py [max_it] [pair]`` selects one; each
# pair writes its own figure filenames, so pairs never overwrite each other.
PAIRS = {
    "default": ({"iter20_anchor":    "A — anchor (σ=0.5, α=0.5, both constant)",
                 "iter20_schedules": "B — scheduled (σ linear widen, α linear decay)"}, ""),
    "i3n50":   ({"iter20_anchor_i3n50":    "A — static, weak initial model (3 iters, 50 pts)",
                 "iter20_schedules_i3n50": "B — scheduled, weak initial model (3 iters, 50 pts)"},
                "_i3n50"),
    # The headline pair: the cell the grid search selected, re-run on seeds
    # disjoint from that search.
    "i5n50":   ({"iter20_anchor_i5n50":    "static  (σ=0.5, α=0.5 held constant)",
                 "iter20_schedules_i5n50": "dynamic  (σ widens, α decays)"},
                "_i5n50"),
}
PAIR = sys.argv[2] if len(sys.argv) > 2 else "default"
EXPS, PAIR_SUF = PAIRS[PAIR]
G, R = "gnew", "rnew"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


# Optional horizon cut: ``python make_figures.py 10`` redraws everything using only
# iterations 0…10 and writes to *_it10.png, leaving the full-length figures intact.
# The runs are stored per iteration, so shortening the horizon is a filter on the
# saved data — no re-running, and the numbers are identical to the full run's first
# ten rounds rather than a fresh sample of them.
MAX_IT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
SUF = f"_it{MAX_IT}" if MAX_IT else ""

# Iteration 0 is the shared initial model at MAE 0.95 (1.85 inside the gap), an
# order of magnitude above everything that follows; plotting from it squashes the
# entire loop into the bottom eighth of the panel. The curves therefore start at
# iteration 1 and every panel uses FIXED limits, identical across experiments and
# across horizons, so anchor and scheduled — and the 7/10/20-iteration cuts — can
# be laid side by side and read against the same scale. Iteration 0 carries no
# comparative information anyway: both arms are the same model there by
# construction. Widen these if a future run leaves the window.
def _limits():
    """Axis limits computed from the pair's FULL-LENGTH data, then reused for
    every horizon — so the 7/10/20-iteration cuts of one pair share a scale and
    stay comparable, while a pair with a different error scale (a weaker initial
    model raises MAE everywhere) gets limits that suit it instead of being
    squashed against constants tuned for another."""
    d = pd.concat([pd.read_csv(RES / f"sweep__{n}.csv", low_memory=False) for n in EXPS])
    d = d[d.iteration >= 1]
    def span(cols, pad=0.10):
        g = pd.concat([d.groupby(["sweep_config", "iteration"])[c].mean() for c in cols])
        lo, hi = float(g.min()), float(g.max())
        m = (hi - lo) * pad
        return (lo - m, hi + m)
    gap = (d[f"{G}_mae"] - d[f"{R}_mae"]).groupby(
        [d.sweep_config, d.iteration]).mean()
    gm = (float(gap.max()) - float(gap.min())) * 0.15
    return (span([f"{G}_mae", f"{R}_mae"]), span([f"{G}_err_in", f"{R}_err_in"]),
            (float(gap.min()) - gm, float(gap.max()) + gm))


YLIM_MAE, YLIM_IN, YLIM_GAP = _limits()


def load(name):
    df = pd.read_csv(RES / f"sweep__{name}.csv", low_memory=False)
    df = df[df.iteration >= 0]
    if MAX_IT is not None:
        df = df[df.iteration <= MAX_IT]
    return df.copy()


def ci(frame, col):
    g = frame.groupby("iteration")[col].agg(["mean", "std", "count"])
    g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    return g.reset_index()


# ── fig 1: the decay, against the disappearing target ────────────────────
def fig_decay(df, name, title):
    d = df[df.iteration > 0].copy()
    d["gap"] = d[f"{G}_mae"] - d[f"{R}_mae"]
    g, iou = ci(d, "gap"), ci(d, "det_iou")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(g.iteration, g["mean"] - g.hw, g["mean"] + g.hw,
                    color="#2ca02c", alpha=0.18, lw=0)
    ax.plot(g.iteration, g["mean"], "-o", color="#2ca02c", ms=4, lw=2,
            label="guided − random MAE  (below 0 = guided better)")
    ax.set_ylim(*YLIM_GAP)
    win = g["mean"] + g.hw < 0
    if win.any():
        ax.fill_between(g.iteration, *YLIM_GAP, where=win,
                        color="#2ca02c", alpha=0.06, lw=0)
    ax.set_xlabel("iteration"); ax.set_ylabel("guided − random MAE")
    ax.set_xticks(range(1, int(g.iteration.max()) + 1, 2 if MAX_IT is None or MAX_IT > 12 else 1))
    # Detector IoU used to share this axis; it now lives in the detection figure so
    # this panel carries one comparison only.
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout(); fig.savefig(OUT / f"fig1_{name}_decay{SUF}.png"); plt.close(fig)


# ── fig 5: detection quality on its own axes ─────────────────────────────
def fig_detection(df, name, title):
    """IoU, centroid distance and detected size per round — the target dissolving,
    kept apart from the guided-vs-random comparison rather than sharing its axis."""
    d = df[df.iteration >= 1]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for col, lab, c, m in (("det_iou", "IoU with the induced weakspot", "#1f77b4", "o"),
                           ("det_distance", "distance to the true centre", "#d62728", "s"),
                           ("det_area", "detected size (area fraction)", "#9467bd", "^")):
        if col not in d.columns or d[col].isna().all():
            continue
        s = ci(d, col)
        ax.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                        color=c, alpha=0.13, lw=0)
        ax.plot(s.iteration, s["mean"], "-" + m, color=c, ms=4, lw=1.8, label=lab)
    ax.set_xlabel("iteration"); ax.set_ylabel("value")
    ax.set_xticks(range(1, int(d.iteration.max()) + 1,
                        2 if MAX_IT is None or MAX_IT > 12 else 1))
    ax.legend(fontsize=8, framealpha=0.95)
    fig.tight_layout(); fig.savefig(OUT / f"fig5_{name}_detection{SUF}.png"); plt.close(fig)


# ── fig 2: absolute curves, in-gap vs out-of-gap ─────────────────────────
def fig_curves(df, name, title):
    """Progression over the loop: whole-area MAE (left) and MAE inside the
    original induced weakspot (right), guided against random in each.

    Two panels of two lines rather than one panel of four: overlaying
    guided/random x inside/outside on a single axis put every line in the same
    band and none could be read. Split this way each panel holds exactly the
    comparison that matters, on a scale suited to it — the global cost on the
    left, the local repair the method is actually for on the right.
    """
    d = df[df.iteration >= 1]
    panels = ((f"{G}_mae", f"{R}_mae", "whole-area MAE", YLIM_MAE),
              (f"{G}_err_in", f"{R}_err_in", "MAE inside the original weakspot", YLIM_IN))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for ax, (gcol, rcol, ylab, ylim) in zip(axes, panels):
        for col, lab, c in ((gcol, "guided", "#2ca02c"), (rcol, "random", "#d62728")):
            s = ci(d, col)
            ax.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                            color=c, alpha=0.15, lw=0)
            ax.plot(s.iteration, s["mean"], "-o", color=c, ms=3.5, lw=1.8, label=lab)
        ax.set_xlabel("iteration"); ax.set_ylabel(ylab); ax.set_ylim(*ylim)
        ax.set_xticks(range(1, int(d.iteration.max()) + 1,
                            2 if MAX_IT is None or MAX_IT > 12 else 1))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"fig2_{name}_curves{SUF}.png"); plt.close(fig)


# ── fig 3: weakspot migration, top-down ──────────────────────────────────
def fig_migration(df, name, title, seed=None):
    """Top-down migration map plus a strip quantifying the target dissolving.

    Ellipses are drawn at 1σ, not 2σ: early rounds detect a region covering most
    of the square (area_frac ≈ 0.95 — the model is bad everywhere, so the error
    surface has no localised peak), and at 2σ those swamp the panel entirely.
    """
    # The map shows one representative trajectory; averaging centres across seeds
    # would place the mean somewhere none of them visited. Pairs use different seed
    # sets, so pick one that exists here rather than assuming a fixed seed.
    if seed is None or seed not in set(df.seed.unique()):
        seed = int(sorted(df.seed.unique())[0])
    s = df[(df.seed == seed) & (df.iteration > 0)].sort_values("iteration")
    cx, cy = s.det_cx.to_numpy(), s.det_cy.to_numpy()
    n = len(cx)
    if n == 0:
        return
    cmap = plt.get_cmap("viridis")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.8, 7.4), gridspec_kw=dict(height_ratios=[3, 1.05]))

    r = float(df.radius.iloc[0])
    if r > 0:
        ax.add_patch(Circle((float(df.center_x.iloc[0]), float(df.center_y.iloc[0])), r,
                            fill=False, ec="red", lw=2.2, ls="--", zorder=6,
                            label="induced weakspot (ground truth)"))

    for i in range(n):
        smaj, smin = s.det_smaj.iloc[i], s.det_smin.iloc[i]
        if not (np.isfinite(smaj) and np.isfinite(smin)):
            continue
        col = cmap(i / max(n - 1, 1))
        # Outline only — twenty translucent fills stacked on one square turned
        # into an opaque wash that hid both the path and the ground-truth circle.
        ax.add_patch(Ellipse((cx[i], cy[i]), 2 * smaj, 2 * smin,      # 1-sigma extent
                             angle=np.degrees(s.det_angle.iloc[i]),
                             fill=False, ec=col, ls="--", lw=1.0,
                             alpha=0.75, zorder=2))

    pts = np.column_stack([cx, cy]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(1, n), lw=1.7,
                        alpha=0.85, zorder=5)
    lc.set_array(np.arange(1, n)); ax.add_collection(lc)
    sc = ax.scatter(cx, cy, c=np.arange(1, n + 1), cmap=cmap, s=46,
                    edgecolor="black", lw=0.6, zorder=6)

    # The detector often returns the same grid cell on different rounds; stack the
    # labels of coincident points instead of overprinting them.
    seen: dict = {}
    for i in range(n):
        key = (round(cx[i], 3), round(cy[i], 3))
        k = seen.get(key, 0); seen[key] = k + 1
        ax.annotate(str(i + 1), (cx[i], cy[i]), fontsize=6.2,
                    xytext=(0, 7 + 8 * k), textcoords="offset points",
                    ha="center", zorder=7,
                    color=cmap(i / max(n - 1, 1)), fontweight="bold")

    fig.colorbar(sc, ax=ax, label="iteration", fraction=0.046, pad=0.03)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # companion strip: the same story as numbers, averaged over every seed
    d = df[df.iteration > 0]
    for col, lab, c in (("det_iou", "IoU with induced gap", "#1f77b4"),
                        ("det_area", "detected area (fraction of square)", "#8c564b"),
                        ("det_distance", "distance to true centre", "#d62728")):
        g = ci(d, col)
        ax2.plot(g.iteration, g["mean"], "-o", ms=2.6, lw=1.4, color=c, label=lab)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("value")
    ax2.set_xticks(range(0, int(d.iteration.max()) + 1, 2 if MAX_IT is None or MAX_IT > 12 else 1))
    ax2.legend(fontsize=6.8, ncol=1, loc="upper right", framealpha=0.9)

    fig.tight_layout(); fig.savefig(OUT / f"fig3_{name}_migration{SUF}.png"); plt.close(fig)


# ── fig 4: A vs B ────────────────────────────────────────────────────────
def fig_compare(frames):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.axhline(0, color="black", lw=1)
    for (name, title), c in zip(EXPS.items(), ("#7f7f7f", "#ff7f0e")):
        d = frames[name][frames[name].iteration > 0].copy()
        d["gap"] = d[f"{G}_mae"] - d[f"{R}_mae"]
        g = ci(d, "gap")
        ax.fill_between(g.iteration, g["mean"] - g.hw, g["mean"] + g.hw,
                        color=c, alpha=0.15, lw=0)
        ax.plot(g.iteration, g["mean"], "-o", color=c, ms=4, lw=2, label=title)
    ax.set_xlabel("iteration"); ax.set_ylabel("guided − random MAE")
    ax.set_ylim(*YLIM_GAP)
    ax.set_xticks(range(1, int(max(frames[n].iteration.max() for n in EXPS)) + 1,
                        2 if MAX_IT is None or MAX_IT > 12 else 1))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / f"fig4_compare{PAIR_SUF}{SUF}.png"); plt.close(fig)


if __name__ == "__main__":
    frames = {}
    for name, title in EXPS.items():
        df = load(name); frames[name] = df
        fig_decay(df, name, title)
        fig_detection(df, name, title)
        fig_curves(df, name, title)
        fig_migration(df, name, title)
        print(f"{name}: {len(df)} rows, {df.seed.nunique()} seeds, "
              f"{int(df.iteration.max())} iterations")
    fig_compare(frames)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p)
