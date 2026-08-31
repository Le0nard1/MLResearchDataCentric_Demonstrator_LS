"""Storyline figures — one fig2-style curves pair and one fig3-style migration
map per storyline step, in the exact style of scripts/iterative/make_figures.py.

Steps covered (output: figures/Storyline/):
  S1  anchor, K=20, constant mix           — works once, then repetition hurts
  S5a exponential dump x0.1 (a0=1), K=8    — 'guided round 1, random after' -> parity
  S5b adaptive severity gate (a0=0.5), K=8 — scale mix with pronouncedness -> parity
  S6  undertrained init (3 iters, 50 pts)  — less pretraining changes nothing
  S7a-d the four policies under scarcity (deficit 0.98, sigma 0.15), 30 seeds
  S8  severity gate under scarcity, K=20   — the win persists
Within each group the y-limits are shared so panels can be laid side by side.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import LineCollection

RES = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training")
OUT = RES / "figures" / "Storyline"
OUT.mkdir(parents=True, exist_ok=True)

G, R = "gnew", "rnew"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(csv, **filters):
    df = pd.read_csv(RES / f"sweep__{csv}.csv", low_memory=False)
    df = df[df.iteration >= 0]
    for k, v in filters.items():
        df = df[df[k] == v]
    return df.copy()


def ci(frame, col):
    g = frame.groupby("iteration")[col].agg(["mean", "std", "count"])
    g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    return g.reset_index()


def limits(frames):
    """Shared per-group axis limits: the span of mean ± CI half-width of every
    frame in the group separately (pooling the frames first would average
    policies against each other and clip the extreme curves)."""
    def span(cols, pad=0.06):
        los, his = [], []
        for f in frames:
            d = f[f.iteration >= 1]
            for c in cols:
                g = ci(d, c)
                los.append(float((g["mean"] - g.hw).min()))
                his.append(float((g["mean"] + g.hw).max()))
        lo, hi = min(los), max(his)
        m = (hi - lo) * pad
        return (lo - m, hi + m)
    return span([f"{G}_mae", f"{R}_mae"]), span([f"{G}_err_in", f"{R}_err_in"])


def xticks(d):
    mx = int(d.iteration.max())
    return range(1, mx + 1, 2 if mx > 12 else 1)


def fig_curves(df, name, ylim_mae, ylim_in):
    d = df[df.iteration >= 1]
    panels = ((f"{G}_mae", f"{R}_mae", "whole-area MAE", ylim_mae),
              (f"{G}_err_in", f"{R}_err_in", "MAE inside the original weakspot", ylim_in))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for ax, (gcol, rcol, ylab, ylim) in zip(axes, panels):
        for col, lab, c in ((gcol, "guided", "#2ca02c"), (rcol, "random", "#d62728")):
            s = ci(d, col)
            ax.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                            color=c, alpha=0.15, lw=0)
            ax.plot(s.iteration, s["mean"], "-o", color=c, ms=3.5, lw=1.8, label=lab)
        ax.set_xlabel("iteration"); ax.set_ylabel(ylab); ax.set_ylim(*ylim)
        ax.set_xticks(xticks(d))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"fig2_{name}_curves.png"); plt.close(fig)


def fig_migration(df, name, seed=None):
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
        ax.add_patch(Ellipse((cx[i], cy[i]), 2 * smaj, 2 * smin,
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

    d = df[df.iteration > 0]
    for col, lab, c in (("det_iou", "IoU with induced gap", "#1f77b4"),
                        ("det_area", "detected area (fraction of square)", "#8c564b"),
                        ("det_distance", "distance to true centre", "#d62728")):
        g = ci(d, col)
        ax2.plot(g.iteration, g["mean"], "-o", ms=2.6, lw=1.4, color=c, label=lab)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("value")
    mx = int(d.iteration.max())
    ax2.set_xticks(range(0, mx + 1, 2 if mx > 12 else 1))
    ax2.legend(fontsize=6.8, ncol=1, loc="upper right", framealpha=0.9)

    fig.tight_layout(); fig.savefig(OUT / f"fig3_{name}_migration.png"); plt.close(fig)


# ── the storyline sets, grouped so limits are shared where comparison matters ──
SEV = "Adaptive (weakspot severity)"
groups = [
    # (group label, [(name, dataframe)])
    ("anchor K20", [
        ("S1_anchor_constant", load("iter20_anchor")),
    ]),
    ("schedules K8 (no scarcity)", [
        ("S5a_exp_dump", load("iter_mix_rate", mix_schedule="Exponential decay",
                              mix_rate=0.1, mix_ratio=1.0)),
        ("S5b_adaptive_severity", load("iter_mix_adaptation", mix_schedule=SEV,
                                       mix_ratio=0.5)),
    ]),
    ("undertrained K20", [
        ("S6_undertrained_i3n50", load("iter20_anchor_i3n50")),
    ]),
    ("scarcity K8, four policies (30 seeds)", [
        ("S7a_scarcity_constant", load("iter_money_policies", mix_schedule="Constant",
                                       radius=0.25, sel_sigma=0.15)),
        ("S7b_scarcity_exp_dump", load("iter_money_policies", mix_schedule="Exponential decay",
                                       radius=0.25, sel_sigma=0.15)),
        ("S7c_scarcity_adaptive_severity", load("iter_money_policies", mix_schedule=SEV,
                                                radius=0.25, sel_sigma=0.15)),
        ("S7d_scarcity_adaptive_size", load("iter_money_policies",
                                            mix_schedule="Adaptive (weakspot size)",
                                            radius=0.25, sel_sigma=0.15)),
    ]),
    ("scarcity K20 severity gate", [
        ("S8_scarcity_severity_K20", load("iter_money_longrun", mix_schedule=SEV,
                                          radius=0.25, sel_sigma=0.15)),
    ]),
]

for label, items in groups:
    ylim_mae, ylim_in = limits([d for _, d in items])
    for name, d in items:
        fig_curves(d, name, ylim_mae, ylim_in)
        fig_migration(d, name)
        print(f"{name}: {d.seed.nunique()} seeds, K={int(d.iteration.max())}  [{label}]")

print("\nwritten to", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name)
