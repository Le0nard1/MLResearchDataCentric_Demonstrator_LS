"""The paper's three composite figures, one per experimental setting.

Each figure is a 2x6 grid: the top row holds, for each of the three mix
policies (static / dynamic decrease / adaptive decrease), a pair of
guided-vs-random panels (whole-area MAE, in-weakspot MAE); the bottom row puts
the corresponding weakspot-migration map directly beneath each policy's pair.
Y-limits are shared within a figure so the three policies read on one scale.

Also prints the 3x3 across-loop gap table (policy x setting) for the paper.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import LineCollection

RES = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training")
OUT = RES / "figures" / "Storyline"
PAPER = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Documents\Paper_DataSelectiveTrainingOnWeakspots\figures")

G, R = "gnew", "rnew"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

SEV = "Adaptive (weakspot severity)"
POLICIES = [("Constant", "static mix"),
            ("Exponential decay", "dynamic decrease"),
            (SEV, "adaptive decrease")]


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


def curves_panel(ax, d, col_g, col_r, ylim):
    for col, lab, c in ((col_g, "guided", "#2ca02c"), (col_r, "random", "#d62728")):
        s = ci(d[d.iteration >= 1], col)
        ax.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                        color=c, alpha=0.15, lw=0)
        ax.plot(s.iteration, s["mean"], "-o", color=c, ms=2.6, lw=1.4, label=lab)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(1, int(d.iteration.max()) + 1, 2))


def migration_panel(ax, df, seed=42):
    if seed not in set(df.seed.unique()):
        seed = int(sorted(df.seed.unique())[0])
    s = df[(df.seed == seed) & (df.iteration > 0)].sort_values("iteration")
    cx, cy = s.det_cx.to_numpy(), s.det_cy.to_numpy()
    n = len(cx)
    cmap = plt.get_cmap("viridis")
    r = float(df.radius.iloc[0])
    ax.add_patch(Circle((float(df.center_x.iloc[0]), float(df.center_y.iloc[0])), r,
                        fill=False, ec="red", lw=1.8, ls="--", zorder=6))
    for i in range(n):
        smaj, smin = s.det_smaj.iloc[i], s.det_smin.iloc[i]
        if not (np.isfinite(smaj) and np.isfinite(smin)):
            continue
        ax.add_patch(Ellipse((cx[i], cy[i]), 2 * smaj, 2 * smin,
                             angle=np.degrees(s.det_angle.iloc[i]),
                             fill=False, ec=cmap(i / max(n - 1, 1)), ls="--",
                             lw=0.8, alpha=0.7, zorder=2))
    pts = np.column_stack([cx, cy]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(1, n), lw=1.4,
                        alpha=0.85, zorder=5)
    lc.set_array(np.arange(1, n)); ax.add_collection(lc)
    sc = ax.scatter(cx, cy, c=np.arange(1, n + 1), cmap=cmap, s=30,
                    edgecolor="black", lw=0.5, zorder=6)
    seen = {}
    for i in range(n):
        key = (round(cx[i], 3), round(cy[i], 3))
        k = seen.get(key, 0); seen[key] = k + 1
        ax.annotate(str(i + 1), (cx[i], cy[i]), fontsize=5.5,
                    xytext=(0, 5 + 7 * k), textcoords="offset points",
                    ha="center", zorder=7, color=cmap(i / max(n - 1, 1)),
                    fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    return sc


# Fixed y-limits, identical for every policy AND every setting figure, so all
# nine panels of each metric read on one scale. Tops are pinned by request
# (whole-area at 0.5, in-weakspot at 1.0); bottoms hug the global data minimum.
YLIM_MAE = None
YLIM_IN = None


def global_limits(all_frames):
    def gmin(cols):
        lo = []
        for d in all_frames:
            dd = d[d.iteration >= 1]
            for c in cols:
                g = ci(dd, c)
                lo.append(float((g["mean"] - g.hw).min()))
        return min(lo)
    lo_mae = gmin([f"{G}_mae", f"{R}_mae"])
    lo_in = gmin([f"{G}_err_in", f"{R}_err_in"])
    return ((lo_mae - 0.05 * (0.5 - lo_mae), 0.5),
            (lo_in - 0.05 * (1.0 - lo_in), 1.0))


def make_setting_figure(name, frames):
    """frames: {policy_name: df} in POLICIES order."""
    dfs = [frames[p] for p, _ in POLICIES]
    ylim_mae, ylim_in = YLIM_MAE, YLIM_IN

    fig = plt.figure(figsize=(13.2, 7.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.55], hspace=0.34,
                          wspace=0.42, left=0.05, right=0.945, top=0.90,
                          bottom=0.06)
    sc = None
    for j, ((pol, lab), d) in enumerate(zip(POLICIES, dfs)):
        ax1 = fig.add_subplot(gs[0, 2 * j])
        curves_panel(ax1, d, f"{G}_mae", f"{R}_mae", ylim_mae)
        ax1.set_title("whole-area MAE", fontsize=7.5)
        if j == 0:
            ax1.legend(fontsize=6.5, framealpha=0.9)
            ax1.set_ylabel("MAE")
        ax2 = fig.add_subplot(gs[0, 2 * j + 1])
        curves_panel(ax2, d, f"{G}_err_in", f"{R}_err_in", ylim_in)
        ax2.set_title("in-weakspot MAE", fontsize=7.5)
        axm = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        sc = migration_panel(axm, d)
        axm.set_xlabel("x₁", fontsize=7)
        if j == 0:
            axm.set_ylabel("x₂", fontsize=7)
        # column header centred over the policy's pair of curve panels
        x_mid = (ax1.get_position().x0 + ax2.get_position().x1) / 2
        fig.text(x_mid, 0.955, lab, ha="center", fontsize=10, fontweight="bold")

    cax = fig.add_axes([0.955, 0.06, 0.012, 0.47])
    fig.colorbar(sc, cax=cax, label="iteration")
    for tgt in (OUT, PAPER):
        fig.savefig(tgt / f"fig_setting_{name}.png", dpi=160)
    plt.close(fig)
    print(f"fig_setting_{name}.png written")


# ── assemble the three settings ────────────────────────────────────────────
settings = {
    "normal": {p: load("iter_paper_normal", mix_schedule=p) for p, _ in POLICIES},
    "lowpretrain": {p: load("iter_paper_lowpretrain", mix_schedule=p) for p, _ in POLICIES},
    "scarcity": {p: load("iter_money_policies", mix_schedule=p,
                         radius=0.25, sel_sigma=0.15) for p, _ in POLICIES},
}

print("=== 3x3 summary: across-loop gap (guided - random), per policy x setting ===")
rows = []
for sname, frames in settings.items():
    for pol, lab in POLICIES:
        d = frames[pol]
        v = d[d.iteration > 0].groupby("seed")["gap_mae"].mean()
        t, p = stats.ttest_1samp(v.values, 0.0)
        rows.append(dict(setting=sname, policy=lab, gap=v.mean(),
                         sem=stats.sem(v.values), p=p,
                         ahead=np.mean(v.values < 0), n=len(v)))
tab = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(tab.round(4).to_string(index=False))

YLIM_MAE, YLIM_IN = global_limits(
    [d for frames in settings.values() for d in frames.values()])
print(f"fixed y-limits: whole-area {tuple(round(v, 3) for v in YLIM_MAE)}, "
      f"in-weakspot {tuple(round(v, 3) for v in YLIM_IN)}")
for sname, frames in settings.items():
    make_setting_figure(sname, frames)
