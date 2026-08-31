"""The Early Stopping section's figure: the three policies under early stopping
in the standard setting, in the setting-figure layout minus the migration row -
one row of six panels (whole-area MAE, in-weakspot MAE per policy)."""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training")
PAPER = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Documents\Paper_DataSelectiveTrainingOnWeakspots\figures")
OUT = RES / "figures" / "Storyline"

G, R = "gnew", "rnew"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
SEV = "Adaptive (weakspot severity)"
POLICIES = [("Constant", "static mix"),
            ("Exponential decay", "dynamic decrease"),
            (SEV, "adaptive decrease")]

df = pd.read_csv(RES / "sweep__iter_es_normal.csv", low_memory=False)
df = df[df["iteration"] >= 1]


def ci(frame, col):
    g = frame.groupby("iteration")[col].agg(["mean", "std", "count"])
    g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    return g.reset_index()


def span(cols, pad=0.06):
    los, his = [], []
    for pol, _ in POLICIES:
        d = df[df["mix_schedule"] == pol]
        for c in cols:
            g = ci(d, c)
            los.append(float((g["mean"] - g.hw).min()))
            his.append(float((g["mean"] + g.hw).max()))
    lo, hi = min(los), max(his)
    m = (hi - lo) * pad
    return (lo - m, hi + m)


ylim_mae = span([f"{G}_mae", f"{R}_mae"])
ylim_in = span([f"{G}_err_in", f"{R}_err_in"])

fig = plt.figure(figsize=(13.2, 2.9))
gs = fig.add_gridspec(1, 6, wspace=0.42, left=0.05, right=0.985,
                      top=0.80, bottom=0.16)
for j, (pol, lab) in enumerate(POLICIES):
    d = df[df["mix_schedule"] == pol]
    for k, (gc, rc, ylab, ylim) in enumerate(
            ((f"{G}_mae", f"{R}_mae", "whole-area MAE", ylim_mae),
             (f"{G}_err_in", f"{R}_err_in", "in-weakspot MAE", ylim_in))):
        ax = fig.add_subplot(gs[0, 2 * j + k])
        for col, l, c in ((gc, "guided", "#2ca02c"), (rc, "random", "#d62728")):
            s = ci(d, col)
            ax.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                            color=c, alpha=0.15, lw=0)
            ax.plot(s.iteration, s["mean"], "-o", color=c, ms=2.6, lw=1.4, label=l)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(1, 9, 2))
        ax.set_title(ylab, fontsize=7.5)
        ax.set_xlabel("iteration", fontsize=7)
        if j == 0 and k == 0:
            ax.legend(fontsize=6.5, framealpha=0.9)
            ax.set_ylabel("MAE")
    x_mid = (fig.axes[2 * j].get_position().x0
             + fig.axes[2 * j + 1].get_position().x1) / 2
    fig.text(x_mid, 0.93, lab, ha="center", fontsize=10, fontweight="bold")

for t in (OUT, PAPER):
    fig.savefig(t / "fig_setting_earlystopping.png", dpi=160)
plt.close(fig)
print("fig_setting_earlystopping.png written")
