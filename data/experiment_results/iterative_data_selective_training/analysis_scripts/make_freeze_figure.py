"""Frozen vs unfrozen under identical conditions (scarce-pool setting):
per-iteration guided-minus-random gap for the three policies, first hidden
layer frozen against the unfrozen protocol."""
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

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
SEV = "Adaptive (weakspot severity)"
POLICIES = [("Constant", "static mix"),
            ("Exponential decay", "dynamic decrease"),
            (SEV, "adaptive decrease")]

df = pd.read_csv(RES / "sweep__iter_freeze_scarce.csv", low_memory=False)
df = df[df["iteration"] >= 1]


def ci(frame):
    g = frame.groupby("iteration")["gap_mae"].agg(["mean", "std", "count"])
    g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    return g.reset_index()


fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), sharey=True)
for ax, (pol, lab) in zip(axes, POLICIES):
    for frozen, l, c in ((False, "unfrozen", "#1f77b4"),
                         (True, "first layer frozen", "#ff7f0e")):
        s = ci(df[(df["mix_schedule"] == pol) & (df["freeze_first"] == frozen)])
        ax.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                        color=c, alpha=0.15, lw=0)
        ax.plot(s.iteration, s["mean"], "-o", color=c, ms=3, lw=1.6, label=l)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(lab, fontsize=9, fontweight="bold")
    ax.set_xlabel("iteration")
    ax.set_xticks(range(1, 9))
axes[0].set_ylabel("gap (guided − random MAE)")
axes[0].legend(fontsize=7, framealpha=0.9)
fig.tight_layout()
for t in (OUT, PAPER):
    fig.savefig(t / "fig_freeze_compare.png", dpi=160)
plt.close(fig)
print("fig_freeze_compare.png written")
