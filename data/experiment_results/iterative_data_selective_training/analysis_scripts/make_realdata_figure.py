"""Real-data example figure: (a) guided vs random eval MAE under the induced
category deficit with the one-hot encoding (static mix); (b) per-iteration gap
for the three categorical encodings under the deficit."""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\real_data_example\real_runs.csv")
PAPER = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Documents\Paper_DataSelectiveTrainingOnWeakspots\figures")
OUT = CSV.parent

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

df = pd.read_csv(CSV)
d = df[(df["setting"] == "deficit") & (df["policy"] == "static")]


def ci(frame, col):
    g = frame.groupby("iteration")[col].agg(["mean", "std", "count"])
    g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    return g.reset_index()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

oh = d[(d["encoding"] == "onehot") & (d["iteration"] >= 1)]
for col, lab, c in (("mae_g", "guided", "#2ca02c"), ("mae_r", "random", "#d62728")):
    s = ci(oh, col)
    ax1.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                     color=c, alpha=0.15, lw=0)
    ax1.plot(s.iteration, s["mean"], "-o", color=c, ms=3, lw=1.6, label=lab)
ax1.set_xlabel("iteration"); ax1.set_ylabel("evaluation MAE")
ax1.set_title("(a) one-hot encoding, static mix", fontsize=8.5)
ax1.set_xticks(range(1, 9)); ax1.legend(fontsize=7)

for enc, lab, c in (("numeric", "numeric (categories dropped)", "#7f7f7f"),
                    ("onehot", "one-hot vector distance", "#1f77b4"),
                    ("stratified", "stratified quotas", "#9467bd")):
    s = ci(d[(d["encoding"] == enc) & (d["iteration"] >= 1)], "gap")
    ax2.fill_between(s.iteration, s["mean"] - s.hw, s["mean"] + s.hw,
                     color=c, alpha=0.12, lw=0)
    ax2.plot(s.iteration, s["mean"], "-o", color=c, ms=3, lw=1.6, label=lab)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xlabel("iteration"); ax2.set_ylabel("gap (guided − random MAE)")
ax2.set_title("(b) the three encodings, static mix", fontsize=8.5)
ax2.set_xticks(range(1, 9)); ax2.legend(fontsize=6.5)

fig.tight_layout()
for t in (OUT, PAPER):
    fig.savefig(t / "fig_realdata.png", dpi=160)
plt.close(fig)
print("fig_realdata.png written")
