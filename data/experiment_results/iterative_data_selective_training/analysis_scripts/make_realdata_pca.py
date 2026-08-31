"""PCA view of the real-data error landscape over the loop.

One run at the ideal setting (category deficit, one-hot distance, static mix,
seed 42): the eval set projected onto its first two principal components in
the standardised distance space, coloured by the guided model's absolute
error, at six snapshots of the loop. The detected weakspot centre (star) and
that round's guided picks (crosses) are overlaid, so the panels read as the
storyline: an elevated region appears, a round of guided selection bumps it
down, the detector moves to the next one.
"""
import sys, os, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
APP = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application"
sys.path.insert(0, APP)
os.chdir(APP)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from scripts.realdata import loop_real as R

PAPER = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Documents\Paper_DataSelectiveTrainingOnWeakspots\figures")
OUT = Path("data/experiment_results/real_data_example")

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False})

res = R.run_real(dict(seed=42, encoding="onehot", policy="static",
                      deficit_col="species", deficit_val="virginica",
                      deficit_frac=0.9), keep_rounds=True)
rd = res["rounds"]
Xe, cats = rd["X_eval_dist"], rd["cats_eval"]
pca = PCA(n_components=2).fit(Xe)
P = pca.transform(Xe)
evr = pca.explained_variance_ratio_

# Colour by the same kNN-smoothed local error the detector operates on — the
# raw per-point error is speckly, the smoothed surface is the landscape.
k = 15
d2 = np.linalg.norm(Xe[:, None, :] - Xe[None, :, :], axis=2)
nn = np.argsort(d2, axis=1)[:, :k]

SNAPS = [0, 1, 2, 4, 6, 8]
fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), sharex=True, sharey=True)
sc = None
for ax, it in zip(axes.ravel(), SNAPS):
    err = rd["err_eval"][it][nn].mean(axis=1)
    vmax = np.quantile(err, 0.97)
    order = np.argsort(err)                      # draw high errors on top
    sc = ax.scatter(P[order, 0], P[order, 1], c=err[order], cmap="inferno",
                    s=16, vmin=0, vmax=vmax, edgecolors="none")
    if it < len(rd["centers"]):
        c2 = pca.transform(rd["centers"][it][None, :])[0]
        ax.scatter(*c2, marker="*", s=190, color="red", edgecolor="white",
                   lw=0.9, zorder=6, label="detected weakspot")
        pk = pca.transform(rd["picks_dist"][it])
        ax.scatter(pk[:, 0], pk[:, 1], marker="x", s=14, color="#00d0ff",
                   lw=0.9, zorder=5, label="guided picks")
    ax.set_title(f"iteration {it} — MAE {res['mae_g'][it]:.1f}"
                 + ("  (initial model)" if it == 0 else ""), fontsize=8.5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
for ax in axes[1]:
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)")
for ax in axes[:, 0]:
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)")
axes[0, 0].legend(fontsize=7, loc="lower left", framealpha=0.9)
fig.tight_layout()
for t in (OUT, PAPER):
    fig.savefig(t / "fig_realdata_pca.png", dpi=160)
plt.close(fig)

# console: which category the detection sits on, per round
for j, cw in enumerate(res["cat_weak"], start=1):
    top = max(cw, key=cw.get) if cw else "-"
    print(f"round {j}: top weakspot category {top} ({cw.get(top, 0):.0%}), "
          f"MAE after {res['mae_g'][j]:.2f}")
print("fig_realdata_pca.png written")
