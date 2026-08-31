"""3-D error landscape over the loop (real data, ideal setting).

Four 3-D panels covering three iterations: the initial landscape with round
1's detected weakspot and data selection, the landscape after round 1 with
round 2's detection and selection, likewise after round 2, and the landscape
after round 3. Height (z) is the kNN-smoothed error surface the detector
operates on, triangulated over the first two principal components; z is on a
SHARED absolute scale so the flattening of the mountain is literal, while
colour is per-panel so the current peaks stay visible as the scale drops.
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
from matplotlib.colors import PowerNorm
import numpy as np
from sklearn.decomposition import PCA

from scripts.realdata import loop_real as R

PAPER = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Documents\Paper_DataSelectiveTrainingOnWeakspots\figures")
OUT = Path("data/experiment_results/real_data_example")

plt.rcParams.update({"figure.dpi": 150, "font.size": 8})

res = R.run_real(dict(seed=42, encoding="onehot", policy="static",
                      deficit_col="species", deficit_val="virginica",
                      deficit_frac=0.9), keep_rounds=True)
rd = res["rounds"]
Xe = rd["X_eval_dist"]
pca = PCA(n_components=2).fit(Xe)
P = pca.transform(Xe)

k = 15
d2 = np.linalg.norm(Xe[:, None, :] - Xe[None, :, :], axis=2)
nn = np.argsort(d2, axis=1)[:, :k]
Z = [rd["err_eval"][it][nn].mean(axis=1) for it in range(len(rd["err_eval"]))]
zmax = float(np.quantile(Z[0], 0.99)) * 1.05


def nearest_z(z, pts2):
    """Surface height at arbitrary PC coordinates (nearest eval point)."""
    i = np.argmin(np.linalg.norm(P[None, :, :] - pts2[:, None, :], axis=2), axis=1)
    return z[i]


SIGMA = 0.5 * rd["med_dist"]                     # the run's selection-kernel width


def weakspot_ellipse(it):
    """2σ ellipse of the identified region in PC space: the weakspot members
    (top local-error quantile) weighted by the selection kernel around the
    detected centre, summarised by their weighted mean and covariance."""
    z = Z[it]
    members = z >= np.quantile(z, 0.85)
    w = np.exp(-0.5 * (np.linalg.norm(Xe - rd["centers"][it], axis=1) / SIGMA) ** 2)
    u = w * members
    if u.sum() < 1e-9:
        return None
    mu = (u @ P) / u.sum()
    d = P - mu
    cov = (u[:, None] * d).T @ d / u.sum()
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 120)
    circ = np.stack([np.cos(t), np.sin(t)])
    return mu, (vecs * (2.0 * np.sqrt(np.maximum(vals, 1e-12)))) @ circ + mu[:, None]


# Top row: 3-D surfaces, rotated so the ridge no longer runs along the line of
# sight. Bottom row: the same states top-down (filled triangulation), where the
# geometry of detection and selection is unambiguous. ONE colour scale is
# shared by every panel (anchored to the initial landscape), so the later
# panels go dark as the errors fall and the progression is read off the colour
# alone.
# Square-root colour scale (PowerNorm γ=0.5): still one shared scale anchored
# to the initial landscape, but with the low range stretched so the later
# panels' residual hills stay differentiated instead of vanishing into black.
norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(np.quantile(Z[0], 0.97)))
fig = plt.figure(figsize=(14.4, 7.0))
tp = None
for panel, it in enumerate([0, 1, 2, 3]):
    z = Z[it]
    if it < 3:
        c2 = pca.transform(rd["centers"][it][None, :])
        pk = pca.transform(rd["picks_dist"][it])
    title = (f"initial model — MAE {res['mae_g'][0]:.1f}" if it == 0
             else f"after iteration {it} — MAE {res['mae_g'][it]:.1f}")
    if it < 3:
        title += f"\ndetection + selection of iteration {it + 1}"

    ax = fig.add_subplot(2, 4, panel + 1, projection="3d")
    ax.plot_trisurf(P[:, 0], P[:, 1], z, cmap="inferno", norm=norm,
                    linewidth=0.08, edgecolor="k", alpha=0.72)
    if it < 3:
        cz = nearest_z(z, c2)[0]
        ax.scatter(c2[0, 0], c2[0, 1], cz + 0.10 * zmax, marker="D", s=110,
                   color="#d62728", edgecolor="white", lw=1.0, zorder=10,
                   label="detected weakspot", depthshade=False)
        pz = nearest_z(z, pk)
        ax.scatter(pk[:, 0], pk[:, 1], pz + 0.05 * zmax, marker="o", s=17,
                   color="#00d0ff", edgecolor="black", lw=0.4, zorder=9,
                   label="selected data", depthshade=False)
        ell = weakspot_ellipse(it)
        if ell is not None:
            _, E = ell
            ax.plot(E[0], E[1], np.zeros(E.shape[1]), ls="--", lw=1.3,
                    color="#d62728", alpha=0.9, zorder=8,
                    label="identified region (2σ)")
    ax.set_zlim(0, zmax)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("PC1", labelpad=-4); ax.set_ylabel("PC2", labelpad=-4)
    ax.set_zlabel("local error", labelpad=-2)
    ax.tick_params(pad=-2, labelsize=6)
    ax.view_init(elev=27, azim=-128)
    if panel == 0:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    ax2 = fig.add_subplot(2, 4, panel + 5)
    tp = ax2.tripcolor(P[:, 0], P[:, 1], z, cmap="inferno", norm=norm,
                       shading="gouraud")
    if it < 3:
        ax2.scatter(c2[0, 0], c2[0, 1], marker="D", s=80, color="#d62728",
                    edgecolor="white", lw=0.9, zorder=6)
        ax2.scatter(pk[:, 0], pk[:, 1], marker="o", s=13, color="#00d0ff",
                    edgecolor="black", lw=0.4, zorder=5)
        ell = weakspot_ellipse(it)
        if ell is not None:
            _, E = ell
            ax2.plot(E[0], E[1], ls="--", lw=1.3, color="#d62728",
                     alpha=0.95, zorder=6)
    ax2.set_xlabel("PC1", fontsize=7)
    if panel == 0:
        ax2.set_ylabel("PC2", fontsize=7)
    ax2.set_title("top-down view", fontsize=7.5)
    ax2.tick_params(labelsize=6)
fig.tight_layout(pad=1.3, rect=[0, 0, 0.945, 1])
cax = fig.add_axes([0.952, 0.09, 0.011, 0.82])
cb = fig.colorbar(tp, cax=cax)
cb.set_label("local error (shared scale)", fontsize=8, labelpad=2)
cb.ax.tick_params(labelsize=7)
for t in (OUT, PAPER):
    fig.savefig(t / "fig_realdata_3d.png", dpi=160)
plt.close(fig)
print("fig_realdata_3d.png written;",
      "MAE:", [round(v, 1) for v in res["mae_g"][:4]])
