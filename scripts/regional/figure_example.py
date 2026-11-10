"""
Figure 1 of Paper C: one campaign on a two-dimensional landscape, n = 1000.

    (a) a Sobol design and its report, (b) box peeling and its report.

Each panel shows the failing set (outline), the tests (dots) and the report: regions flagged
(orange, hatched), cleared (blue) or undetermined (grey). The seed is printed in the caption.

    python -m scripts.regional.figure_example --seed 9800
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scripts.regional.problems import make
from scripts.regional.report import run_report

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
CLEARED, FLAGGED, UNDET = "#9ec5f4", "#eb6834", "#e4e3df"


def panel(ax, problem, strategy, seed, n, title):
    out, fam, cells, X, e, got = run_report(problem, n, strategy, "cart", seed, details=True)
    g = np.linspace(0.0025, 0.9975, 400)
    G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    ids = fam.ids(G)
    lab = np.array([{"cleared": 0, "undetermined": 1, "flagged": 2}[cells.get(int(c), ("undetermined",))[0]]
                    for c in ids]).reshape(400, 400)
    ax.imshow(lab, origin="lower", extent=(0, 1, 0, 1), cmap=ListedColormap([CLEARED, UNDET, FLAGGED]),
              vmin=0, vmax=2, interpolation="nearest")
    ax.contourf(g, g, (lab == 2).astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(g, g, ids.reshape(400, 400), levels=np.unique(ids) + 0.5, colors=SURFACE, linewidths=1.0)
    bad = (problem.mean_failure(G) > 0.1).reshape(400, 400).astype(float)
    ax.contour(g, g, bad, levels=[0.5], colors=INK, linewidths=1.4)
    ax.scatter(X[:, 0], X[:, 1], s=2.5, c=MUTED, linewidths=0)
    Xc = np.vstack([v for v in got.values()]) if got else np.zeros((0, 2))
    ax.scatter(Xc[:, 0], Xc[:, 1], s=2.5, c=INK, marker="x", linewidths=0.4)
    ax.set_title(f"{title}\nundet. {out['undetermined_mass']:.2f}, found {out['recall']:.2f}, "
                 f"hidden {out['hidden_bad']:.2f}", fontsize=6.5, color=INK, linespacing=1.3)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED), s.set_linewidth(0.6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="rosenbrock_low")
    ap.add_argument("--seed", type=int, default=9801)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="figure_example.pdf")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.0), facecolor=SURFACE)
    for ax, (s, t) in zip(axes, [("sobol", "(a) Sobol design"), ("peel", "(b) Box peeling")]):
        panel(ax, make(a.regime, 2, a.seed), s, a.seed, a.n, t)
    fig.legend(handles=[Patch(facecolor=FLAGGED, hatch="////", label="flagged"),
                        Patch(facecolor=CLEARED, label="cleared"),
                        Patch(facecolor=UNDET, label="undetermined"),
                        plt.Line2D([], [], color=INK, lw=1.4, label="failing set")],
               loc="lower center", ncol=4, fontsize=6.5, frameon=False, handlelength=1.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.12, wspace=0.06)
    fig.savefig(a.out, facecolor=SURFACE)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, facecolor=SURFACE)
    print("saved", a.out)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
