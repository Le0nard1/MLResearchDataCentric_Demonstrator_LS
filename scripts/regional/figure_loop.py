"""
Figure 1 of Paper C: the loop on one campaign, step by step, and its result against a Sobol
design of the same budget. Two-dimensional landscape, n = 1000, half for exploration.

    (a) estimate   the representative stream and the landscape estimated from it
    (b) narrow     the tests of each round and the box that steers the next round
    (c) allocate   the partition built on the exploration, shaded by the confirmation tests
                   each region receives
    (d) report     the labels from the confirmation tests
    (e) baseline   a Sobol design of the same budget, with the same partition and report

    python -m scripts.regional.figure_loop --out figures/loop.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, Patch, Rectangle

from scripts.regional.campaign import interpolate
from scripts.regional.problems import make
from scripts.regional.report import run_report

SURFACE, INK, MUTED, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#a9a8a3"
CLEARED, FLAGGED, UNDET = "#9ec5f4", "#eb6834", "#e4e3df"
BLUES = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
ROUNDS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal: round 1 to 4
N_GRID = 300


def grid():
    g = np.linspace(0.5 / N_GRID, 1 - 0.5 / N_GRID, N_GRID)
    return g, np.array(np.meshgrid(g, g)).reshape(2, -1).T


def frame(ax, title):
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED), s.set_linewidth(0.6)
    ax.set_title(title, fontsize=7, color=INK, loc="left", linespacing=1.25)


def outline(ax, problem):
    g, G = grid()
    bad = (problem.mean_failure(G) > 0.1).reshape(N_GRID, N_GRID).astype(float)
    ax.contour(g, g, bad, levels=[0.5], colors=INK, linewidths=1.2)


def cells(ax, fam):
    g, G = grid()
    ids = fam.ids(G).reshape(N_GRID, N_GRID)
    ax.contour(g, g, ids, levels=np.unique(ids) + 0.5, colors=SURFACE, linewidths=0.9)
    return ids


def labels(ax, fam, result):
    g, G = grid()
    ids = fam.ids(G)
    lab = np.array([{"cleared": 0, "undetermined": 1, "flagged": 2}[result.get(int(c), ("undetermined",))[0]]
                    for c in ids]).reshape(N_GRID, N_GRID)
    ax.imshow(lab, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=2, interpolation="nearest",
              cmap=ListedColormap([CLEARED, UNDET, FLAGGED]))
    ax.contourf(g, g, (lab == 2).astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    cells(ax, fam)


def score(out):
    return f"undet. {out['undetermined_mass']:.2f}, found {out['recall']:.2f}\nhidden {out['hidden_bad']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="rosenbrock_low")
    ap.add_argument("--seed", type=int, default=9809)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--strategy", default="weakspot")
    ap.add_argument("--out", default="figure_loop.pdf")
    a = ap.parse_args()
    plt.rcParams["hatch.linewidth"] = 0.5
    problem = make(a.regime, 2, a.seed)
    trace = []
    out, fam, result, X, e, got = run_report(problem, a.n, a.strategy, "cart", a.seed, details=True, trace=trace)
    *boxes, info = trace
    n_rep = info["n_rep"]
    fig, axes = plt.subplots(1, 5, figsize=(6.75, 2.05), facecolor=SURFACE)

    ax = axes[0]                                         # (a) estimate
    g, G = grid()
    land = interpolate(X[:n_rep], e[:n_rep], G, 25).reshape(N_GRID, N_GRID)
    ax.imshow(land, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=max(land.max(), 0.2),
              cmap=LinearSegmentedColormap.from_list("seq", BLUES), interpolation="bilinear")
    rep, fail = X[:n_rep], e[:n_rep] > 0.5
    ax.scatter(*rep[~fail].T, s=2, c=MUTED, linewidths=0)
    ax.scatter(*rep[fail].T, s=5, c=INK, marker="x", linewidths=0.6)
    outline(ax, problem)
    frame(ax, f"(a) Estimate\n{n_rep} tests from $Q$\nand the landscape")

    ax = axes[1]                                         # (b) narrow
    starts = [b[1] for b in boxes] + [len(X)]
    ax.scatter(*rep.T, s=2, c=FAINT, linewidths=0)
    for t, (box, _) in enumerate(boxes):
        Xt = X[starts[t]:starts[t + 1]]
        ax.scatter(*Xt.T, s=2, c=ROUNDS[t], linewidths=0)
        if hasattr(box, "radius"):                       # hard region: a ball, a circle in 2-D
            ax.add_patch(Circle(box.centre, box.radius, fill=False, ec=ROUNDS[t], lw=1.0))
        else:                                            # box peeling: a box
            lo, hi = box.low, np.minimum(box.high, 1)
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=ROUNDS[t], lw=1.0))
    outline(ax, problem)
    frame(ax, f"(b) Narrow\n{len(X) - n_rep} guided tests\nin {len(boxes)} rounds")

    ax = axes[2]                                         # (c) allocate
    alloc = info["alloc"]
    ids = fam.ids(G)
    level = alloc[ids].reshape(N_GRID, N_GRID).astype(float)
    ax.imshow(level, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=max(level.max(), 1),
              cmap=LinearSegmentedColormap.from_list("seq", BLUES), interpolation="nearest")
    idg = cells(ax, fam)
    for c in np.flatnonzero(alloc):
        ys, xs = np.nonzero(idg == c)
        if len(xs) > 2500:                               # label only regions large enough to read
            ax.text(xs.mean() / N_GRID, ys.mean() / N_GRID, str(alloc[c]), fontsize=5.5, color=INK,
                    ha="center", va="center")
    outline(ax, problem)
    frame(ax, f"(c) Allocate\n{len(alloc)} regions, {int(alloc.sum())}\nconfirmation tests")

    ax = axes[3]                                         # (d) report
    labels(ax, fam, result)
    Xc = np.vstack(list(got.values())) if got else np.zeros((0, 2))
    ax.scatter(*Xc.T, s=1.5, c=MUTED, linewidths=0)
    outline(ax, problem)
    frame(ax, "(d) Report\n" + score(out))

    ax = axes[4]                                         # (e) baseline
    out_s, fam_s, result_s, Xs, _, _ = run_report(make(a.regime, 2, a.seed), a.n, "sobol", "cart", a.seed,
                                                  details=True)
    labels(ax, fam_s, result_s)
    ax.scatter(*Xs.T, s=1.5, c=MUTED, linewidths=0)
    outline(ax, problem)
    frame(ax, "(e) Sobol, same $n$\n" + score(out_s))

    fig.legend(handles=[plt.Line2D([], [], color=INK, lw=1.2, label="failing set"),
                        plt.Line2D([], [], ls="", marker="x", color=INK, ms=3, label="failure"),
                        Patch(facecolor=BLUES[4], label="estimate / tests allocated"),
                        *[Patch(facecolor="none", edgecolor=ROUNDS[t], label=f"round {t + 1}") for t in (0, 3)],
                        Patch(facecolor=FLAGGED, hatch="////", label="flagged"),
                        Patch(facecolor=CLEARED, label="cleared"),
                        Patch(facecolor=UNDET, label="undetermined")],
               loc="lower center", ncol=8, fontsize=6, frameon=False, handlelength=1.3,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.75, bottom=0.11, wspace=0.05)
    fig.savefig(a.out, facecolor=SURFACE)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=220, facecolor=SURFACE)
    print("saved", a.out, out, out_s)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
