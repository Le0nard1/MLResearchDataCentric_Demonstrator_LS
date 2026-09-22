"""
Generate the error-landscape figure for the paper
(``figures/fig_error_landscape.png``, referenced from the closing experiment
section of ``main.tex``).

The figure tells one round of the method as four maps over the same input square,
averaged over seeds and computed with a single detector:

    (a) error before      (b) detected weakspot
    (c) guided - before   (d) random - before      ((c) and (d) share one scale)

The difference maps are the point of the figure: they show *where* each retraining
spent an identical budget of new points. Guided selection empties the gap; the
random baseline spreads the same points over the whole square and leaves the gap
largely intact.

All computation lives in ``scripts/dataselect/error_landscape.py``, shared with the
Streamlit page's "Error Landscape" tab, so figure and app cannot diverge.

Usage
-----
    python -m scripts.generate_error_landscape_figure
    python -m scripts.generate_error_landscape_figure --seeds 50
    python -m scripts.generate_error_landscape_figure --detector "Peaks over Threshold (EVT)"
    python -m scripts.generate_error_landscape_figure --out some/other/path.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Allow running both as a module (-m) and as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dataselect import error_landscape as EL

APP = Path(__file__).resolve().parents[1]
RESULTS = APP / "data" / "experiment_results" / "data_selective_training"
FIGDIR = RESULTS / "figures"
PAPER_FIGDIR = (APP.parent / "Documents" / "Paper_DataSelectionOnModelWeakness" / "figures")

# The 50 seeds of the isolation experiments, so this study rests on the same seed
# set as Section "The Dataset Mix".
SEEDS = [42, 0, 7, 1, 3, 5, 11, 17, 23, 99, 2, 4, 6, 8, 13, 19, 29, 37, 53, 71,
         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
         115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]

C_TRUE = "#d62728"      # induced (ground-truth) weakspot
C_DET = "#ffffff"       # detected weakspot
CMAP_SEQ = "viridis"    # (a) error and (b) detection surface
CMAP_DIFF = "PuOr_r"    # (c), (d) change in error: purple removed, orange added

# One detector carries the whole study, so the figure shows what a single, concrete
# identification does rather than an average over methods that no practitioner runs.
# Of the broad sweep's five this is the one whose whole-area advantage lies closest to
# their mean (-0.062 against -0.069), so it is representative rather than favourable.
PAPER_DETECTOR = "kNN Performance Mapping"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=FIGDIR / "fig_error_landscape.png",
                   help="output PNG path")
    p.add_argument("--copy-to-paper", action="store_true", default=True,
                   help="also write the PNG into the paper's figures/ (default: on)")
    p.add_argument("--no-copy-to-paper", dest="copy_to_paper", action="store_false")
    p.add_argument("--seeds", type=int, default=len(SEEDS),
                   help=f"how many seeds to average over (default: {len(SEEDS)})")
    p.add_argument("--detector", action="append", default=None,
                   help=f"detector to use; repeatable. Default: {PAPER_DETECTOR}.")
    p.add_argument("--land-res", type=int, default=EL.LAND_RES_DEFAULT,
                   help=f"rendering-grid resolution (default: {EL.LAND_RES_DEFAULT})")
    p.add_argument("--dpi", type=int, default=200, help="output DPI (default: 200)")
    p.add_argument("--recompute", action="store_true",
                   help="ignore the cached study and re-run every round")
    return p.parse_args(argv)


# -------------------------------------------------------------
# Panel helpers
# -------------------------------------------------------------
def _frame(ax, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, pad=6)


def _map(ax, Z, cmap, vmin, vmax):
    """Draw a landscape as an image over the unit square."""
    return ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                     vmin=vmin, vmax=vmax, interpolation="bilinear")


def _true_circle(ax, center, radius, label=None):
    ax.add_patch(Circle(center, radius, fill=False, ec=C_TRUE, lw=1.8,
                        ls="--", zorder=5, label=label))


def panel_setup(ax, study):
    """Ground-truth landscape with the induced gap and the initial training points."""
    rep, c, r = study["rep"], study["center"], study["radius"]
    land_flat = np.column_stack([study["land_xx"].ravel(), study["land_yy"].ravel()])
    Z = EL.P.true_function(land_flat, n_bumps=int(study["params"]["n_bumps"]))
    ax.imshow(Z.reshape(study["land_xx"].shape), origin="lower", extent=(0, 1, 0, 1),
              cmap="Greys", alpha=0.55, interpolation="bilinear")
    if len(rep["X_excl"]):
        ax.scatter(rep["X_excl"][:, 0], rep["X_excl"][:, 1], s=4, c=C_TRUE,
                   alpha=0.35, linewidths=0, label="withheld points", zorder=3)
    ax.scatter(rep["X_tr0"][:, 0], rep["X_tr0"][:, 1], s=7, c="#1a1a1a",
               alpha=0.85, linewidths=0, label="initial training points", zorder=4)
    _true_circle(ax, c, r, label="induced gap")
    _frame(ax, "(a) Setup: induced weakspot")
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.85, handlelength=1.4,
              borderpad=0.3)


def _annotate(ax, land, study, fmt="{:+.2f}", inside_only=False):
    """Stamp the panel with its mean value inside (and outside) the induced gap.

    The maps differ subtly at this operating point - a wide kernel mixed half-and-half
    with uniform rehearsal is only mildly concentrated - so the numbers are what make
    the comparison readable rather than merely suggestive.
    """
    i, o = EL.region_means(land, study["land_xx"], study["land_yy"],
                           study["center"], study["radius"])
    txt = (f"in gap {fmt.format(i)}" if inside_only
           else f"in gap {fmt.format(i)}\noutside {fmt.format(o)}")
    ax.text(0.985, 0.015, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#111111", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#999999",
                      alpha=0.88, lw=0.5))


def panel_detected(ax, study, title):
    """Mean detector surface with the detected extent and the induced weakspot."""
    xx = study["xx"]
    im = ax.imshow(study["surf_mean"].reshape(xx.shape), origin="lower",
                   extent=(0, 1, 0, 1), cmap=CMAP_SEQ, vmin=0, vmax=1,
                   interpolation="bilinear")
    ext = study["rep"]["ext"]
    if ext is not None:
        ax.plot(ext["ellipse_x"], ext["ellipse_y"], color=C_DET, lw=1.8, zorder=6,
                label="detected extent (2$\\sigma$)")
    _true_circle(ax, study["center"], study["radius"], label="induced weakspot")
    _frame(ax, title)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85, handlelength=1.4,
              borderpad=0.3)
    ax.text(0.985, 0.015,
            f"distance {np.nanmean(study['distance']):.3f}\n"
            f"IoU {np.nanmean(study['iou']):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#111111", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#999999",
                      alpha=0.88, lw=0.5))
    return im


# -------------------------------------------------------------
# The figure
# -------------------------------------------------------------
def _whole_bounds(*maps, symmetric: bool = False, q: float = 99.0):
    """Colour-bar limits rounded outwards to whole numbers, with integer ticks.

    The bars are read off the page rather than measured, so they start and end on a
    whole number instead of on whatever percentile the data happen to give.
    """
    v = float(np.percentile(np.abs(np.concatenate([m.ravel() for m in maps])), q))
    hi = max(1.0, float(np.ceil(v - 1e-9)))
    if symmetric:
        return -hi, hi, np.arange(-hi, hi + 0.5, 1.0)
    return 0.0, hi, np.arange(0.0, hi + 0.5, 1.0)


def render(study: dict, out: Path, dpi: int = 200) -> Path:
    """Four panels: the error before, what the detector found, and what each
    retraining changed. (c) and (d) share one scale so the two arms are comparable."""
    c, r = study["center"], study["radius"]

    e_lo, e_hi, e_ticks = _whole_bounds(study["init"])
    d_lo, d_hi, d_ticks = _whole_bounds(study["d_guided"], study["d_random"],
                                        symmetric=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0), constrained_layout=True)

    # ---- (a) the error landscape before selection --------------------------
    ax = axes[0, 0]
    im_e = _map(ax, study["init"], CMAP_SEQ, e_lo, e_hi)
    _true_circle(ax, c, r, label="induced weakspot")
    _frame(ax, "(a) Error before selection")
    _annotate(ax, study["init"], study, fmt="{:.2f}")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85, handlelength=1.4,
              borderpad=0.3)
    cb = fig.colorbar(im_e, ax=ax, location="right", fraction=0.046, pad=0.02,
                      ticks=e_ticks)
    cb.set_label("mean absolute error  $|\\hat{f}-f|$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---- (b) what the detector found ---------------------------------------
    im_d = panel_detected(axes[0, 1], study, "(b) Detected weakspot")
    cbd = fig.colorbar(im_d, ax=axes[0, 1], location="right", fraction=0.046,
                       pad=0.02, ticks=[0, 1])
    cbd.set_label("detection surface (normalised)", fontsize=9)
    cbd.ax.tick_params(labelsize=8)

    # ---- (c), (d) what each retraining changed, on one shared scale ---------
    # Each gets its own bar, laid out like (a) and (b), but both use the same limits.
    for ax, Z, t in zip(axes[1, :], (study["d_guided"], study["d_random"]),
                        ("(c) Guided $-$ before", "(d) Random $-$ before")):
        im = _map(ax, Z, CMAP_DIFF, d_lo, d_hi)
        _true_circle(ax, c, r)
        _frame(ax, t)
        _annotate(ax, Z, study)
        cbg = fig.colorbar(im, ax=ax, location="right", fraction=0.046, pad=0.02,
                           ticks=d_ticks)
        cbg.set_label("change in absolute error  $\\Delta|\\hat{f}-f|$", fontsize=9)
        cbg.ax.tick_params(labelsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("$x_2$", fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("$x_1$", fontsize=9)

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# -------------------------------------------------------------
# Reported numbers
# -------------------------------------------------------------
def report(study: dict) -> dict:
    """Every number the paper's text quotes, in one dict (also written as JSON)."""
    s = EL.summarise(study)
    lx, ly, c, r = (study["land_xx"], study["land_yy"], study["center"],
                    study["radius"])
    for key, name in (("init", "before"), ("guided", "guided"), ("random", "random")):
        i, o = EL.region_means(study[key], lx, ly, c, r)
        s[f"land_{name}_in"], s[f"land_{name}_out"] = i, o
    for key, name in (("d_guided", "guided"), ("d_random", "random")):
        i, o = EL.region_means(study[key], lx, ly, c, r)
        s[f"dland_{name}_in"], s[f"dland_{name}_out"] = i, o
    s["gap_frac_of_area"] = float(np.pi * r ** 2)
    return s


def main(argv=None) -> int:
    args = parse_args(argv)
    detectors = args.detector or [PAPER_DETECTOR]
    seeds = SEEDS[:max(1, min(args.seeds, len(SEEDS)))]
    n_runs = len(seeds) * len(detectors)

    # Re-running 250 rounds to nudge a colour bar is wasteful, so the computed study
    # is cached beside the figure and reused unless the recipe changed.
    cache = args.out.with_name(args.out.stem + "_study.pkl")
    recipe = dict(seeds=seeds, detectors=detectors, land_res=args.land_res,
                  op=EL.OPERATING_POINT)
    study = None
    if cache.exists() and not args.recompute:
        with open(cache, "rb") as f:
            cached = pickle.load(f)
        if cached.get("_recipe") == recipe:
            study = cached
            print(f"reusing cached study ({n_runs} rounds) from {cache}")

    if study is None:
        print(f"Running the error-landscape study: {len(seeds)} seeds x "
              f"{len(detectors)} detectors = {n_runs} rounds "
              f"at the operating point (radius {EL.OPERATING_POINT['radius']}, "
              f"mix {EL.OPERATING_POINT['mix_ratio']}, "
              f"sigma {EL.OPERATING_POINT['sel_sigma']}) ...")

        def _progress(done, total):
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  {done}/{total}", end="\r", flush=True)

        study = EL.run_study(seeds, detectors=detectors, land_res=args.land_res,
                             progress=_progress)
        print()
        study["_recipe"] = recipe
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(study, f)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = render(study, args.out, dpi=args.dpi)
    print(f"wrote {out}")

    rep = report(study)
    (args.out.with_suffix(".json")).write_text(json.dumps(rep, indent=2),
                                               encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.json')}")

    if args.copy_to_paper:
        PAPER_FIGDIR.mkdir(parents=True, exist_ok=True)
        dest = PAPER_FIGDIR / out.name
        dest.write_bytes(out.read_bytes())
        print(f"copied to {dest}")

    print("\n--- numbers for the paper ---")
    for k, v in rep.items():
        print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
