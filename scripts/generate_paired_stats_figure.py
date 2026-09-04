"""
Generate the paired per-configuration comparison figure referenced by ``main.tex``:

    paired_improvement.png   Delta mean IoU@q=0.90 of every method against the
                             strongest base method, with clustered bootstrap 95%
                             CIs, win/loss counts and Holm-corrected Wilcoxon p.

Source : data/experiment_results/weakspot_experiment/sweep_results.csv (the live sweep CSV).
Output : straight into the paper's figures/ folder, so no manual copy is needed.

Why this figure exists (reviewer feedback, ICTAI submission 719):
  Reviewer 3 objected that the headline 0.135-vs-0.128 mean-IoU gap was reported
  without "paired effects by configuration, clustered uncertainty, multiplicity
  correction, and win/loss rates". Every method is evaluated on the *same* 1215
  configurations, so the comparison is naturally paired: differencing within a
  configuration removes the (large) configuration-difficulty variance that the
  raw per-method standard deviation of ~0.22 is actually measuring.

Statistics reported, all computed within configuration:
  * mean paired difference vs the reference method;
  * clustered bootstrap 95% CI -- configurations are resampled in the 81 cells
    of (n_bumps x noise_std x centre), because the 15 runs sharing a cell share a
    ground-truth landscape and gap location and are not independent. This is
    wider, and more honest, than resampling the 1215 rows individually;
  * Wilcoxon signed-rank p, Holm-corrected over the 16 comparisons. The rank test
    rather than a paired t-test because ~68% of per-configuration IoU values are
    exactly zero, so the differences are nowhere near normal;
  * win / loss / tie counts, i.e. on how many configurations the method actually
    beats the reference.

Re-run whenever the sweep CSV changes:
    python -m scripts.generate_paired_stats_figure
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Anchored to this file, not to the working directory: ``Documents`` is a sibling
# of ``Application``, so a path relative to the cwd silently creates a stray tree
# when the script is launched from inside ``Application``.
_APP_ROOT = Path(__file__).resolve().parents[1]           # .../Application
_REPO_ROOT = _APP_ROOT.parent                             # .../MS_Research_Demonstrator

CSV_PATH = _APP_ROOT / "data/experiment_results/weakspot_experiment/sweep_results.csv"
OUT_DIR = (_REPO_ROOT / "Documents/Paper_Advanced EnsembleMethods"
           / "Paper_Advanced-Ensembles-for-Weakspot-Identification/figures")
FILENAME = "paired_improvement.png"

# Reference: the strongest base method by mean IoU, and also EVT x GPR's own
# localizer component -- so the hardest possible comparison for the ensemble.
REFERENCE = "Peaks over Threshold (EVT)"
PRIMARY = "iou_q90"

# Method tiers -- must match scripts/weakspot/detection.py registry keys and the
# colours used in generate_method_comparison_figures.py.
BASE_METHODS = [
    "Gaussian Process Regression",
    "Polynomial Response Surface",
    "RBF Interpolation",
    "kNN Performance Mapping",
    "LOESS Local Regression",
    "Quantile Regression",
    "Peaks over Threshold (EVT)",
    "Bayesian Optimization (EI)",
]
SIMPLE_ENSEMBLES = [
    "kNN + GPR (mean)",
    "kNN + Quantile (mean)",
]
ADVANCED_ENSEMBLES = [
    "Anchored GPR (QR prior)",
    "Gated GPR (QR anchor)",
    "QR × GPR (geometric)",
    "EVT × GPR (geometric)",
    "GPR/kNN (variance-weighted)",
    "GPR/kNN (disagreement-amplified)",
    "Local GPR (QR-localised)",
]
TIER = {m: "Base" for m in BASE_METHODS}
TIER.update({m: "Simple" for m in SIMPLE_ENSEMBLES})
TIER.update({m: "Advanced" for m in ADVANCED_ENSEMBLES})
TIER_COLOR = {"Base": "#ff7f0e", "Simple": "#888888", "Advanced": "#1f77b4"}

# Cluster = one (landscape x noise x gap-centre) cell; the runs inside it share a
# ground-truth function and a gap location, so they are not independent draws.
CLUSTER_KEYS = ["n_bumps", "noise_std", "center_x", "center_y"]
CONFIG_KEYS = ["param_key"] + CLUSTER_KEYS

N_BOOT = 20000
SEED = 0

# Fonts matched to generate_method_comparison_figures.py.
FS_LABEL, FS_TICK, FS_VALUE, FS_LEGEND = 16, 15, 12, 14


def _load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}. Run the sweep first.")
    df = pd.read_csv(CSV_PATH)
    for col in [PRIMARY, "method"] + CLUSTER_KEYS + ["param_key"]:
        if col not in df.columns:
            sys.exit(f"CSV missing required column: {col}")
    return df


def _paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration, one column per method -- the paired design."""
    wide = df.pivot_table(index=CONFIG_KEYS, columns="method",
                          values=PRIMARY).reset_index()
    if REFERENCE not in wide.columns:
        sys.exit(f"Reference method {REFERENCE!r} not in the CSV.")
    missing = wide[[m for m in TIER if m in wide.columns]].isna().sum().sum()
    if missing:
        print(f"  warning: {missing} missing method/configuration cells; "
              f"those configurations are dropped from the affected comparisons.")
    return wide


def _clustered_ci(delta: pd.Series, cluster: pd.Series,
                  rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap the mean paired difference by resampling whole clusters.

    Clusters here are equal-sized, so the cluster mean-of-means is the overall
    mean; resampling clusters propagates the within-cluster correlation that a
    naive row-level bootstrap ignores (and so understates).
    """
    means = delta.groupby(cluster, sort=False).mean().values
    draws = rng.choice(means, (N_BOOT, len(means)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _holm(pvals: list[float]) -> list[float]:
    """Holm step-down correction; returns adjusted p in the input order."""
    order = np.argsort(pvals)
    n, adjusted, running = len(pvals), [0.0] * len(pvals), 0.0
    for k, idx in enumerate(order):
        running = max(running, min((n - k) * pvals[idx], 1.0))
        adjusted[idx] = running
    return adjusted


def compute(df: pd.DataFrame) -> pd.DataFrame:
    wide = _paired_table(df)
    cluster = wide[CLUSTER_KEYS].astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(SEED)

    rows = []
    for method in wide.columns:
        if method not in TIER or method == REFERENCE:
            continue
        pair = wide[[method, REFERENCE]].dropna()
        delta = pair[method] - pair[REFERENCE]
        lo, hi = _clustered_ci(delta, cluster.loc[pair.index], rng)
        naive = rng.choice(delta.values, (N_BOOT, len(delta)),
                           replace=True).mean(axis=1)
        rows.append(dict(
            method=method, tier=TIER[method],
            mean_iou=pair[method].mean(), delta=delta.mean(),
            lo=lo, hi=hi,
            naive_lo=float(np.percentile(naive, 2.5)),
            naive_hi=float(np.percentile(naive, 97.5)),
            win=int((delta > 0).sum()), loss=int((delta < 0).sum()),
            tie=int((delta == 0).sum()),
            p=stats.wilcoxon(pair[method], pair[REFERENCE],
                             zero_method="wilcox").pvalue,
        ))

    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p"].tolist())
    return out.sort_values("delta", ascending=False).reset_index(drop=True)


def _fmt_p(p: float) -> str:
    if p >= 0.10:
        return f"p={p:.2f}"
    if p >= 0.001:
        return f"p={p:.3f}"
    exponent = int(np.floor(np.log10(p)))
    return f"p<1e{exponent + 1}"


def plot(stats_df: pd.DataFrame, ref_mean: float, n_config: int,
         n_cluster: int) -> Path:
    """Forest plot, sized for a full-width (two-column) IEEE float.

    16 labelled rows will not survive being scaled to a single 3.5 in column, so
    the aspect ratio is chosen for ``figure*``: at 7 in wide the tick labels land
    at roughly 9-10 pt. Win/loss and p live in a right-hand margin outside the
    axes, so the plotting area carries only data.
    """
    ordered = stats_df.iloc[::-1].reset_index(drop=True)  # best at the top
    ypos = np.arange(len(ordered))
    colors = [TIER_COLOR[t] for t in ordered["tier"]]

    # constrained_layout is off on purpose: the margin annotations are drawn with
    # clip_on=False outside the axes and would not be accounted for by it.
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    fig.subplots_adjust(left=0.255, right=0.735, top=0.945, bottom=0.155)

    ax.axvline(0.0, color="black", linewidth=1.4, linestyle="--", zorder=1)
    for y, row, color in zip(ypos, ordered.itertuples(), colors):
        ax.plot([row.lo, row.hi], [y, y], color=color, linewidth=2.6,
                solid_capstyle="round", zorder=2)
        for cap in (row.lo, row.hi):
            ax.plot([cap, cap], [y - 0.2, y + 0.2], color=color, linewidth=2.0,
                    zorder=2)
    ax.scatter(ordered["delta"], ypos, s=95, color=colors, edgecolor="black",
               linewidth=0.9, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(ordered["method"], fontsize=FS_TICK)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlabel(r"$\Delta$ mean IoU vs. Peaks over Threshold (EVT)"
                  "\n" r"(right of 0 $=$ better than the strongest base method)",
                  fontsize=FS_LABEL)
    # No in-image title: the LaTeX caption carries it, and any title wide enough
    # to be useful overruns the axes and collides with the margin headers.
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.grid(axis="x", color="lightgray", linewidth=0.7)
    ax.set_axisbelow(True)

    span = float(ordered["hi"].max() - ordered["lo"].min())
    ax.set_xlim(float(ordered["lo"].min()) - 0.05 * span,
                float(ordered["hi"].max()) + 0.05 * span)

    # Right margin: win/loss counts and the Holm-corrected significance.
    for col, header, x in (("wl", "win/loss", 1.035), ("p", "Holm $p$", 1.30)):
        ax.text(x, 1.012, header, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS_VALUE, style="italic", color="dimgray",
                clip_on=False)
    for y, row in zip(ypos, ordered.itertuples()):
        yfrac = (y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(1.035, yfrac, f"{row.win}/{row.loss}", transform=ax.transAxes,
                ha="left", va="center", fontsize=FS_VALUE, family="monospace",
                clip_on=False)
        ax.text(1.30, yfrac, _fmt_p(row.p_holm), transform=ax.transAxes,
                ha="left", va="center", fontsize=FS_VALUE, family="monospace",
                clip_on=False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=TIER_COLOR[t], edgecolor="black", label=t)
               for t in ("Base", "Simple", "Advanced")]
    ax.legend(handles=handles, fontsize=FS_LEGEND, ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.135))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / FILENAME
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    df = _load()
    wide = _paired_table(df)
    n_config = len(wide)
    n_cluster = wide[CLUSTER_KEYS].astype(str).agg("|".join, axis=1).nunique()
    ref_mean = wide[REFERENCE].mean()

    s = compute(df)

    print(f"\nReference: {REFERENCE}  (mean IoU {ref_mean:.4f}, "
          f"{n_config} configurations in {n_cluster} clusters)\n")
    header = (f"{'method':34s} {'IoU':>6s} {'dIoU':>8s} {'clustered 95% CI':>20s} "
              f"{'naive 95% CI':>20s} {'W/L/T':>15s} {'p_Holm':>9s}")
    print(header)
    print("-" * len(header))
    for r in s.itertuples():
        print(f"{r.method:34s} {r.mean_iou:6.3f} {r.delta:+8.4f} "
              f"[{r.lo:+8.4f},{r.hi:+8.4f}] [{r.naive_lo:+8.4f},{r.naive_hi:+8.4f}] "
              f"{r.win:4d}/{r.loss:4d}/{r.tie:4d} {r.p_holm:9.2e}")

    path = plot(s, ref_mean, n_config, n_cluster)
    print(f"\n  wrote {path}")

    csv_path = OUT_DIR / "paired_stats.csv"
    s.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
