"""Plots for the focused single-axis / 2D sweeps around the big-experiment sweet spot.

Storyline these figures serve:
  1. Weakspot-guided training is NOT universally better -> we swept a big space.
  2. A few parameters dominate the outcome (which, and how much).
  3. Anchored at a big-sweep sweet spot, we isolate each parameter and compare the
     GUIDED (weakspot) model against the RANDOM baseline over the whole area AND
     inside the weakspot, with the performance difference made explicit.

Every focused row already carries, for one initial model that BOTH the guided and
the random retrain continue from (fair warm-start):
    whole-area MAE : init_mae, guided_mae, base_mae
    in-weakspot err: init_err_in, guided_err_in, base_err_in
    convenience diff: gap_mae_guided_minus_base (= guided_mae - base_mae)
We aggregate the 3 seeds x 3 detectors per swept value (mean + 95% CI).

Run:  python scripts/dataselect/plot_focused.py
Figures are written to  data/experiment_results/data_selective_training/figures/
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve()
APP = HERE.parents[2]
RESULTS = APP / "data" / "experiment_results" / "data_selective_training"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# consistent visual language
C_GUIDED = "#1f77b4"   # weakspot-guided (our method)
C_RANDOM = "#d62728"   # random baseline
C_INIT   = "#7f7f7f"   # initial model (shared reference, pre-retrain)
C_WIN    = "#2ca02c"   # guided better
C_LOSE   = "#d62728"   # guided worse

# swept axis + human label for each single-axis config
SINGLE = {
    "best_vary_iters_initial": ("iters_initial", "Initial training time (MLP iterations)"),
    "best_vary_mix":           ("mix_ratio",     "Rehearsal mix ratio (0 = uniform, 1 = pure guided)"),
    "best_vary_radius":        ("radius",        "Induced weakspot gap size (radius)"),
    "best_vary_sel_sigma":     ("sel_sigma",     "Selection-kernel width (sel_sigma) — wide → narrow"),
    "best_vary_n_select":      ("n_select",      "Number of curated points added (n_select)"),
    "iso_gaussian":            ("sel_sigma",     "Kernel width (sel_sigma), mix=1 — wide → narrow = coverage → focus"),
    "iso_mix":                 ("mix_ratio",     "Mix ratio, sel_sigma=0.1 (narrow) — 0 → 1 = coverage → focus"),
}
# configs whose x-axis is a kernel width: plot HIGH→LOW so "focus" points right,
# developing symmetrically to the mix axis (0 → 1 = coverage → focus).
REVERSE_X = {"best_vary_sel_sigma", "iso_gaussian"}


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"sweep__{name}.csv")
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    return df


def _agg(df: pd.DataFrame, axis: str, col: str):
    """mean + 95% CI half-width of `col` grouped by the swept axis."""
    g = df.groupby(axis)[col]
    m = g.mean()
    sem = g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))
    ci = 1.96 * sem.fillna(0.0)
    return m.index.values, m.values, ci.values


def _panel(ax, axis, df, init_c, guided_c, base_c, ylabel):
    """One comparison panel: init / guided / random lines with CI bands."""
    x, gi, gci = _agg(df, axis, guided_c)
    _, bi, bci = _agg(df, axis, base_c)
    _, ii, _   = _agg(df, axis, init_c)
    ax.plot(x, ii, "--", color=C_INIT, lw=1.4, label="Initial model (pre-retrain)", zorder=2)
    ax.plot(x, bi, "-o", color=C_RANDOM, ms=4, lw=1.8, label="Random baseline", zorder=3)
    ax.fill_between(x, bi - bci, bi + bci, color=C_RANDOM, alpha=0.15, zorder=1)
    ax.plot(x, gi, "-o", color=C_GUIDED, ms=4, lw=1.8, label="Weakspot-guided (ours)", zorder=4)
    ax.fill_between(x, gi - gci, gi + gci, color=C_GUIDED, alpha=0.15, zorder=1)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return x, gi, bi


def _diff_panel(ax, x, guided, base, xlabel):
    """Lower panel: performance difference = random - guided (positive => guided better)."""
    d = base - guided
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(x, d, "-o", color="k", ms=3, lw=1.2)
    ax.fill_between(x, 0, d, where=d >= 0, color=C_WIN, alpha=0.30, interpolate=True,
                    label="guided better")
    ax.fill_between(x, 0, d, where=d < 0, color=C_LOSE, alpha=0.30, interpolate=True,
                    label="guided worse")
    ax.set_ylabel("Δ = random − guided")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)


def plot_single(name: str):
    axis, xlabel = SINGLE[name]
    df = _load(name)
    fig = plt.figure(figsize=(12, 6.2))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.22)

    # --- whole area (left) ---
    axA = fig.add_subplot(gs[0, 0])
    x, g, b = _panel(axA, axis, df, "init_mae", "guided_mae", "base_mae",
                     "Whole-area MAE")
    axA.legend(fontsize=8, loc="best")
    axA.tick_params(labelbottom=False)
    axAd = fig.add_subplot(gs[1, 0], sharex=axA)
    _diff_panel(axAd, x, g, b, xlabel)

    # --- in-weakspot (right) ---
    axB = fig.add_subplot(gs[0, 1])
    dfw = df.dropna(subset=["guided_err_in", "base_err_in"])
    xw, gw, bw = _panel(axB, axis, dfw, "init_err_in", "guided_err_in", "base_err_in",
                        "In-weakspot error")
    axB.legend(fontsize=8, loc="best")
    axB.tick_params(labelbottom=False)
    axBd = fig.add_subplot(gs[1, 1], sharex=axB)
    _diff_panel(axBd, xw, gw, bw, xlabel)

    if name in REVERSE_X:            # kernel-width axes read wide → narrow
        axA.invert_xaxis(); axB.invert_xaxis()

    out = FIGDIR / f"fig_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_effect_sizes():
    """Which parameter most swings the whole-area advantage (biggest effect)?
    For each swept axis: range of the mean (random - guided) across its values."""
    spans = []
    for name, (axis, label) in SINGLE.items():
        if name.startswith("iso_"):     # isolation configs use special anchors; exclude
            continue
        df = _load(name)
        m = df.groupby(axis).apply(
            lambda s: (s["base_mae"] - s["guided_mae"]).mean(), include_groups=False)
        spans.append((label.split(" (")[0], m.min(), m.max(), m.max() - m.min()))
    spans.sort(key=lambda t: t[3])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ys = np.arange(len(spans))
    for y, (lab, lo, hi, span) in zip(ys, spans):
        ax.plot([lo, hi], [y, y], color=C_GUIDED, lw=6, solid_capstyle="round", alpha=0.8)
        ax.plot(lo, y, "o", color=C_LOSE, ms=7)
        ax.plot(hi, y, "o", color=C_WIN, ms=7)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in spans])
    ax.set_xlabel("Whole-area advantage  Δ = random − guided MAE  (positive = guided better)")
    ax.grid(axis="x", alpha=0.25)
    out = FIGDIR / "fig_effect_sizes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_global_sensitivity():
    """Sensitivity of EVERY swept axis, computed from the broad sweep (alpha_boundary).
    For each parameter, the mean whole-area advantage (base-guided MAE) at each of its
    levels; the bar spans [min..max] across levels (its main effect), sorted by span.
    Answers 'which of ALL the swept parameters move the outcome, and by how much'."""
    LABELS = {
        "mix_ratio": "Rehearsal mix ratio", "sel_sigma": "Selection-kernel width",
        "n_select": "Curated points added", "iters_initial": "Initial training time",
        "n_train": "Initial training-set size", "n_bumps": "Function complexity",
        "sel_method": "Selection strategy", "radius": "Induced gap size",
        "noise_std": "Label-noise level", "iters_retrain": "Retraining time",
    }
    use = ["guided_mae", "base_mae"] + list(LABELS)
    df = pd.read_csv(RESULTS / "sweep__alpha_boundary.csv",
                     usecols=lambda c: c in use).dropna(subset=["guided_mae", "base_mae"])
    df["adv"] = df["base_mae"] - df["guided_mae"]
    spans = []
    for k, lab in LABELS.items():
        m = df.groupby(k)["adv"].mean()
        spans.append((lab, float(m.min()), float(m.max()), float(m.max() - m.min())))
    spans.sort(key=lambda t: t[3])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ys = np.arange(len(spans))
    for y, (lab, lo, hi, span) in zip(ys, spans):
        ax.plot([lo, hi], [y, y], color=C_GUIDED, lw=6, solid_capstyle="round", alpha=0.8)
        ax.plot(lo, y, "o", color=C_LOSE, ms=7)
        ax.plot(hi, y, "o", color=C_WIN, ms=7)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in spans])
    ax.set_xlabel("Whole-area advantage  Δ = random − guided MAE  (positive = guided better)")
    ax.grid(axis="x", alpha=0.25)
    out = FIGDIR / "fig_global_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sensitivity_heatmap():
    """Sensitivity heatmap over the broad sweep: one row per swept parameter, one
    column per level (in increasing order), cell colour = mean whole-area advantage
    (base-guided MAE) at that level, cell text = the level value. Rows sorted by span
    (most influential on top). A single global colour scale, so a flat/amber row is a
    near-inert parameter and a strong gradient (mix, kernel width) is an influential
    one."""
    LABELS = {
        "mix_ratio": "Rehearsal mix ratio", "sel_sigma": "Selection-kernel width",
        "n_select": "Curated points added", "iters_initial": "Initial training time",
        "n_train": "Initial training-set size", "n_bumps": "Function complexity",
        "sel_method": "Selection strategy", "radius": "Induced gap size",
        "noise_std": "Label-noise level", "iters_retrain": "Retraining time",
    }
    SHORT = {"Weakpoint distance": "dist", "Weight by landscape": "land",
             "Shape aware": "shape"}
    def fmt(v):
        return SHORT.get(v, v) if isinstance(v, str) else f"{v:g}"

    use = ["guided_mae", "base_mae"] + list(LABELS)
    df = pd.read_csv(RESULTS / "sweep__alpha_boundary.csv",
                     usecols=lambda c: c in use).dropna(subset=["guided_mae", "base_mae"])
    df["adv"] = df["base_mae"] - df["guided_mae"]
    rows = []
    for k, lab in LABELS.items():
        m = df.groupby(k)["adv"].mean()
        rows.append((lab, list(m.index), list(m.values), float(m.max() - m.min())))
    rows.sort(key=lambda r: -r[3])                     # most influential on top
    ncol = max(len(r[1]) for r in rows)
    M = np.full((len(rows), ncol), np.nan)
    for i, (lab, levs, vals, span) in enumerate(rows):
        M[i, :len(vals)] = vals
    vmax = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    for i, (lab, levs, vals, span) in enumerate(rows):
        for j, (lev, v) in enumerate(zip(levs, vals)):
            ax.text(j, i - 0.22, fmt(lev), ha="center", va="center", fontsize=7,
                    color="#333333")
            ax.text(j, i + 0.20, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="black", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"level {j+1}" for j in range(ncol)])
    ax.set_xlabel("parameter level (increasing →); cell value above, "
                  "mean advantage below")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Δ = random − guided MAE  (green = guided better)")
    out = FIGDIR / "fig_sensitivity_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_detector():
    df = _load("best_vary_detector")
    def bars(ax, col_g, col_b, xlabel):
        grp = df.groupby("method").apply(
            lambda s: pd.Series({"adv": (s[col_b] - s[col_g]).mean()}), include_groups=False)
        grp = grp.sort_values("adv")
        colors = [C_WIN if v >= 0 else C_LOSE for v in grp["adv"]]
        ax.barh(np.arange(len(grp)), grp["adv"], color=colors, alpha=0.85)
        ax.set_yticks(np.arange(len(grp)))
        ax.set_yticklabels(grp.index, fontsize=8)
        ax.axvline(0, color="k", lw=0.9)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.5), sharey=False)
    bars(a1, "guided_mae", "base_mae", "Whole-area advantage  (random − guided MAE)")
    bars(a2, "guided_err_in", "base_err_in", "In-weakspot advantage  (random − guided err)")
    fig.tight_layout()
    out = FIGDIR / "fig_best_vary_detector.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmaps():
    def heat(ax, name, rows, cols, rlab, clab):
        df = _load(name)
        piv = df.pivot_table(index=rows, columns=cols,
                             values="gap_mae_guided_minus_base", aggfunc="mean")
        # show ADVANTAGE = -(guided - base) = base - guided so green/positive = guided better
        adv = -piv
        vmax = np.nanmax(np.abs(adv.values))
        im = ax.imshow(adv.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(adv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in adv.columns], rotation=0, fontsize=8)
        ax.set_yticks(range(len(adv.index)))
        ax.set_yticklabels([f"{r:g}" for r in adv.index], fontsize=8)
        ax.set_xlabel(clab)
        ax.set_ylabel(rlab)
        for i in range(adv.shape[0]):
            for j in range(adv.shape[1]):
                v = adv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Δ = random − guided MAE")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    heat(a1, "best_2d_sigma_mix", "sel_sigma", "mix_ratio",
         "sel_sigma (kernel width)", "mix_ratio (rehearsal)")
    heat(a2, "best_2d_mix_iters", "mix_ratio", "iters_initial",
         "mix_ratio (rehearsal)", "iters_initial (init training)")
    fig.tight_layout()
    out = FIGDIR / "fig_2d_interactions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_isolation():
    """Paired figure: the two independent routes to coverage that both prevent
    catastrophic forgetting — uniform rehearsal (mix, narrow kernel) vs kernel
    width (sel_sigma, pure guided). The Gaussian axis is drawn wide→narrow so both
    panels develop identically: left = coverage (≈ baseline), right = pure focus
    (forgetting). Shared y-axis for a direct comparison."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)

    def panel(ax, name, axis, reverse, xlabel, logx=False):
        df = _load(name)
        x, gi, gci = _agg(df, axis, "guided_mae")
        _, bi, _   = _agg(df, axis, "base_mae")
        _, ii, _   = _agg(df, axis, "init_mae")
        ax.plot(x, ii, "--", color=C_INIT, lw=1.3, label="Initial model")
        ax.plot(x, bi, "-o", color=C_RANDOM, ms=4, lw=1.8, label="Random baseline")
        ax.plot(x, gi, "-o", color=C_GUIDED, ms=4, lw=1.8, label="Weakspot-guided (ours)")
        ax.fill_between(x, gi + gci, gi - gci, color=C_GUIDED, alpha=0.15)
        # shade guided-vs-baseline difference
        ax.fill_between(x, gi, bi, where=gi <= bi, color=C_WIN, alpha=0.22, interpolate=True)
        ax.fill_between(x, gi, bi, where=gi > bi, color=C_LOSE, alpha=0.22, interpolate=True)
        if logx:
            ax.set_xscale("log")
        if reverse:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, which="both")
        # coverage / focus annotations along the bottom
        ax.text(0.02, 0.02, "coverage\n(≈ baseline)", transform=ax.transAxes,
                fontsize=8, color="#2ca02c", va="bottom", ha="left")
        ax.text(0.98, 0.02, "pure focus\n→ forgetting", transform=ax.transAxes,
                fontsize=8, color="#d62728", va="bottom", ha="right")
        return ax

    panel(axL, "iso_mix", "mix_ratio", False,
          "mix_ratio   (0 = pure uniform  →  1 = pure guided)")
    axL.set_ylabel("Whole-area MAE")
    axL.legend(fontsize=8, loc="upper center")
    panel(axR, "iso_gaussian", "sel_sigma", True,
          "sel_sigma  (log scale;  wide  →  narrow)", logx=True)

    fig.tight_layout()
    out = FIGDIR / "fig_isolation_forgetting.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_forgetting_arc():
    """The full 'random-better → guided-better → converge' arc — realised on the
    KERNEL-WIDTH axis (mix=1, pure guided), NOT on mix (where mix=0 is by
    construction the tie/baseline). Narrow sigma = over-concentration = forgetting
    (random wins); mid sigma = guided wins; sigma→∞ = kernel flattens to uniform =
    guided converges to the random baseline. Log-x because sigma spans 0.05..30."""
    df = _load("iso_gaussian")
    x, gi, gci = _agg(df, "sel_sigma", "guided_mae")
    _, bi, _   = _agg(df, "sel_sigma", "base_mae")
    _, ii, _   = _agg(df, "sel_sigma", "init_mae")
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(x, ii, "--", color=C_INIT, lw=1.3, label="Initial model")
    ax.plot(x, bi, "-", color=C_RANDOM, lw=2.0, label="Random baseline")
    ax.plot(x, gi, "-o", color=C_GUIDED, ms=5, lw=1.9, label="Weakspot-guided (mix=1)")
    ax.fill_between(x, gi + gci, gi - gci, color=C_GUIDED, alpha=0.15)
    ax.fill_between(x, gi, bi, where=gi <= bi, color=C_WIN, alpha=0.22, interpolate=True)
    ax.fill_between(x, gi, bi, where=gi > bi, color=C_LOSE, alpha=0.22, interpolate=True)
    ax.set_xscale("log")
    # crossover where guided overtakes random
    cross = next((x[i] for i in range(1, len(x)) if gi[i] <= bi[i] < gi[i-1]), None)
    if cross:
        ax.axvline(cross, color="k", lw=0.8, ls=":")
    ax.set_xlabel("sel_sigma  (log scale;  narrow / over-concentrated  →  wide / uniform)")
    ax.set_ylabel("Whole-area MAE")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.95, "narrow σ:\nforgetting\n(random wins)", transform=ax.transAxes,
            fontsize=9, color="#d62728", va="top", ha="left")
    ax.text(0.44, 0.30, "guided wins", transform=ax.transAxes,
            fontsize=10, color="#2ca02c", va="center", ha="center")
    ax.text(0.97, 0.30, "wide σ:\n→ baseline\n(converge)", transform=ax.transAxes,
            fontsize=9, color="#555555", va="center", ha="right")
    out = FIGDIR / "fig_forgetting_arc_sigma.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_forgetting_arc_mix():
    """The SAME arc on the mix axis (the 'mirror'): narrow kernel (sel_sigma=0.1),
    sweep mix. Plotted mix 1→0 so it reads forgetting → guided wins → converge, and
    unlike the sigma tail the mix=0 end converges EXACTLY (pure uniform ≡ baseline)."""
    df = _load("iso_mix")
    x, gi, gci = _agg(df, "mix_ratio", "guided_mae")
    _, bi, _   = _agg(df, "mix_ratio", "base_mae")
    _, ii, _   = _agg(df, "mix_ratio", "init_mae")
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(x, ii, "--", color=C_INIT, lw=1.3, label="Initial model")
    ax.plot(x, bi, "-", color=C_RANDOM, lw=2.0, label="Random baseline")
    ax.plot(x, gi, "-o", color=C_GUIDED, ms=5, lw=1.9, label="Weakspot-guided (sel_sigma=0.1)")
    ax.fill_between(x, gi + gci, gi - gci, color=C_GUIDED, alpha=0.15)
    ax.fill_between(x, gi, bi, where=gi <= bi, color=C_WIN, alpha=0.22, interpolate=True)
    ax.fill_between(x, gi, bi, where=gi > bi, color=C_LOSE, alpha=0.22, interpolate=True)
    ax.invert_xaxis()   # mix 1 → 0 : forgetting on the left, convergence on the right
    ax.set_xlabel("mix_ratio   (1 = pure guided / focus  →  0 = pure uniform = baseline)")
    ax.set_ylabel("Whole-area MAE")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.95, "mix→1:\nforgetting\n(random wins)", transform=ax.transAxes,
            fontsize=9, color="#d62728", va="top", ha="left")
    ax.text(0.55, 0.25, "guided wins", transform=ax.transAxes,
            fontsize=10, color="#2ca02c", va="center", ha="center")
    ax.text(0.97, 0.30, "mix=0:\npure uniform\n= baseline", transform=ax.transAxes,
            fontsize=9, color="#555555", va="center", ha="right")
    out = FIGDIR / "fig_forgetting_arc_mix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    made = []
    for name in SINGLE:
        made.append(plot_single(name))
    made.append(plot_isolation())
    made.append(plot_forgetting_arc())
    made.append(plot_forgetting_arc_mix())
    made.append(plot_effect_sizes())
    made.append(plot_detector())
    made.append(plot_heatmaps())
    print("wrote:")
    for p in made:
        print("  ", p.relative_to(APP))


if __name__ == "__main__":
    main()
