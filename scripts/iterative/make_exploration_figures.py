"""Exploration figures: static vs dynamic vs self-gating data selection.

Reruns the variant comparison keeping the full per-iteration trajectories (the
earlier pass kept only summary statistics), writes the raw data next to the
figures so nothing has to be recomputed, and draws:

  expl_k10_variants_gap.png      guided-minus-random per iteration, all seven
                                 variants at a 10-round horizon
  expl_k10_variants_summary.png  their across-loop means with 95% CIs, ranked
  expl_a035_k20_gap.png          the narrative plot: at alpha=0.35 over 20 rounds,
                                 static wins round 1 then degrades significantly,
                                 while relaxing the focus keeps the early win and
                                 removes the degradation
  expl_a035_k20_controls.png     what each schedule actually applied (alpha, sigma)
  expl_signals.png               why the size gate fails and the severity gate does
                                 not — the two candidate controller inputs over the
                                 loop, from the saved 20-round runs

Everything lands in <results>/figures/Exploration/. Titles are omitted throughout,
matching the main figure set, so any of these can be promoted into the paper.
"""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
APP = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application")
sys.path.insert(0, str(APP))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.iterative.loop import run_iterative, default_config

RES = APP / "data/experiment_results/iterative_data_selective_training"
OUT = RES / "figures/Exploration"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 7, 1, 3, 5, 11, 17, 23, 99, 2, 4, 6, 8, 13]
SEV = "Adaptive (weakspot severity)"
SIZ = "Adaptive (weakspot size)"

VARIANTS_K10 = {
    "static  σ=.5, α=.5":            {},
    "dynamic  σ widen, α decay":     dict(sigma_schedule="Linear widen",
                                          mix_schedule="Linear decay"),
    "self-gate severity":            dict(sigma_schedule=SEV, mix_schedule=SEV,
                                          sigma_rate=1.0),
    "self-gate size":                dict(sigma_schedule=SIZ, mix_schedule=SIZ,
                                          sigma_rate=1.0),
    "mix only  σ=.1 fixed":          dict(sel_sigma=0.1, mix_schedule="Linear decay"),
    "mix only gated  σ=.1 fixed":    dict(sel_sigma=0.1, mix_schedule=SEV),
    "sigma only  α=1 fixed":         dict(mix_ratio=1.0, sigma_schedule="Linear widen"),
}
VARIANTS_A035 = {
    "static  α=.35":   dict(mix_ratio=0.35),
    "dynamic  α=.35":  dict(mix_ratio=0.35, sigma_schedule="Linear widen",
                            mix_schedule="Linear decay"),
    "self-gate  α=.35": dict(mix_ratio=0.35, sigma_schedule=SEV,
                             mix_schedule=SEV, sigma_rate=1.0),
}

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def collect(variants: dict, K: int, tag: str) -> pd.DataFrame:
    """Run every (variant x seed) and keep each round's gap and applied controls."""
    cached = OUT / f"raw_{tag}.csv"
    if cached.exists():
        return pd.read_csv(cached)
    rows = []
    for name, kw in variants.items():
        for s in SEEDS:
            r = run_iterative(default_config(seed=s, n_iterations=K,
                                             single_shot=False, **kw),
                              keep_rounds=False)
            gk, rk = r["driver"]
            g = np.asarray(r["tracks"][gk]["mae"], float)
            b = np.asarray(r["tracks"][rk]["mae"], float)
            for i in range(1, K + 1):
                rows.append(dict(variant=name, seed=s, iteration=i,
                                 gap=g[i] - b[i], guided=g[i], random=b[i],
                                 alpha=r["sched"]["mix"][i - 1],
                                 sigma=r["sched"]["sigma"][i - 1],
                                 severity=r["sched"]["severity"][i - 1],
                                 area=r["det"]["area"][i - 1]))
        print("  ran:", name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cached, index=False)
    return df


def band(ax, sub, col, colour, label, dash="-"):
    g = sub.groupby("iteration")[col].agg(["mean", "std", "count"])
    hw = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    ax.fill_between(g.index, g["mean"] - hw, g["mean"] + hw, color=colour,
                    alpha=0.12, lw=0)
    ax.plot(g.index, g["mean"], dash, color=colour, lw=2, ms=4, marker="o",
            label=label)


def fig_k10(df):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axhline(0, color="black", lw=1)
    for i, name in enumerate(VARIANTS_K10):
        s = df[df.variant == name]
        g = s.groupby("iteration").gap.mean()
        ax.plot(g.index, g.values, "-o", ms=3.5, lw=1.8, color=cmap(i), label=name)
    ax.set_xlabel("iteration"); ax.set_ylabel("guided − random MAE")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.95)
    fig.tight_layout(); fig.savefig(OUT / "expl_k10_variants_gap.png"); plt.close(fig)


def fig_k10_summary(df):
    rows = []
    for name in VARIANTS_K10:
        v = df[df.variant == name].groupby("seed").gap.mean()
        rows.append(dict(variant=name, mean=v.mean(),
                         hw=1.96 * v.std(ddof=1) / np.sqrt(len(v))))
    d = pd.DataFrame(rows).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colours = ["#2ca02c" if m + h < 0 else "#d62728" if m - h > 0 else "#7f7f7f"
               for m, h in zip(d["mean"], d.hw)]
    ax.barh(d.variant, d["mean"], xerr=d.hw, color=colours, height=0.62,
            error_kw=dict(lw=1, capsize=3))
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("across-loop guided − random MAE  (negative = guided better)")
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "expl_k10_variants_summary.png"); plt.close(fig)


def fig_a035(df):
    colours = {"static  α=.35": "#7f7f7f", "dynamic  α=.35": "#ff7f0e",
               "self-gate  α=.35": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0, color="black", lw=1)
    for name, c in colours.items():
        band(ax, df[df.variant == name], "gap", c, name)
    ax.set_xlabel("iteration"); ax.set_ylabel("guided − random MAE")
    ax.set_xticks(range(1, 21, 2)); ax.set_ylim(-0.05, 0.07)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout(); fig.savefig(OUT / "expl_a035_k20_gap.png"); plt.close(fig)


def fig_a035_controls(df):
    colours = {"static  α=.35": "#7f7f7f", "dynamic  α=.35": "#ff7f0e",
               "self-gate  α=.35": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for name, c in colours.items():
        s = df[df.variant == name]
        axes[0].plot(s.groupby("iteration").alpha.mean(), "-o", ms=3, lw=1.8,
                     color=c, label=name)
        axes[1].plot(s.groupby("iteration").sigma.mean(), "-o", ms=3, lw=1.8,
                     color=c, label=name)
    axes[0].set_ylabel("applied rehearsal mix α"); axes[1].set_ylabel("applied kernel width σ")
    for a in axes:
        a.set_xlabel("iteration"); a.set_xticks(range(1, 21, 2)); a.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "expl_a035_k20_controls.png"); plt.close(fig)


def fig_signals():
    """The two candidate gate inputs, from the saved 20-round runs."""
    d = pd.concat([pd.read_csv(RES / f"sweep__{n}.csv", low_memory=False)
                   for n in ("iter20_anchor", "iter20_schedules")])
    d = d[d.iteration >= 1]
    g = d.groupby("iteration")[["severity", "det_area", "det_iou"]].mean()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(g.index, g.severity, "-o", ms=4, lw=2, color="#8c564b",
            label="severity  (error inside detected region ÷ outside)")
    ax.plot(g.index, g.det_area, "-s", ms=4, lw=2, color="#9467bd",
            label="detected area  (fraction of the input square)")
    ax.axhline(1.0, color="#8c564b", lw=1, ls=":")
    ax.annotate("severity = 1: the detected region is no worse than the rest",
                xy=(10, 1.0), xytext=(6, 1.35), fontsize=7.5, color="#8c564b",
                arrowprops=dict(arrowstyle="->", color="#8c564b", lw=0.8))
    ax.set_xlabel("iteration"); ax.set_ylabel("signal value")
    ax.set_xticks(range(1, 21, 2))
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout(); fig.savefig(OUT / "expl_signals.png"); plt.close(fig)


if __name__ == "__main__":
    print("collecting K=10 variants…", flush=True)
    d10 = collect(VARIANTS_K10, 10, "k10_variants")
    print("collecting alpha=0.35 K=20 variants…", flush=True)
    d20 = collect(VARIANTS_A035, 20, "a035_k20")
    fig_k10(d10); fig_k10_summary(d10)
    fig_a035(d20); fig_a035_controls(d20)
    fig_signals()
    for p in sorted(OUT.glob("*")):
        print("  ", p.name)
