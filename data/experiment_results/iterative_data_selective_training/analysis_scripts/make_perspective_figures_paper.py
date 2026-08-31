"""Figures for the paper's perspectives section.

fig_es:   the retraining-protocol trade-off. (a) standard setting, static mix:
          per-iteration gap with the fixed 400-epoch budget vs early stopping -
          ES makes the round-1 whole-area win visible and the rest noisy.
          (b) scarce setting: across-loop gap of the three policies under both
          protocols - ES destroys the scarcity victory.
fig_arch: across-loop gap of the three policies against network architecture
          (sorted by parameter count), standard and scarce settings.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RES = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training")
PAPER = Path(r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Documents\Paper_DataSelectiveTrainingOnWeakspots\figures")
OUT = RES / "figures" / "Storyline"

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

SEV = "Adaptive (weakspot severity)"
POL = [("Constant", "static"), ("Exponential decay", "dynamic"), (SEV, "adaptive")]


def load(csv, **f):
    df = pd.read_csv(RES / f"sweep__{csv}.csv", low_memory=False)
    df = df[df["iteration"] > 0]
    for k, v in f.items():
        df = df[df[k] == v]
    return df


def ci_iter(d, col="gap_mae"):
    g = d.groupby("iteration")[col].agg(["mean", "std", "count"])
    g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    return g.reset_index()


def auc_ci(d):
    v = d.groupby("seed")["gap_mae"].mean().values
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)


# ── fig_es ────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.9, 5.6))

for name, lab, c in ((load("iter_paper_normal", mix_schedule="Constant"),
                      "fixed 400-epoch budget", "#1f77b4"),
                     (load("iter_es_normal", mix_schedule="Constant"),
                      "early stopping", "#ff7f0e")):
    g = ci_iter(name)
    ax1.fill_between(g.iteration, g["mean"] - g.hw, g["mean"] + g.hw,
                     color=c, alpha=0.15, lw=0)
    ax1.plot(g.iteration, g["mean"], "-o", ms=3.5, lw=1.8, color=c, label=lab)
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xlabel("iteration"); ax1.set_ylabel("gap (guided − random MAE)")
ax1.set_title("(a) standard setting, static mix", fontsize=8.5)
ax1.legend(fontsize=7)

width = 0.36
x = np.arange(3)
for off, csv, filt, lab, c in ((-width / 2, "iter_money_policies",
                                dict(radius=0.25, sel_sigma=0.15),
                                "fixed 400-epoch budget", "#1f77b4"),
                               (width / 2, "iter_es_scarce", {},
                                "early stopping", "#ff7f0e")):
    for i, (pol, _) in enumerate(POL):
        m, hw = auc_ci(load(csv, mix_schedule=pol, **filt))
        ax2.bar(x[i] + off, m, width, yerr=hw, capsize=3, color=c,
                label=lab if i == 0 else None)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels([l for _, l in POL])
ax2.set_ylabel("across-loop gap")
ax2.set_title("(b) scarce-pool setting, all policies", fontsize=8.5)
ax2.legend(fontsize=7)
fig.tight_layout()
for t in (OUT, PAPER):
    fig.savefig(t / "fig_perspective_es.png", dpi=160)
plt.close(fig)
print("fig_perspective_es.png written")

# ── fig_arch ──────────────────────────────────────────────────────────────
try:
    an = load("iter_arch_normal")
    asc = load("iter_arch_scarce")
except FileNotFoundError:
    print("architecture sweeps not on disk yet - skipped")
    raise SystemExit

order = (an.groupby("arch")["n_model_params"].first().sort_values())
labels = [a.split(" (")[0] for a in order.index]
COLS = {"Constant": "#1f77b4", "Exponential decay": "#ff7f0e", SEV: "#2ca02c"}

fig, axes = plt.subplots(2, 1, figsize=(4.9, 5.6), sharex=True)
for ax, d, title in ((axes[0], an, "(a) standard setting"),
                     (axes[1], asc, "(b) scarce-pool setting")):
    for pol, lab in POL:
        ms, hws = [], []
        for a in order.index:
            m, hw = auc_ci(d[(d["arch"] == a) & (d["mix_schedule"] == pol)])
            ms.append(m); hws.append(hw)
        ax.errorbar(range(len(order)), ms, yerr=hws, marker="o", ms=4,
                    capsize=3, lw=1.6, color=COLS[pol], label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("across-loop gap")
    ax.set_title(title, fontsize=8.5)
axes[0].legend(fontsize=7)
axes[1].set_xticks(range(len(order)))
axes[1].set_xticklabels(
    [f"{l}\n{f'{p/1000:.0f}k' if p >= 1000 else int(p)}"
     for l, p in zip(labels, order)], fontsize=6.5)
axes[1].set_xlabel("architecture (parameters)")
fig.tight_layout()
for t in (OUT, PAPER):
    fig.savefig(t / "fig_perspective_arch.png", dpi=160)
plt.close(fig)
print("fig_perspective_arch.png written")

print("\n=== arch numbers ===")
for nm, d in (("normal", an), ("scarce", asc)):
    print(nm)
    for pol, lab in POL:
        row = []
        for a in order.index:
            m, hw = auc_ci(d[(d["arch"] == a) & (d["mix_schedule"] == pol)])
            row.append(f"{a.split(' (')[0]}:{m:+.3f}")
        print(f"  {lab:9s}", "  ".join(row))
