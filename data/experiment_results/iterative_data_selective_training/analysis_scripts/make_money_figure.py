"""Final figure: four-policy comparison at the scarcity operating point (30 seeds)
and the K=20 persistence of the severity-gated advantage."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"
OUT = DIR + r"\figures\Exploration\fig_money_policies.png"

SCHED = ["Constant", "Exponential decay",
         "Adaptive (weakspot severity)", "Adaptive (weakspot size)"]
LBL = {"Constant": "Constant", "Exponential decay": "Exp. dump ×0.1",
       "Adaptive (weakspot severity)": "Adaptive (severity)",
       "Adaptive (weakspot size)": "Adaptive (size)"}
COL = {"Constant": "tab:blue", "Exponential decay": "tab:orange",
       "Adaptive (weakspot severity)": "tab:green",
       "Adaptive (weakspot size)": "tab:red"}

mp = pd.read_csv(f"{DIR}\\sweep__iter_money_policies.csv")
mp = mp[(mp["iteration"] > 0) & (mp["radius"] == 0.25) & (mp["sel_sigma"] == 0.15)]
lr = pd.read_csv(f"{DIR}\\sweep__iter_money_longrun.csv")
lr = lr[lr["iteration"] > 0]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) across-loop auc gap per policy, 30 seeds, with 95% CI
ax = axes[0, 0]
for i, sch in enumerate(SCHED):
    v = mp[mp["mix_schedule"] == sch].groupby("seed")["gap_mae"].mean().values
    ci = stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)
    ax.bar(i, v.mean(), yerr=ci, capsize=4, color=COL[sch])
    ax.text(i, 0.002, f"{np.mean(v < 0):.0%} seeds\nahead", ha="center", fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([LBL[s] for s in SCHED], fontsize=7)
ax.set_ylabel("across-loop gap (guided − random MAE)")
ax.set_title("(a) Four policies, 30 seeds, 95% CI\n(deficit 0.98, σ=0.15, radius 0.25, α₀=0.5, K=8)")

# (b) per-iteration gap per policy
ax = axes[0, 1]
for sch in SCHED:
    p = mp[mp["mix_schedule"] == sch].groupby("iteration")["gap_mae"].mean()
    ax.plot(p.index, p.values, marker="o", ms=3, color=COL[sch], label=LBL[sch])
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("iteration"); ax.set_ylabel("MAE gap (guided − random)")
ax.set_title("(b) Per-iteration gap: dump and severity gate\nhold the lead; constant focus decays it")
ax.legend(fontsize=7)

# (c) K=20 persistence: adaptive severity vs constant (mean over 4 geometries)
ax = axes[1, 0]
for sch, c in [("Adaptive (weakspot severity)", "tab:green"), ("Constant", "tab:blue")]:
    p = lr[lr["mix_schedule"] == sch].groupby("iteration")["gap_mae"].mean()
    ax.plot(p.index, p.values, marker="o", ms=3, color=c, label=LBL[sch])
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("iteration"); ax.set_ylabel("MAE gap (guided − random)")
ax.set_title("(c) K=20: the severity gate stays ahead for 20 rounds;\nconstant focus collapses (mean over 4 geometries)")
ax.legend(fontsize=8)

# (d) realised mix of the severity gate (it keeps ~0.2-0.3 focus alive)
ax = axes[1, 1]
a = lr[lr["mix_schedule"] == "Adaptive (weakspot severity)"]
p = a.groupby("iteration")["mix_used"].mean()
ax.plot(p.index, p.values, marker="o", ms=3, color="tab:green", label="α used (severity gate)")
p2 = a.groupby("iteration")["gap_err_in"].mean()
ax2 = ax.twinx()
ax2.plot(p2.index, p2.values, marker="s", ms=3, color="tab:purple", label="in-weakspot gap")
ax2.axhline(0, color="k", lw=0.6, ls=":")
ax2.set_ylabel("in-weakspot error gap", color="tab:purple")
ax.set_xlabel("iteration"); ax.set_ylabel("realised mix α", color="tab:green")
ax.set_title("(d) The gate keeps ~0.2–0.3 focus alive and the\nin-weakspot advantage persists all 20 rounds")

fig.suptitle("Money run: under persistent scarcity (pool_deficit 0.98) guidance beats random across the loop\n"
             "K=8: 30 seeds | K=20: 15 seeds | n_train 30, n_select 100, warm start, new-only, Quantile Regression",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=150)
print("saved", OUT)
