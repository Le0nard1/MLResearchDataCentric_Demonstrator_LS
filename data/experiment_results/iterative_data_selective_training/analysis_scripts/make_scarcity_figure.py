"""Checkpoint figure: the scarcity discovery (confirm + longrun sweeps, 15 seeds)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"
OUT = DIR + r"\figures\Exploration\fig_scarcity_guided_wins.png"

cf = pd.read_csv(f"{DIR}\\sweep__iter_scarcity_confirm.csv")
cf = cf[cf["iteration"] > 0]
lr = pd.read_csv(f"{DIR}\\sweep__iter_scarcity_longrun.csv")
lr = lr[lr["iteration"] > 0]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
SCHED_ORDER = ["Constant", "Exponential decay",
               "Adaptive (weakspot severity)", "Adaptive (weakspot size)"]
COLORS = {"Constant": "tab:blue", "Exponential decay": "tab:orange",
          "Adaptive (weakspot severity)": "tab:green",
          "Adaptive (weakspot size)": "tab:red"}

# (a) auc gap vs deficit per schedule
ax = axes[0, 0]
ps = cf.groupby(["mix_schedule", "pool_deficit", "seed"])["gap_mae"].mean().reset_index()
for sch in SCHED_ORDER:
    d = ps[ps["mix_schedule"] == sch].groupby("pool_deficit")["gap_mae"]
    m, s = d.mean(), d.sem()
    ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3,
                label=sch, color=COLORS[sch])
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("pool_deficit (candidate scarcity inside the gap)")
ax.set_ylabel("across-loop gap (guided − random MAE)")
ax.set_title("(a) Scarcity flips the sign — deficit ≈ 0.98 optimal")
ax.legend(fontsize=7)

# (b) severity trajectory per deficit (Constant): the target stays real
ax = axes[0, 1]
c = cf[cf["mix_schedule"] == "Constant"]
for dv, sub in c.groupby("pool_deficit"):
    p = sub.groupby("iteration")["severity"].mean()
    ax.plot(p.index, p.values, marker="o", ms=3, label=f"deficit {dv}")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xlabel("iteration"); ax.set_ylabel("detected severity (err in / err out)")
ax.set_title("(b) Under scarcity the weakspot never heals\n(detector keeps a real target)")
ax.legend(fontsize=7)

# (c) long-run whole-area MAE, guided vs random (Constant, deficit .98)
ax = axes[1, 0]
cc = lr[lr["mix_schedule"] == "Constant"]
g = cc.groupby("iteration")["gnew_mae"].mean(); r = cc.groupby("iteration")["rnew_mae"].mean()
ax.plot(g.index, g.values, marker="o", ms=3, label="guided", color="tab:blue")
ax.plot(r.index, r.values, marker="s", ms=3, label="random", color="tab:gray")
ax.set_xlabel("iteration"); ax.set_ylabel("whole-area MAE")
ax.set_title("(c) K=20, deficit 0.98, constant α=0.5\n(guided slightly ahead throughout)")
ax.legend(fontsize=8)

# (d) long-run in-weakspot error: the persistent advantage
ax = axes[1, 1]
gi = cc.groupby("iteration")["gnew_err_in"].mean(); ri = cc.groupby("iteration")["rnew_err_in"].mean()
ax.plot(gi.index, gi.values, marker="o", ms=3, label="guided", color="tab:blue")
ax.plot(ri.index, ri.values, marker="s", ms=3, label="random", color="tab:gray")
ax.set_xlabel("iteration"); ax.set_ylabel("in-weakspot error")
ax.set_title("(d) In-weakspot error: guided stays ahead\nevery round (−0.10 at K=20)")
ax.legend(fontsize=8)

fig.suptitle("Persistent candidate scarcity (pool_deficit) is what makes iterative guidance pay\n"
             "confirm: 15 seeds, K=8 | longrun: 15 seeds, K=20 | α=0.5, n_train 30, n_select 100, radius 0.25",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=150)
print("saved", OUT)
