"""Rank the guided-wins-search cells: where does guided actually beat random
ACROSS the loop, not just at round 1?"""
import numpy as np
import pandas as pd
from scipy import stats

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"
AX = ["mix_schedule", "mix_ratio", "iters_initial", "n_train", "n_select", "pool_deficit"]

df = pd.read_csv(f"{DIR}\\sweep__iter_guided_wins_search.csv")
df = df[df["iteration"] > 0]
print(f"{df['param_key'].nunique()} trajectories on disk")

# one auc_gap per seed per cell
per_seed = df.groupby(AX + ["seed"])["gap_mae"].mean().reset_index()

def cell_stats(g):
    v = g["gap_mae"].values
    t, p = stats.ttest_1samp(v, 0.0) if len(v) > 2 else (np.nan, np.nan)
    return pd.Series({"auc_gap": v.mean(), "sem": stats.sem(v) if len(v) > 1 else np.nan,
                      "t": t, "p": p, "frac_ahead": np.mean(v < 0), "n": len(v)})

cells = per_seed.groupby(AX).apply(cell_stats, include_groups=False).reset_index()
cells = cells.sort_values("auc_gap")

pd.set_option("display.width", 250)
print("\n=== TOP 15 cells (most guided-ahead across the loop) ===")
print(cells.head(15).round(4).to_string(index=False))

print("\n=== BOTTOM 5 (worst) ===")
print(cells.tail(5).round(4).to_string(index=False))

print("\n=== marginal effect of each axis (mean auc_gap) ===")
for a in AX:
    print(f"\n{a}:")
    print(per_seed.groupby(a)["gap_mae"].agg(["mean", "sem"]).round(4))

print("\n=== interaction: pool_deficit x iters_initial x mix_schedule (mean auc_gap) ===")
print(per_seed.pivot_table(index=["pool_deficit", "iters_initial"],
                           columns="mix_schedule", values="gap_mae",
                           aggfunc="mean").round(4))

# per-iteration profile for the top 3 cells
print("\n=== per-iteration gap profile, top 3 cells ===")
for _, r in cells.head(3).iterrows():
    m = np.ones(len(df), dtype=bool)
    for a in AX:
        m &= df[a] == r[a]
    prof = df[m].groupby("iteration")[["gap_mae", "gap_err_in", "det_iou", "mix_used"]].mean()
    print("\ncell:", {a: r[a] for a in AX})
    print(prof.round(3).T)
