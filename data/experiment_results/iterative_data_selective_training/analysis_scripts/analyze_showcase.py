"""Showcase search: which (radius, sigma, alpha, deficit, schedule) cell gives the
strongest, statistically solid across-the-loop guided win?"""
import numpy as np
import pandas as pd
from scipy import stats

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"
AX = ["mix_schedule", "mix_ratio", "radius", "sel_sigma", "pool_deficit"]
pd.set_option("display.width", 250)

df = pd.read_csv(f"{DIR}\\sweep__iter_showcase_search.csv")
df = df[df["iteration"] > 0]
print(f"{df.groupby(AX + ['seed']).ngroups} trajectories")

ps = df.groupby(AX + ["seed"])["gap_mae"].mean().reset_index()

def cell_stats(g):
    v = g["gap_mae"].values
    t, p = stats.ttest_1samp(v, 0.0)
    return pd.Series({"auc_gap": v.mean(), "sem": stats.sem(v), "t": t, "p": p,
                      "frac_ahead": np.mean(v < 0), "n": len(v)})

cells = ps.groupby(AX).apply(cell_stats, include_groups=False).reset_index().sort_values("auc_gap")
print("\n=== ALL cells ranked (most guided-ahead first), top 20 ===")
print(cells.head(20).round(4).to_string(index=False))

print("\n=== marginals ===")
for a in AX:
    print(f"{a}: " + "  ".join(f"{k}={v:+.4f}" for k, v in ps.groupby(a)["gap_mae"].mean().items()))

# best cell detail
best = cells.iloc[0]
m = np.ones(len(df), dtype=bool)
for a in AX:
    m &= df[a] == best[a]
sub = df[m]
print("\n=== BEST CELL ===")
print({a: best[a] for a in AX}, f"auc {best['auc_gap']:+.4f}±{best['sem']:.4f} p={best['p']:.4f}")
prof = sub.groupby("iteration")[["gap_mae", "gap_err_in", "det_iou", "severity",
                                 "gnew_mae", "rnew_mae", "gnew_err_in", "rnew_err_in"]].mean()
print(prof.round(3).T.to_string())
fin = sub[sub["iteration"] == 8].groupby("seed")["gap_mae"].mean()
t, p = stats.ttest_1samp(fin.values, 0)
print(f"final-iter gap {fin.mean():+.4f}±{stats.sem(fin.values):.4f} t={t:.2f} p={p:.4f} "
      f"seeds ahead {np.mean(fin.values < 0):.0%}")

# also best per schedule for the paper's 4-policy comparison
print("\n=== best cell per schedule ===")
print(cells.groupby("mix_schedule").first().round(4).to_string())
