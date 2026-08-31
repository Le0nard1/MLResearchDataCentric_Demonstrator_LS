"""Money-run analysis: the paper's four-policy table at the scarcity operating
point (30 seeds, K=8) and the K=20 persistence check."""
import numpy as np
import pandas as pd
from scipy import stats

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"
pd.set_option("display.width", 250)

def cell_stats(g):
    v = g["gap_mae"].values
    t, p = stats.ttest_1samp(v, 0.0)
    return pd.Series({"auc_gap": v.mean(), "sem": stats.sem(v), "t": t, "p": p,
                      "frac_ahead": np.mean(v < 0), "n": len(v)})

print("=" * 110)
print("MONEY RUN (K=8, 30 seeds, deficit 0.98, n_train 30, n_select 100)")
mp = pd.read_csv(f"{DIR}\\sweep__iter_money_policies.csv")
mp = mp[mp["iteration"] > 0]
AX = ["radius", "sel_sigma", "mix_schedule"]
ps = mp.groupby(AX + ["seed"])["gap_mae"].mean().reset_index()
print("\nacross-loop auc gap by geometry x policy:")
print(ps.groupby(AX).apply(cell_stats, include_groups=False).round(4).to_string())

print("\nfinal-iteration (K=8) gap by geometry x policy:")
fin = mp[mp["iteration"] == 8].groupby(AX + ["seed"])["gap_mae"].mean().reset_index()
print(fin.groupby(AX).apply(cell_stats, include_groups=False).round(4).to_string())

print("\nper-iteration gap, Adaptive severity, both showcase geometries:")
for (r, s) in [(0.25, 0.15), (0.35, 0.3)]:
    sub = mp[(mp["mix_schedule"] == "Adaptive (weakspot severity)")
             & (mp["radius"] == r) & (mp["sel_sigma"] == s)]
    prof = sub.groupby("iteration")[["gap_mae", "gap_err_in", "mix_used", "severity"]].mean()
    print(f"\nradius {r}, sigma {s}:")
    print(prof.round(3).T.to_string())

print("\n" + "=" * 110)
print("MONEY LONG RUN (K=20, 15 seeds)")
lr = pd.read_csv(f"{DIR}\\sweep__iter_money_longrun.csv")
lr = lr[lr["iteration"] > 0]
ps2 = lr.groupby(AX + ["seed"])["gap_mae"].mean().reset_index()
print("\nacross-loop auc gap:")
print(ps2.groupby(AX).apply(cell_stats, include_groups=False).round(4).to_string())
print("\nper-iteration gap (Adaptive severity):")
a = lr[lr["mix_schedule"] == "Adaptive (weakspot severity)"]
print(a.pivot_table(index=["radius", "sel_sigma"], columns="iteration",
                    values="gap_mae", aggfunc="mean").round(3).to_string())
print("\nper-iteration err_in gap (Adaptive severity):")
print(a.pivot_table(index=["radius", "sel_sigma"], columns="iteration",
                    values="gap_err_in", aggfunc="mean").round(2).to_string())
for (r, s) in [(0.25, 0.15), (0.35, 0.3)]:
    v = lr[(lr["mix_schedule"] == "Adaptive (weakspot severity)") & (lr["radius"] == r)
           & (lr["sel_sigma"] == s) & (lr["iteration"] == 20)].groupby("seed")["gap_mae"].mean()
    if len(v):
        t, p = stats.ttest_1samp(v.values, 0)
        print(f"final-iter K=20 (r={r}, s={s}): {v.mean():+.4f}±{stats.sem(v.values):.4f} "
              f"t={t:.2f} p={p:.4f} seeds ahead {np.mean(v.values < 0):.0%}")
