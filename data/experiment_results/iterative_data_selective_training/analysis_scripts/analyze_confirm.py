"""15-seed confirmation: schedule x pool_deficit crossover, and the K=20 long run."""
import numpy as np
import pandas as pd
from scipy import stats

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"
pd.set_option("display.width", 250)

def per_seed_auc(df, by):
    d = df[df["iteration"] > 0]
    return d.groupby(by + ["seed"])["gap_mae"].mean().reset_index()

def cell_stats(g):
    v = g["gap_mae"].values
    t, p = stats.ttest_1samp(v, 0.0)
    return pd.Series({"auc_gap": v.mean(), "sem": stats.sem(v), "t": t, "p": p,
                      "frac_ahead": np.mean(v < 0), "n": len(v)})

print("=" * 110)
print("CONFIRM (K=8, 15 seeds): auc gap by mix_schedule x pool_deficit  (neg = guided ahead)")
cf = pd.read_csv(f"{DIR}\\sweep__iter_scarcity_confirm.csv")
cf = cf[cf["iteration"] >= 0]
ps = per_seed_auc(cf, ["mix_schedule", "pool_deficit"])
st = ps.groupby(["mix_schedule", "pool_deficit"]).apply(cell_stats, include_groups=False)
print(st.round(4).to_string())

print("\nauc_gap pivot (rows=deficit, cols=schedule):")
print(ps.pivot_table(index="pool_deficit", columns="mix_schedule",
                     values="gap_mae", aggfunc="mean").round(4))

print("\nFINAL-iteration gap pivot:")
fin = cf[cf["iteration"] == 8]
print(fin.pivot_table(index="pool_deficit", columns="mix_schedule",
                      values="gap_mae", aggfunc="mean").round(4))

print("\nPer-iteration profile at each deficit (Constant schedule):")
c = cf[(cf["mix_schedule"] == "Constant") & (cf["iteration"] > 0)]
print(c.pivot_table(index="pool_deficit", columns="iteration", values="gap_mae",
                    aggfunc="mean").round(3))
print("\ndet_iou by iteration x deficit (Constant):")
print(c.pivot_table(index="pool_deficit", columns="iteration", values="det_iou",
                    aggfunc="mean").round(3))
print("\nseverity by iteration x deficit (Constant):")
print(c.pivot_table(index="pool_deficit", columns="iteration", values="severity",
                    aggfunc="mean").round(2))

print("\n" + "=" * 110)
print("LONG RUN (K=20, deficit 0.98, 15 seeds)")
lr = pd.read_csv(f"{DIR}\\sweep__iter_scarcity_longrun.csv")
lr = lr[lr["iteration"] >= 0]
d = lr[lr["iteration"] > 0]
print("\nper-iteration mean gap:")
print(d.pivot_table(index="mix_schedule", columns="iteration", values="gap_mae",
                    aggfunc="mean").round(3))
print("\nper-iteration mean err_in gap:")
print(d.pivot_table(index="mix_schedule", columns="iteration", values="gap_err_in",
                    aggfunc="mean").round(2))
ps2 = per_seed_auc(lr, ["mix_schedule"])
print("\nacross-loop stats:")
print(ps2.groupby("mix_schedule").apply(cell_stats, include_groups=False).round(4).to_string())
for sch in d["mix_schedule"].unique():
    v = lr[(lr["mix_schedule"] == sch) & (lr["iteration"] == 20)].groupby("seed")["gap_mae"].mean()
    t, p = stats.ttest_1samp(v.values, 0.0)
    print(f"final-iter (K=20) {sch}: gap {v.mean():+.4f}±{stats.sem(v.values):.4f}  "
          f"t={t:.2f} p={p:.4f}  seeds ahead {np.mean(v.values < 0):.0%}")
print("\nabsolute MAE at K=20 (guided vs random, Constant):")
cc = lr[(lr["mix_schedule"] == "Constant") & (lr["iteration"].isin([0, 1, 10, 20]))]
print(cc.groupby("iteration")[["gnew_mae", "rnew_mae", "gnew_err_in", "rnew_err_in"]].mean().round(3))
