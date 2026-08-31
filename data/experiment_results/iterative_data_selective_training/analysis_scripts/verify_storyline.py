"""Verify the paper storyline against the existing iterative sweep CSVs.

Checks, in storyline order:
 1. Round-1 guided win (conditions map + anchors).
 2. Degradation from round 2 on.
 3. Mechanism: IoU / severity collapse after round 1.
 4. Fixes: constant vs exponential-dump vs adaptive schedules (mix adaptation + rate).
 5. What the existing UNDERTRAINED runs (i3n50 / i5n50, K=20) already show.
 6. What the existing POOL-DEFICIT cells (conditions map) already show.
"""
import pandas as pd
import numpy as np

DIR = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application\data\experiment_results\iterative_data_selective_training"

def load(name):
    df = pd.read_csv(f"{DIR}\\sweep__{name}.csv")
    return df[df["iteration"] >= 0]

def per_iter(df, by=None, val="gap_mae", maxit=8):
    d = df[df["iteration"] > 0]
    if by:
        t = d.pivot_table(index=by, columns="iteration", values=val, aggfunc="mean")
    else:
        t = d.groupby("iteration")[val].mean().to_frame().T
    return t.iloc[:, :maxit].round(3)

def auc(df, by):
    d = df[df["iteration"] > 0]
    g = d.groupby(by + ["param_key"])["gap_mae"].mean().reset_index()
    s = g.groupby(by)["gap_mae"].agg(["mean", "sem", "count"])
    return s.round(4)

print("=" * 100)
print("A) CONDITIONS MAP: per-iteration mean gap (guided - random; neg = guided ahead)")
cm = load("iter_conditions_map")
print("overall:"); print(per_iter(cm))
print("\nby pool_deficit:"); print(per_iter(cm, ["pool_deficit"]))
print("\nby pool_deficit x mix_ratio:"); print(per_iter(cm, ["pool_deficit", "mix_ratio"]))
print("\nIoU by iteration (mean):")
print(cm[cm.iteration > 0].groupby("iteration")["det_iou"].mean().round(3).to_frame().T)
print("\nseverity by iteration (mean):")
print(cm[cm.iteration > 0].groupby("iteration")["severity"].mean().round(3).to_frame().T)

print("\n" + "=" * 100)
print("B) MIX ADAPTATION (alpha0 x schedule): across-loop AUC gap")
ma = load("iter_mix_adaptation")
print(auc(ma, ["mix_ratio", "mix_schedule"]))
print("\niteration-1 gap by schedule (should all share the round-1 repair):")
print(per_iter(ma, ["mix_ratio", "mix_schedule"], maxit=4))

print("\nMIX RATE (alpha0=1):")
mr = load("iter_mix_rate")
print(auc(mr, ["mix_schedule", "mix_rate"]))

print("\n" + "=" * 100)
print("C) UNDERTRAINED anchors, K=20, constant alpha=0.5 (existing runs)")
for name in ["iter20_anchor", "iter20_anchor_i3n50", "iter20_anchor_i5n50"]:
    a = load(name)
    d = a[a.iteration > 0]
    seeds = d.groupby(["seed", "param_key"])["gap_mae"].mean()
    it1 = a[a.iteration == 1]["gap_mae"]
    late = a[a.iteration >= 2]["gap_mae"]
    print(f"\n{name}: init_mae={a['init_mae'].mean():.3f}  "
          f"it1 gap={it1.mean():+.4f}  it2+ gap={late.mean():+.4f}  "
          f"auc={seeds.mean():+.4f}±{seeds.sem():.4f}  "
          f"seeds guided ahead={np.mean(seeds < 0):.0%} of {len(seeds)}")
    print("  per-iter gap 1..10:", per_iter(a, maxit=10).values.round(3))

print("\n" + "=" * 100)
print("D) SCHEDULES at undertrained anchors (iter20_schedules_i3n50 / i5n50)")
for name in ["iter20_schedules", "iter20_schedules_i3n50", "iter20_schedules_i5n50"]:
    s = load(name)
    print(f"\n{name} (grid: " + ", ".join(
        f"{c}={sorted(s[c].unique())}" for c in ["mix_schedule", "sigma_schedule"]
        if c in s and s[c].nunique() > 1) + ")")
    print(auc(s, ["mix_schedule", "sigma_schedule"]))
