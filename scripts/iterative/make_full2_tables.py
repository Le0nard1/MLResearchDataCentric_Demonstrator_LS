"""
LaTeX tables for Paper B, Section 4 (full run ``full2``): mean error reduction over the
eight rounds against random selection, Wilcoxon signed-rank over runs, Holm-corrected
over all cells of a table; bold where the corrected p < 0.01.

    python -m scripts.iterative.make_full2_tables
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.iterative import iter_budget as IB

K = ["dataset", "region", "seed", "round"]
DS = [("synth2d", "Syn."), ("brazilian_houses", "Braz."), ("diamonds", "Diam."),
      ("houses", "Calif."), ("medical_charges", "Med."), ("nyc_taxi", "Taxi"),
      ("sulfur", "Sulf.")]
ARMS = [("ws_default", r"Static, $\alpha=0.2$"), ("ws_half", r"Static, $\alpha=0.5$"),
        ("ws_tuned", r"Static, $\alpha=1$"), ("ws_dynamic", r"Dynamic, $\alpha=1$"),
        ("ws_adaptive", r"Severity-adaptive, $\alpha=1$"),
        ("ws_half_adaptive", r"Severity-adaptive, $\alpha=0.5$"),
        ("ws_half_size_pow05", r"Size-adaptive (power $0.5$), $\alpha=0.5$"),
        ("ws_half_size_pow2", r"Size-adaptive (power $2$), $\alpha=0.5$"),
        ("ws_half_size_switch", r"Size-adaptive (switch), $\alpha=0.5$"),
        ("ws_size_switch", r"Size-adaptive (switch), $\alpha=1$"),
        ("ws_half_size_pow2_fixedreg", r"Size-adaptive $\alpha$ ($r^{2}$), fixed region"),
        ("ws_half_adaptreg", r"Static $\alpha=0.5$, adaptive region"),
        ("hybrid_fixedreg", r"Focus + coverage, fixed region"),
        ("kcenter", r"Iterated k-center (coverage)"),
        ("kcenter_half_uniform", r"k-center half + uniform half"),
        ("uniform_sched_kc", r"k-center + uniform, matched share"),
        ("hybrid_a025", r"Focus + coverage, $\alpha=0.25$"),
        ("hybrid_a075", r"Focus + coverage, $\alpha=0.75$"),
        ("toperr", r"Top error"),
        ("toperr_kc", r"Top error + coverage"),
        ("kc_land", r"Weakspot-weighted k-center"),
        ("hybrid_size_pow2_kc", r"Focus + coverage")]
# Compact main-text table; the full list goes to the appendix.
MAIN = [("ws_half", r"Static focus, $\alpha=0.5$"), ("ws_dynamic", r"Dynamic, $\alpha=1$"),
        ("ws_half_adaptive", r"Severity-adaptive, $\alpha=0.5$"),
        ("ws_half_size_pow2", r"Size-adaptive ($r^{2}$), $\alpha=0.5$"),
        ("ws_half_adaptreg", r"Static focus, elevated area"),
        ("kcenter", r"k-center (coverage)"),
        ("kcenter_half_uniform", r"k-center half + uniform half"),
        ("hybrid_size_pow2_kc", r"Focus + coverage")]
PAIRS = [("hybrid_size_pow2_kc", "kcenter_half_uniform"), ("hybrid_size_pow2_kc", "kcenter"),
         ("hybrid_size_pow2_kc", "hybrid_fixedreg"), ("kcenter_half_uniform", "kcenter"),
         ("hybrid_size_pow2_kc", "uniform_sched_kc"), ("uniform_sched_kc", "kcenter"),
         ("uniform_sched_kc", "kcenter_half_uniform"),
         ("kc_land", "hybrid_size_pow2_kc"), ("kc_land", "kcenter"),
         ("hybrid_size_pow2_kc", "toperr_kc"), ("toperr_kc", "uniform_sched_kc"),
         ("toperr_kc", "kcenter"), ("toperr", "ws_tuned"),
         ("hybrid_a025", "hybrid_size_pow2_kc"), ("hybrid_a075", "hybrid_size_pow2_kc")]
COND = [("sparse_init:0.25:std", "Repairable"), ("sparse_all:0.25:std", r"Scarce, $\rho=0.25$"),
        ("sparse_all:0.1:std", r"Scarce, $\rho=0.1$"), ("sparse_all:0.03:std", r"Scarce, $\rho=0.03$"),
        ("sparse_init:0.25:short", "Repairable, short init."),
        ("sparse_all:0.1:short", r"Scarce $\rho=0.1$, short init.")]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def ci(v, B=10000, seed=0):
    """95% percentile bootstrap interval of the mean over independent runs."""
    v = np.asarray(v, float)
    rs = np.random.default_rng(seed)
    m = v[rs.integers(0, len(v), (B, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def per_run_gain(d):
    r = d[d.arm == "random"].groupby(K).mae.mean()
    out = []
    for arm, x in d[~d.arm.isin(["random", "random_new"])].groupby("arm"):
        w = x.set_index(K).mae
        g = (100 * (r - w) / r).dropna().rename("G").reset_index()
        g = g[g["round"] > 0]
        out.append(g.groupby(["dataset", "region", "seed"]).G.mean().reset_index().assign(arm=arm))
    return pd.concat(out)


def table(cells, rows, cols, row_lab, col_lab):
    keys = [(r, c) for r in rows for c in cols]
    means = {k: cells[k].mean() for k in keys}
    p = holm([wilcoxon(cells[k]).pvalue if np.any(cells[k] != 0) else 1.0 for k in keys])
    padj = dict(zip(keys, p))
    lines = []
    for r in rows:
        vals = []
        for c in cols:
            v = f"{means[(r, c)]:.1f}"
            vals.append(rf"$\mathbf{{{v}}}$" if padj[(r, c)] < 0.01 else f"${v}$")
        lines.append(f"{row_lab[r]} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines)


def absolute(d):
    """Appendix table of absolute errors and the retraining-regime figures of Section 4.5:
    test MAE of the initial model and after round 8, replay against new points only."""
    K = ["dataset", "region", "seed"]
    fin = d[d["round"] == 8].groupby(K + ["arm"]).mae.mean().unstack("arm")
    init = d[(d["round"] == 0) & (d.arm == "random")].groupby(K).mae.mean()
    cols = ["random_new", "random", "ws_half", "ws_half_size_pow2", "kcenter"]
    tab = pd.concat([init.rename("initial"), fin[[c for c in cols if c in fin]]], axis=1)
    names = dict(DS) | {"synth2d": "Synthetic", "brazilian_houses": "Brazilian houses",
                        "diamonds": "Diamonds", "houses": "California housing",
                        "medical_charges": "Medical charges", "nyc_taxi": "NYC taxi trips",
                        "sulfur": "Sulfur"}
    print("\n% Table: absolute errors (initial, new only, random, static 0.5, size r^2, k-center)")
    m = tab.groupby("dataset").mean()
    for ds, _ in DS:
        if ds in m.index:
            print(f"{names[ds]} & " + " & ".join(f"${v:.3f}$" for v in m.loc[ds]) + r" \\")
    real = tab.reset_index()
    real = real[real.dataset != "synth2d"]
    print("% Retraining regime, real datasets, random selection: replay vs new points only "
          f"{100 * ((real.random - real.random_new) / real.random_new).mean():+.1f}% final MAE; "
          f"below the initial MAE in {100 * (real.random < real.initial).mean():.0f}% "
          f"(replay) vs {100 * (real.random_new < real.initial).mean():.0f}% (new only) of runs")


def main():
    d = pd.read_csv(IB.RES / "iter_full2.csv")
    kc = IB.RES / "iter_full2_kc.csv"           # k-center baseline, paired with full2
    if kc.exists():
        d = pd.concat([d, pd.read_csv(kc)])
    abl = IB.RES / "iter_full2_abl.csv"          # ablation and hybrid, paired with full2
    if abl.exists():
        x = pd.read_csv(abl)
        d = pd.concat([d, x[x.arm.isin(["ws_half_size_pow2_fixedreg", "ws_half_adaptreg",
                                        "hybrid_size_pow2_kc"])]])
    for f in ["iter_full2_ctl.csv", "iter_full2_ctl2.csv", "iter_full2_err.csv", "iter_full2_wkc.csv"]:   # controls, alpha variants
        if (IB.RES / f).exists():
            d = pd.concat([d, pd.read_csv(IB.RES / f)])
    absolute(d)
    g = per_run_gain(d)
    have = set(g.arm)
    arms = [(a, l) for a, l in ARMS if a in have]
    main_arms = [(a, l) for a, l in MAIN if a in have]
    cells = {(a, ds): g[(g.arm == a) & (g.dataset == ds)].G.values
             for a, _ in arms for ds, _ in DS}
    print("% Main table: strategies x datasets")
    print(" & ".join(["Strategy"] + [l for _, l in DS]) + r" \\")
    print(table(cells, [a for a, _ in main_arms], [ds for ds, _ in DS], dict(main_arms), dict(DS)))
    print("\n% Appendix table: all strategies x datasets")
    print(table(cells, [a for a, _ in arms], [ds for ds, _ in DS], dict(arms), dict(DS)))
    # paired comparisons between strategies (pooled and per dataset, Holm over datasets)
    run = g.set_index(["dataset", "region", "seed", "arm"]).G.unstack("arm")
    for x, y in PAIRS:
        if x not in run or y not in run:
            continue
        dlt = (run[x] - run[y]).dropna()
        have_ds = set(dlt.index.get_level_values(0))
        per = [(ds, dlt.xs(ds, level=0)) for ds, _ in DS if ds in have_ds]
        padj = holm([wilcoxon(v).pvalue for _, v in per])
        lo, hi = ci(dlt.values)
        print(f"\n% {x} minus {y}: pooled {dlt.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] "
              f"(n={len(dlt)}, p={wilcoxon(dlt).pvalue:.0e})")
        print("%   " + ", ".join(f"{ds}: {v.mean():+.2f}{'*' if p < 0.01 else ''}"
                                for (ds, v), p in zip(per, padj)))
    print("\n% Pooled mean against random selection, 95% bootstrap CI over runs")
    for a, _ in arms:
        v = run[a].dropna().values
        lo, hi = ci(v)
        print(f"%   {a}: {v.mean():.2f} [{lo:.2f}, {hi:.2f}] (n={len(v)})")
    g["kind"] = np.where(g.dataset == "synth2d", "syn", "real")
    sel = ["ws_half", "ws_tuned", "ws_half_adaptreg", "hybrid_size_pow2_kc"]
    cells2 = {}
    for a in sel:
        for reg, _ in COND:
            for kind in ["syn", "real"]:
                cells2[(reg, (a, kind))] = g[(g.arm == a) & (g.region == reg) & (g.kind == kind)].G.values
    cols = [(a, k) for k in ["syn", "real"] for a in sel]
    print("\n% Table: conditions; columns:", cols)
    print(table(cells2, [r for r, _ in COND], cols, dict(COND), {}))


if __name__ == "__main__":
    main()
