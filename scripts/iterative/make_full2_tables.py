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
        ("kcenter", r"Iterated k-center (coverage)")]
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


def main():
    d = pd.read_csv(IB.RES / "iter_full2.csv")
    kc = IB.RES / "iter_full2_kc.csv"           # k-center baseline, paired with full2
    if kc.exists():
        d = pd.concat([d, pd.read_csv(kc)])
    g = per_run_gain(d)
    cells = {(a, ds): g[(g.arm == a) & (g.dataset == ds)].G.values
             for a, _ in ARMS for ds, _ in DS}
    print("% Table: schedules x datasets")
    print(" & ".join(["Strategy"] + [l for _, l in DS]) + r" \\")
    print(table(cells, [a for a, _ in ARMS], [ds for ds, _ in DS], dict(ARMS), dict(DS)))
    g["kind"] = np.where(g.dataset == "synth2d", "syn", "real")
    sel = ["ws_half", "ws_tuned", "ws_half_size_switch", "ws_half_size_pow2"]
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
