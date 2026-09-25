"""
Summary of the budgeted-curation experiment (budget.py, paper Section 4.3).

Per (dataset, rho, noise) and arm: error reduction vs. random selection in %, whole
test set and inside the under-represented region, with paired Wilcoxon signed-rank
tests over seeds, Holm-corrected within each (dataset, rho, noise, metric) over the arms.
Also the diagnostics of each condition: initial inside/outside error ratio and how
often the detected centre lies in the region.

    python -m scripts.dataselect.summarise_budget
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path("data/experiment_results/data_selective_training")
IN, OUT = RES / "budget.csv", RES / "budget_summary.csv"
KEYS = ["dataset", "rho", "noise"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def main():
    df = pd.read_csv(IN)
    df = df[df.arm != "ERROR"]
    df["arm_a"] = np.where(df.alpha.notna() & ~df.arm.isin(["random", "stratified"]),
                           df.arm + " (" + df.alpha.map("{:g}".format) + ")", df.arm)
    rows = []
    for key, g in df.groupby(KEYS):
        ref = g[g.arm == "random"].set_index("seed")
        diag = dict(zip(KEYS, key), n_seeds=len(ref),
                    init_ratio=float((ref.init_in / ref.init_out).median()),
                    detect_in=float(g.drop_duplicates("seed").detect_in_region.mean()),
                    random_mae=float(ref.mae.mean()), random_in=float(ref.err_in.mean()))
        block = []
        for arm, a in g[g.arm != "random"].groupby("arm_a"):
            a = a.set_index("seed").loc[ref.index]
            r = dict(diag, arm=arm)
            for m in ("mae", "err_in"):
                r[f"{m}_red"] = float(100 * (ref[m].mean() - a[m].mean()) / ref[m].mean())
                d = ref[m] - a[m]
                r[f"{m}_p"] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            r["n_in_region"] = float(a.n_in_region.mean())
            block.append(r)
        for m in ("mae", "err_in"):
            adj = holm([r[f"{m}_p"] for r in block])
            for r, q in zip(block, adj):
                r[f"{m}_p_holm"] = float(q)
        rows += block
    s = pd.DataFrame(rows)
    s.to_csv(OUT, index=False)

    star = lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else "")
    s["cell"] = [f"{a:+.1f}{star(pa)} ({b:+.1f}{star(pb)})" for a, pa, b, pb in
                 zip(s.mae_red, s.mae_p_holm, s.err_in_red, s.err_in_p_holm)]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    diag = s.drop_duplicates(KEYS)[KEYS + ["n_seeds", "init_ratio", "detect_in",
                                            "random_mae", "random_in"]]
    print(diag.round(3).to_string(index=False))
    for (ds,), g in s.groupby(["dataset"]):
        print(f"\n=== {ds}: % vs random, whole (inside region)")
        print(g.pivot_table(index="arm", columns=["noise", "rho"], values="cell",
                            aggfunc="first").to_string())


ARMS_TEX = [("stratified", "Stratified"), ("kcenter (0.2)", "k-center (0.2)"),
            ("density (0.2)", "Density (0.2)"), ("loss (0.2)", "Loss (0.2)"),
            ("rho (0.2)", "RHO-LOSS (0.2)"), ("rho_landscape (0.2)", "RHO-landscape (0.2)"),
            ("weakspot (0.2)", "Weakspot (0.2)"), ("kcenter (1)", "k-center (1)"),
            ("rho (1)", "RHO-LOSS (1)"), ("weakspot (1)", "Weakspot (1)"),
            ("all_data", "All data")]
DS_TEX = [("synth2d", "Synthetic"), ("houses", "California"), ("medical_charges", "Medical")]


def latex(s, rho, inside=False):
    """Table body: arms x (dataset, noise), % vs random at one rho."""
    m = "err_in" if inside else "mae"
    star = lambda p: "^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else "")
    lines = []
    for arm, lab in ARMS_TEX:
        cells = []
        for ds, _ in DS_TEX:
            for nz in ("clean", "outlier"):
                r = s[(s.dataset == ds) & (s.noise == nz) & (s.rho == rho) & (s.arm == arm)]
                r = r.iloc[0]
                cells.append(f"${r[m + '_red']:+.1f}{star(r[m + '_p_holm'])}$")
        lines.append(f"{lab:<22}& " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
