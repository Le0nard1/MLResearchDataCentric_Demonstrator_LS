"""
95% bootstrap confidence intervals for every cell of Table 1 (Section 4.4).

Statistic as in table_comparison.py: per condition 100 * mean paired difference to the
unguided random baseline / mean baseline error, averaged over the conditions of a column
(four region conditions per clean/outlier column; eight per further dataset; 24 for All).
Seeds are resampled with replacement within each condition, the same draw for every
method (paired); 2000 resamples, percentile intervals. Prints the half-widths as LaTeX rows.

    python -m scripts.dataselect.table_comparison_ci
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.dataselect import table_comparison as T

RES = T.RES
B = 2000
ROWS = [("uniform", "Uniform"), None,
        ("kcenter", "k-center"), ("kcenter_a1", "k-center ($\\alpha=1$)"),
        ("craig_w_a1", "CRAIG ($\\alpha=1$, weighted)"), ("loss", "Loss"), ("jtt", "JTT"),
        ("craig", "CRAIG"), ("glister", "GLISTER"), ("rho", "RHO-LOSS"),
        ("rho_a1", "RHO-LOSS ($\\alpha=1$)"), None,
        ("ws_default", "Weakspot, default"), ("ws_tuned", "Weakspot, tuned"),
        ("rho_landscape_a1", "RHO-landscape ($\\alpha=1$)"),
        ("rho_filter_a1", "RHO-filter ($\\alpha=1$)"),
        ("kc_land", "Weakspot-weighted k-center"),
        ("kc_land_perm", "\\quad permuted errors (control)"), None,
        ("all_data", "All data")]
MAIN = ["synth2d", "houses", "medical_charges"]
BENCH = ["brazilian_houses", "diamonds", "sulfur", "nyc_taxi"]


def load():
    parts = [T.load(), pd.read_csv(RES / "budget_v2_bench.csv")]
    for f in ["pilot_craig_glister.csv", "pilot_craig_glister_bench.csv", "hybrid_main.csv",
              "hybrid_bench.csv"]:
        x = pd.read_csv(RES / f)
        parts.append(x[x.arm != "uniform"])
    d = pd.concat(parts, ignore_index=True)
    return d[d.arm != "ERROR"].drop_duplicates(T.KEYS + ["arm"])


def main():
    d = load()
    arms = [r[0] for r in ROWS if r]
    rng = np.random.RandomState(0)
    boot = {}                                   # (dataset, region, noise) -> {arm: (B,)}
    point = {}
    for key, g in d.groupby(["dataset", "region", "noise"]):
        ref = g[g.arm == "random"].set_index("seed").mae
        seeds = ref.index.to_numpy()
        idx = rng.randint(len(seeds), size=(B, len(seeds)))
        r = ref.to_numpy()
        rb = r[idx].mean(1)
        boot[key], point[key] = {}, {}
        for arm in arms:
            x = g[g.arm == arm].set_index("seed").mae
            if x.empty:
                continue
            x = x.loc[seeds].to_numpy()
            point[key][arm] = 100 * (r - x).mean() / r.mean()
            boot[key][arm] = 100 * (r[idx] - x[idx]).mean(1) / rb

    def cell(arm, keys):
        keys = [k for k in keys if arm in boot[k]]
        if not keys:
            return None
        pt = np.mean([point[k][arm] for k in keys])
        b = np.mean([boot[k][arm] for k in keys], axis=0)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return pt, lo, hi

    cols = [[k for k in boot if k[0] == ds and k[2] == nz] for ds in MAIN
            for nz in ("clean", "outlier")]
    cols += [[k for k in boot if k[0] == ds] for ds in BENCH]
    cols += [[k for k in boot if k[0] in MAIN]]
    out = []
    for item in ROWS:
        if item is None:
            out.append(r"\midrule")
            continue
        arm, lab = item
        cells = []
        for keys in cols:
            c = cell(arm, keys)
            cells.append("--" if c is None else f"$\\pm{(c[2] - c[1]) / 2:.1f}$")
        out.append(f"{lab:<32}& " + " & ".join(cells) + r" \\")
    print("\n".join(out))
    # check: the point estimates reproduce Table 1
    print("\ncheck kc_land All:", round(cell("kc_land", cols[-1])[0], 2),
          " ws_tuned All:", round(cell("ws_tuned", cols[-1])[0], 2))


if __name__ == "__main__":
    main()
