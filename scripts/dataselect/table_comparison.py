"""
Table 1 of the paper (Section 4.4): baseline comparison including the matched-alpha check.

Merges budget_v2_main.csv (competitors at alpha = 0.2, our methods) and
budget_v2_main_a1.csv (competitors and the two RHO-LOSS combinations at alpha = 1, same
seeds). Per condition: error reduction vs. random selection in %
(100 * mean paired difference / mean random error), paired Wilcoxon over seeds,
Holm-corrected over all methods of the condition. Columns average the four region
conditions; All / Inside average all 24 conditions. Prints LaTeX rows and the paired
margins quoted in the text.

    python -m scripts.dataselect.table_comparison
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.dataselect.summarise_budget import holm

RES = Path("data/experiment_results/data_selective_training")
KEYS = ["dataset", "region", "noise", "seed"]
ROWS = [("uniform", "Uniform"),
        None,
        ("kcenter", "k-center"), ("kcenter_a1", "k-center ($\\alpha=1$)"),
        ("loss", "Loss"), ("jtt", "JTT"),
        ("rho", "RHO-LOSS"), ("rho_a1", "RHO-LOSS ($\\alpha=1$)"),
        None,
        ("ws_default", "Weakspot, default"), ("ws_tuned", "Weakspot, tuned"),
        ("rho_landscape_a1", "RHO-landscape ($\\alpha=1$)"),
        ("rho_filter_a1", "RHO-filter ($\\alpha=1$)"),
        None,
        ("all_data", "All data")]
DS = ["synth2d", "houses", "medical_charges"]


def load():
    m = pd.read_csv(RES / "budget_v2_main.csv")
    a = pd.read_csv(RES / "budget_v2_main_a1.csv")
    m, a = m[m.arm != "ERROR"], a[a.arm != "ERROR"]
    return pd.concat([m, a[a.arm != "uniform"]], ignore_index=True)


def main():
    d = load()
    arms = [r[0] for r in ROWS if r]
    rows = []
    for key, g in d.groupby(["dataset", "region", "noise"]):
        ref = g[g.arm == "random"].set_index("seed")
        block = []
        for arm in arms:
            x = g[g.arm == arm].set_index("seed").loc[ref.index]
            r = dict(zip(["dataset", "region", "noise"], key), arm=arm)
            for mcol in ("mae", "err_in"):
                diff = ref[mcol] - x[mcol]
                r[mcol] = 100 * diff.mean() / ref[mcol].mean()
                r[mcol + "_p"] = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            block.append(r)
        for r, q in zip(block, holm([r["mae_p"] for r in block])):
            r["ph"] = q
        rows += block
    s = pd.DataFrame(rows)
    s.to_csv(RES / "table_comparison.csv", index=False)

    lines = []
    for item in ROWS:
        if item is None:
            lines.append(r"\midrule")
            continue
        arm, lab = item
        g = s[s.arm == arm]
        cells = [g[(g.dataset == ds) & (g.noise == nz)].mae.mean()
                 for ds in DS for nz in ("clean", "outlier")]
        win = int(((g.ph < 0.05) & (g.mae > 0)).sum())
        los = int(((g.ph < 0.05) & (g.mae < 0)).sum())
        vals = cells + [g.mae.mean(), g.err_in.mean()]
        lines.append(f"{lab:<28}& " + " & ".join(f"${v:.1f}$" for v in vals)
                     + f" & {win}/{los} \\\\")
    print("\n".join(lines))

    # Paired margins over all seed-level pairs (whole-test % vs random per seed).
    ref = d[d.arm == "random"].set_index(KEYS).mae.rename("r")
    e = d[d.arm != "random"].join(ref, on=KEYS)
    e["adv"] = 100 * (e.r - e.mae) / e.r
    w = e.pivot_table(index=KEYS, columns="arm", values="adv").reset_index()

    def margin(a, b):
        x = w[a] - w[b]
        per = w.assign(x=x).groupby("dataset").x.mean().round(2).to_dict()
        return f"{a} - {b}: {x.mean():+.2f} (p={wilcoxon(x).pvalue:.1e}) {per}"

    print()
    for a, b in [("ws_tuned", "rho"), ("ws_tuned", "rho_a1"), ("ws_tuned", "kcenter_a1"),
                 ("rho_filter_a1", "rho_a1"), ("rho_filter_a1", "rho"),
                 ("rho_landscape_a1", "rho_a1"), ("rho_filter_a1", "ws_tuned"),
                 ("rho_filter_a1", "kcenter_a1"), ("kcenter_a1", "kcenter")]:
        print(margin(a, b))
    print("\nloss/jtt at alpha=1 (whole, all):",
          round(s[s.arm == "loss"].mae.mean(), 2), round(s[s.arm == "jtt"].mae.mean(), 2))


if __name__ == "__main__":
    main()
