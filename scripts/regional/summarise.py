"""
Tables of a regional testing stage, per regime, dimension and strategy.

    python -m scripts.regional.summarise --stage pilot
"""
import argparse

import pandas as pd

from scripts.regional.campaign import STRATEGIES, VARIANTS
from scripts.regional.run import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    args = ap.parse_args()
    d = pd.read_csv(OUT / f"{args.stage}.csv")
    d["flagged"] = d.label == "flagged"
    names = [s for s in dict.fromkeys(STRATEGIES + VARIANTS) if s in set(d.strategy)]
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    # F1 of the region against the true failing set; no overlap counts as zero
    d["f1"] = (2 * d.precision * d.recall / (d.precision + d.recall).where(lambda s: s > 0)).fillna(0)
    for m in ("flagged", "f1", "precision", "recall", "true_mass", "covered"):
        print(f"\n{m}\n", d.groupby(["regime", "d", "strategy"])[m].mean().unstack()[names].round(2))
    print("\nvalidity\n", d.groupby("strategy")[["covered", "global_covered", "false_flag", "false_clear"]]
          .mean().loc[names].round(3))
    print("\nmedians\n", d.groupby("strategy")[["global_upper", "draws"]].median().loc[names].round(3))


if __name__ == "__main__":
    main()
