"""
Experiment runner of Paper C: every strategy on every regime, paired per seed.

    python -m scripts.regional.run --stage smoke
    python -m scripts.regional.run --stage pilot --workers 12
    python -m scripts.regional.run --stage fixed --workers 3      (Setting B: fixed data)

One row per regime, dimension, budget, strategy and seed in
``data/experiment_results/regional_testing/<stage>.csv``. Finished jobs are skipped, so a
run can be stopped and restarted. The runner sets itself to idle priority.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.regional.benchmarks import BENCHMARKS
from scripts.regional.campaign import BASELINES, SCALAR, STRATEGIES, VARIANTS, run, score
from scripts.regional.landscapes import LANDSCAPES, test_set
from scripts.regional.partition import DEEP, HYBRIDS, METHODS, evaluate
from scripts.regional.problems import PLANTED, REAL, TRAINED, make
from scripts.regional.report import run_report

OUT = Path("data/experiment_results/regional_testing")
TRUTH = ("random", "binning", "weakspot", "soft", "peel", "subspace")
RARE = tuple(f"{r}_rare" for r in LANDSCAPES)
FIXED = dict(budgets=(2000, 10000, 50000), strategies=METHODS, kind="fixed")
LSE = ("gp_lse", "cross_entropy")
ABLATION = tuple(f"{s}{o}" for s in ("peel", "weakspot")
                 for o in ("", "|kappa=0.1", "|kappa=0.5", "|conf=0.15", "|conf=0.4", "|rounds=2", "|rounds=8"))
REPORT = ("fixed_cart", "random+cart", "sobol+cart", "gp_lse+cart", "cross_entropy+cart",
          "weakspot+cart", "subspace+cart", "peel+cart", "peel+weakspot_pca_deep")
STAGES = {
    "smoke": dict(regimes=("no_failure", "hard", "hard_dense"), dims=(2,), budgets=(500,),
                  seeds=range(9000, 9002)),
    "pilot": dict(regimes=PLANTED + TRAINED, dims=(2, 5), budgets=(1000,),
                  seeds=range(9100, 9150)),
    "variants_smoke": dict(regimes=("no_failure", "hard", "hard_dense"), dims=(5,), budgets=(1000,),
                           seeds=range(9000, 9002), strategies=VARIANTS),
    "variants": dict(regimes=PLANTED + TRAINED, dims=(2, 5, 10), budgets=(1000,),
                     seeds=range(9200, 9250), strategies=VARIANTS),
    # d = 0: the dimension is fixed by each benchmark
    "bench_smoke": dict(regimes=("otl_iid", "robot_shift"), dims=(0,), budgets=(1000,),
                        seeds=range(9000, 9002), strategies=VARIANTS),
    "truth_smoke": dict(regimes=("ackley_low", "herbie_rot"), dims=(5,), budgets=(300, 1000),
                        seeds=range(9000, 9002), strategies=TRUTH),
    "truth": dict(regimes=LANDSCAPES, dims=(2, 5, 10), budgets=(300, 1000, 3000),
                  seeds=range(9400, 9450), strategies=TRUTH),
    "benchmarks": dict(regimes=BENCHMARKS, dims=(0,), budgets=(1000,),
                       seeds=range(9300, 9350), strategies=VARIANTS),
    # Setting A baselines, paired with the truth stage (same regimes and seeds)
    "truth_base": dict(regimes=LANDSCAPES, dims=(2, 5, 10), budgets=(300, 1000, 3000),
                       seeds=range(9400, 9450), strategies=BASELINES),
    # Setting B: fixed data, one sample of size n per seed, no queries
    "fixed_smoke": dict(FIXED, regimes=("ackley_low", "rastrigin_rot_rare", "four_branch"),
                        dims=(5,), budgets=(2000, 10000), seeds=range(9000, 9002)),
    "fixed": dict(FIXED, regimes=LANDSCAPES + RARE + ("four_branch",), dims=(2, 5, 10),
                  seeds=range(9500, 9520)),
    "fixed_bench": dict(FIXED, regimes=BENCHMARKS, dims=(0,), seeds=range(9600, 9620)),
    # hybrids of CART and weakspot (added after the fixed stage), on fresh seeds
    "hybrid_smoke": dict(FIXED, regimes=("herbie_low", "rastrigin_rot_rare", "four_branch"), dims=(5,),
                         budgets=(2000, 10000), seeds=range(9000, 9002), strategies=HYBRIDS),
    "hybrid": dict(FIXED, regimes=LANDSCAPES + RARE + ("four_branch",), dims=(2, 5, 10),
                   budgets=(2000, 10000), seeds=range(9700, 9710), strategies=HYBRIDS),
    "hybrid_pca_smoke": dict(FIXED, regimes=("herbie_low", "rastrigin_rot", "four_branch"), dims=(5, 10),
                             budgets=(2000, 10000), seeds=range(9000, 9002), strategies=("weakspot_pca",)),
    # the supervised-PCA variant, paired with the hybrid stage (same regimes and seeds)
    "hybrid_pca": dict(FIXED, regimes=LANDSCAPES + RARE + ("four_branch",), dims=(2, 5, 10),
                       budgets=(2000, 10000), seeds=range(9700, 9710), strategies=("weakspot_pca",)),
    # refinement without the stop rule, paired with the hybrid and fixed_bench stages
    "deep_smoke": dict(FIXED, regimes=("herbie_low", "rastrigin_rot_rare", "four_branch"), dims=(5,),
                       budgets=(2000, 10000), seeds=range(9000, 9002), strategies=DEEP),
    "deep": dict(FIXED, regimes=LANDSCAPES + RARE + ("four_branch",), dims=(2, 5, 10),
                 budgets=(2000, 10000), seeds=range(9700, 9710), strategies=DEEP),
    "deep_bench": dict(FIXED, regimes=BENCHMARKS, dims=(0,), seeds=range(9600, 9620),
                       strategies=("weakspot_pca",) + DEEP),
    # the final refinement on the seeds of the fixed stage, for its row of the table
    "deep_fixed": dict(FIXED, regimes=LANDSCAPES + RARE + ("four_branch",), dims=(2, 5, 10),
                       seeds=range(9500, 9520), strategies=("weakspot_pca_deep",)),
    # Experiment 2 again: the baselines and scalar steering, paired with the benchmarks stage
    "bench_extra_smoke": dict(regimes=("otl_iid", "robot_shift"), dims=(0,), budgets=(1000,),
                              seeds=range(9000, 9002), strategies=BASELINES + SCALAR),
    "bench_extra": dict(regimes=BENCHMARKS, dims=(0,), budgets=(1000,), seeds=range(9300, 9350),
                        strategies=BASELINES + SCALAR),
    # scalar steering on the landscapes, paired with the truth stage
    "truth_scalar_smoke": dict(regimes=("ackley_low", "herbie_rot"), dims=(5,), budgets=(1000,),
                               seeds=range(9000, 9002), strategies=SCALAR),
    "truth_scalar": dict(regimes=LANDSCAPES, dims=(5, 10), budgets=(1000,), seeds=range(9400, 9450),
                         strategies=SCALAR),
    # stronger baselines: GP level set estimation and cross-entropy search, paired with the
    # truth and benchmarks stages
    "truth_lse": dict(regimes=LANDSCAPES, dims=(5, 10), budgets=(1000,), seeds=range(9400, 9450),
                      strategies=LSE),
    "bench_lse": dict(regimes=BENCHMARKS, dims=(0,), budgets=(1000,), seeds=range(9300, 9350),
                      strategies=LSE),
    # real data: a model trained on logged process data and on census data
    "real_smoke": dict(regimes=REAL, dims=(0,), budgets=(1000,), seeds=range(9000, 9002),
                       strategies=("random", "peel", "gp_lse")),
    "real": dict(regimes=REAL, dims=(0,), budgets=(1000,), seeds=range(9300, 9350),
                 strategies=VARIANTS + BASELINES + LSE),
    "real_fixed": dict(FIXED, regimes=REAL, dims=(0,), budgets=(2000, 5000), seeds=range(9600, 9620),
                       strategies=METHODS + ("weakspot_pca_deep",)),
    # the settings of the campaign, one at a time
    "ablation": dict(regimes=LANDSCAPES + BENCHMARKS, dims=(5,), budgets=(1000,),
                     seeds=range(9900, 9920), strategies=ABLATION),
    # the full report in the query setting (Definition 1), on fresh seeds
    "report_smoke": dict(regimes=("herbie_low", "otl_iid", "sulfur"), dims=(5,), budgets=(1000,),
                         seeds=range(9000, 9001), strategies=REPORT, kind="report"),
    "report": dict(regimes=LANDSCAPES, dims=(5, 10), budgets=(1000, 3000), seeds=range(9800, 9820),
                   strategies=REPORT, kind="report"),
    "report_bench": dict(regimes=BENCHMARKS + REAL, dims=(0,), budgets=(1000, 3000),
                         seeds=range(9800, 9820), strategies=REPORT, kind="report"),
}


def spec(s):
    """A strategy with optional overrides of the campaign settings, as in "peel|kappa=0.1"."""
    name, *kv = s.split("|")
    return name, {k: (int(v) if k == "rounds" else float(v)) for k, v in (p.split("=") for p in kv)}


def job(regime, d, n, seed, strategies=STRATEGIES, kind="campaign"):
    problem = make(regime, d, seed, n)
    rows = []
    if kind == "report":
        for s in strategies:
            try:
                if s == "fixed_cart":                   # no queries: random data, split, tree
                    out = evaluate(problem, "cart", n, seed)
                else:
                    strategy, partition = s.split("+")
                    out = run_report(problem, n, strategy, partition, seed)
                rows.append(dict(regime=regime, d=d, n=n, seed=seed, strategy=s, **out))
            except Exception as ex:                      # noqa: BLE001
                rows.append(dict(regime=regime, d=d, n=n, seed=seed, strategy=s, tau="ERROR:" + repr(ex)))
        return rows
    if kind == "fixed":
        for s in strategies:
            try:
                rows.append(dict(regime=regime, d=d, n=n, seed=seed, strategy=s,
                                 **evaluate(problem, s, n, seed)))
            except Exception as ex:                      # noqa: BLE001
                rows.append(dict(regime=regime, d=d, n=n, seed=seed, strategy=s, events="ERROR:" + repr(ex)))
        return rows
    for s in strategies:
        try:
            name, kw = spec(s)
            report, region, X, e = run(problem, n, name, seed, tests=True, **kw)
            extra = test_set(problem, X, e, 0.1, seed) if regime in LANDSCAPES else {}
            rows.append(dict(regime=regime, d=d, n=n, seed=seed, strategy=s, **report,
                             **score(problem, region, report, 0.1, seed), **extra))
        except Exception as ex:                          # noqa: BLE001
            rows.append(dict(regime=regime, d=d, n=n, seed=seed, strategy=s, label="ERROR:" + repr(ex)))
    return rows


def idle():
    """Run at idle priority, so that a long stage does not slow the machine; the workers
    started by joblib inherit the priority class on Windows."""
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k.SetPriorityClass(k.GetCurrentProcess(), 0x40)      # IDLE_PRIORITY_CLASS
    else:
        os.nice(19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    idle()
    from joblib import Parallel, delayed
    cfg = STAGES[args.stage]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.stage}.csv"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["regime"], int(r["d"]), int(r["n"]), int(r["seed"])) for r in csv.DictReader(f)}
    jobs = [(r, d, n, s) for r in cfg["regimes"] for d in cfg["dims"] for n in cfg["budgets"]
            for s in cfg["seeds"] if (r, d, n, s) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    names = None
    strategies = cfg.get("strategies", STRATEGIES)
    for rows in Parallel(n_jobs=args.workers, return_as="generator_unordered")(
            delayed(job)(*j, strategies, cfg.get("kind", "campaign")) for j in jobs):
        names = names or list(dict.fromkeys(k for r in rows for k in r))   # union, in order
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=names, restval="")
            if new:
                w.writeheader()
            w.writerows(rows)
    print("done", path, flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
