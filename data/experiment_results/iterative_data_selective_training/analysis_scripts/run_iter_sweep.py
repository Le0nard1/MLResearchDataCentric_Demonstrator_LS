"""Headless driver for the iterative parameter sweeps.

Runs one or more sweep configs (scripts/iterative/sweep_configs/<name>.json)
through the same resumable code path as the app's Parameter Sweep tab:
workers only compute, the main process does every CSV append, and completed
param_keys are skipped so the run can be interrupted and resumed freely.

Usage:  python run_iter_sweep.py <config> [<config> ...] [--workers N]
"""
import argparse
import os
import sys
import time

APP = r"C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application"
sys.path.insert(0, APP)
os.chdir(APP)  # sweep.RESULTS_DIR is relative to the CWD

from joblib import Parallel, delayed, parallel_backend  # noqa: E402


def run_config(name: str, workers: int) -> None:
    from scripts.iterative import sweep as SW
    SW.set_active_config(name)
    remaining = SW.remaining_combos({})
    total = len(SW.parameter_grid())
    print(f"\n=== {name}: {total} configs, {len(remaining)} remaining -> "
          f"{SW.CSV_PATH} ===", flush=True)
    if not remaining:
        return
    t0, done, n = time.time(), 0, len(remaining)
    with parallel_backend("loky", inner_max_num_threads=1):
        with Parallel(n_jobs=workers) as par:
            i = 0
            while i < n:
                chunk = remaining[i:i + workers * 2]
                for rows in par(delayed(SW.safe_run_one)(p) for p in chunk):
                    SW.append_rows(SW.CSV_PATH, rows)
                i += len(chunk)
                done += len(chunk)
                el = time.time() - t0
                eta = el / done * (n - done) / 60
                print(f"  {name}: {done}/{n}  elapsed {el/60:.1f} min  "
                      f"ETA {eta:.1f} min", flush=True)
    print(f"=== {name} done in {(time.time() - t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for cfg in args.configs:
        run_config(cfg, args.workers)
    print("ALL SWEEPS COMPLETE", flush=True)
