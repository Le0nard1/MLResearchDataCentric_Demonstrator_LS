# Reproducing Paper B: iterated weakspot curation

Code and instructions for Paper B, the iterated study that follows Paper A
([`REPRODUCE_PAPER_A.md`](REPRODUCE_PAPER_A.md)). Paper A selects training data at the
detected weakspot once; Paper B repeats the selection over eight rounds of retraining at
equal compute per round, follows how the detected weakspot drifts, and compares ways to
adapt the focus and to combine it with coverage-based selection. Every number, table and
figure of the paper can be recomputed from this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands are run from the repository root. The synthetic task is generated on the
fly. The six benchmark datasets (Brazilian houses, diamonds, California housing, medical
charges, NYC taxi trips, sulfur) are the preprocessed versions of the numerical
regression benchmark of Grinsztajn et al. (2022); they are downloaded from OpenML on
first use and cached under `~/scikit_learn_data/grinsztajn/`. The engine is
deterministic: a run is fixed by dataset, condition and seed, and every strategy of a
run shares its data, its initial model and its random-selection control.

## The engine

`scripts/iterative/iter_budget.py` runs the loop; the module docstring describes it.
Each paper stage runs 7 datasets x 6 conditions x 20 seeds (8200-8219) = 840 jobs and
appends one row per strategy, run and round to
`data/experiment_results/iterative_weakspot_curation/iter_<stage>.csv`. Finished jobs
are skipped, so an interrupted stage resumes where it stopped.

| Stage | Strategies | Runtime (4 workers) |
|---|---|---|
| `full2` | random control; static, dynamic, severity- and size-adaptive focus; random selection on new points only | several hours |
| `full2_kc` | iterated k-center | 9 min |
| `full2_abl` | ablation of the size-adaptive schedule; focus + coverage; gain inside the detected region | not re-timed |
| `full2_ctl` | k-center half + uniform half; focus + coverage with the fixed region | 40 min |
| `full2_ctl2` | matched control (same share uniform, rest k-center); focus + coverage at alpha 0.25 and 0.75 | 52 min |
| `full2_err` | top error; top error + coverage | 36 min |
| `full2_wkc` | weakspot-weighted k-center (Paper A's best strategy, applied in every round) | 25 min |

```bash
python -m scripts.iterative.iter_budget --stage full2_smoke          # one job, ~1 min
python -m scripts.iterative.iter_budget --stage full2 --workers 8
python -m scripts.iterative.iter_budget --stage full2_kc --workers 8  # and so on
```

`--datasets` and `--n-seeds` restrict a stage. Runtimes were measured on a 14-core laptop
CPU; the engine sets its worker processes to idle priority.

## Paper -> code

| Paper | Command | Source |
|---|---|---|
| Fig. 1 (Experiment 1, static focus per round) | `python -m scripts.iterative.make_full2_figures --out figures --skip-maps` | `iter_full2.csv` |
| Fig. 2 (weakspot dynamics) | same command | `iter_full2.csv` (`err_in`, `n_in_region`, `size`) |
| Fig. 3 (per-round map, main text) | `python -m scripts.iterative.make_full2_figures --out figures --maps-small-only` | re-runs seed 8203 with a trace |
| Fig. 4 (focus + coverage per round) | same command as Fig. 1 | `iter_full2*.csv` |
| Fig. 5 (per-round map, appendix) | `python -m scripts.iterative.make_full2_figures --out figures` | re-runs seed 8203 with a trace |
| Tables 1-3, paired comparisons, 95% CIs | `python -m scripts.iterative.make_full2_tables` | all `iter_full2*.csv` |
| Table 4 (absolute errors), retraining regime (Sec. 4.5) | same command | `iter_full2.csv`, `iter_full2_kc.csv` |
| Gain inside the detected region (Sec. 4.3) | column `det_gain` | `iter_full2_abl.csv` |
| Region points collected (Sec. 5, counting account) | column `n_in_region`, summed over rounds | `iter_full2.csv` |
| k-center and sparse areas (Sec. 5) | `python -m scripts.iterative.analysis_kcenter_sparsity` | replays k-center, model-free |

The table script prints LaTeX rows, the pooled mean of every strategy with a 95%
bootstrap interval, and paired comparisons (Wilcoxon, Holm-corrected over datasets).

## Reusing the methods

The selection rules are independent of the experiment loop:

| Function | What it does |
|---|---|
| `scripts.dataselect.budget.landscape(U, e, k)` | error landscape: leave-one-out, distance-weighted kNN mean of the errors |
| `scripts.iterative.selection.select(method, n, n_focus, U, e, land, used, rs, ...)` | one round of any strategy of the paper (focus, focus + coverage, k-center, weighted k-center, top error, controls) |
| `scripts.iterative.selection.held_out_measures(U, e, rs, ...)` | severity and size of the detected weakspot, cross-fitted, no ground truth needed |
| `scripts.iterative.selection.elevated_area(land, q, kappa)` | the clearly elevated part of the landscape |
| `scripts.iterative.selection.weighted_kcenter(U, used, land, n)` | k-center with the distance scaled by the landscape |
| `scripts.iterative.iter_budget.alpha_schedule(...)` | static, dynamic, severity- and size-adaptive schedules of the focused share |

`python -m scripts.iterative.example_reuse` runs the recommended strategy (size-adaptive
focus from the elevated area, rest by k-center, replay) on a toy task in about a minute;
replace its `make_data` by your own reserve and initial set.

## Results

The result CSVs (about 80 MB) are not committed to this repository; the commands above
regenerate them. The development runs that fixed the settings before the main runs use
separate seeds: stages `pilot`, `pilot2`, `pilot3`, `pilot4` (seeds 9000-9209) and `full`
(seeds 8100-8119).
