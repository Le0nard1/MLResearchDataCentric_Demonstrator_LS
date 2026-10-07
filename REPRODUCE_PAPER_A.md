# Reproducing Paper A: single-round weakspot curation

Code, configurations and per-seed results in this repository for Paper A (under review),
which selects training data at the statistically detected weakspot in a single round.
Every number and figure of the paper can be recomputed from this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands are run from the repository root. The synthetic task is generated on the fly.
The two benchmark datasets, California housing (OpenML 44138) and medical charges
(OpenML 44146), are the preprocessed versions of the numerical regression benchmark of
Grinsztajn et al. (2022); they are downloaded on first use and cached under
`~/scikit_learn_data/grinsztajn/`. All experiments are seeded and deterministic; runtimes
below were measured with 19 worker processes on a 14-core machine.

## Paper → code

| Paper | Command | Output | Runtime |
|---|---|---|---|
| §4.2 broad sweep, sensitivity figure (App. C) | `python run_sweep.py --config alpha_boundary --workers <n>` then `python -m scripts.dataselect.plot_focused` | `sweep__alpha_boundary.csv` (≈1.7 GB, not included), `fig_global_sensitivity.pdf` (and `.png`) | ≈1.6 M runs, cluster-scale |
| §4.3 tuning | `python -m scripts.dataselect.budget_v2 --stage tune` then `python -m scripts.dataselect.summarise_tune tune` | `budget_v2_tune.csv`, tuned setting in `budget_v2_tuned.json` | 35 min |
| §4.3 detector comparison (17 detectors) | `python -m scripts.dataselect.detector_pilot` | `detector_pilot.csv` | 5 min |
| §4.4 comparison, Table 1 | `python -m scripts.dataselect.budget_v2 --stage main`, `python -m scripts.dataselect.budget_v2 --stage main_a1`, then `python -m scripts.dataselect.table_comparison` | `budget_v2_main.csv`, `budget_v2_main_a1.csv`, `table_comparison.csv` | 51 + 19 min |
| §4.4 further benchmark datasets (Table 1, right-hand columns) | `python -m scripts.dataselect.budget_v2 --stage bench` then `python -m scripts.dataselect.summarise_bench` | `budget_v2_bench.csv`, `bench_summary.csv` | 6 min |
| §4.4 CRAIG and GLISTER (Table 1) | `python -m scripts.dataselect.pilot_craig_glister --seeds 50` and `--bench` | `pilot_craig_glister.csv`, `pilot_craig_glister_bench.csv` | 38 + 9 min |
| §4.4 weakspot-weighted k-center, split variant and controls (Table 1) | `python -m scripts.dataselect.pilot_hybrid --seeds 50 --out hybrid_main.csv` and `--bench --out hybrid_bench.csv` | `hybrid_main.csv`, `hybrid_bench.csv` | 19 + 5 min |
| App. C, confidence intervals of Table 1 | `python -m scripts.dataselect.table_comparison_ci` | printed LaTeX rows | 1 min |
| §4.4 severity / headroom analysis | `python -m scripts.dataselect.severity_analysis` | `budget_v2_severity.csv` | seconds |
| §4.5 reweighting, fixed-data and location-control tables and figures | `python -m scripts.dataselect.fixed_data --stage setup`, `--stage pilot`, `--stage main`, `--stage frontier`, then `python -m scripts.dataselect.plot_fixed_data` | `fixed_data*.csv`, `fixed_data_maps.npz`, `fig_fixed_data_*.pdf` (and `.png`) | not re-timed |
| App. B, beyond two dimensions | `python -m scripts.dataselect.fixed_data_hd --stage frontier`, then `python -m scripts.dataselect.plot_fixed_data_hd` | `fixed_data_hd_frontier.csv`, `fig_fixed_data_hd.pdf` (and `.png`) | not re-timed |

Outputs are written to `data/experiment_results/data_selective_training/`, figures to its
`figures/` subfolder. Scripts that append results skip configurations already present, so an
interrupted run resumes; delete the corresponding CSV for a fresh run.

## Included results

The broad sweep's raw results (`sweep__alpha_boundary.csv`, about 1.7 GB) exceed GitHub's file limit and are not included; every other result the paper reports is.

`data/experiment_results/data_selective_training/` contains the per-seed results behind the
paper, so the tables can be checked without re-running the experiments:

- `budget_v2_tune.csv` — tuning of weakspot selection (seeds 6100–6109)
- `budget_v2_main.csv`, `budget_v2_main_a1.csv` — comparison, competitors at α = 0.2 and α = 1 (seeds 7000–7049)
- `budget_v2_bench.csv` — further datasets: diamonds, sulfur, Brazilian houses, nyc-taxi (seeds 7000–7009)
- `detector_pilot.csv` — selection with each of the 17 detectors (seeds 6200–6204)
- `fixed_data*.csv`, `fixed_data_maps.npz`, `fixed_data_hp.json` — reweighting (pilot seeds 3000–3019, reported seeds 4000–4049)
- `fixed_data_hd*.csv`, `fixed_data_hd_hp.json` — reweighting beyond two dimensions
- `pilot_craig_glister.csv`, `pilot_craig_glister_bench.csv` — CRAIG and GLISTER in the comparison, paired with the files above (seeds 7000–7049 and 7000–7009)
- `hybrid_main.csv`, `hybrid_bench.csv` — weakspot-weighted k-center, the split variant and their controls, paired with the files above (seeds 7000–7049 and 7000–7009)
- `budget_v2_tuned.json` — the tuned setting used in the comparison

`python -m scripts.dataselect.table_comparison` rebuilds Table 1 from the included files.

## Layout

```
run_sweep.py                        broad-sweep runner (Section 4.2)
scripts/weakspot/                   weakspot identification: models, detectors, extraction
scripts/dataselect/pipeline.py      synthetic target, sampling, selection kernels
scripts/dataselect/sweep.py         broad sweep; sweep_configs/alpha_boundary.json is its grid
scripts/dataselect/budget.py        first budget design and shared helpers (data loading, RHO-LOSS)
scripts/dataselect/budget_v2.py     training-budget protocol: tuning, comparison, alpha = 1 check
scripts/dataselect/baselines.py     selection baselines (k-center, rehearsal)
scripts/dataselect/fixed_data*.py   reweighting of a fixed dataset (Section 4.5, Appendix B)
scripts/dataselect/summarise_*.py   summaries and tables
```
