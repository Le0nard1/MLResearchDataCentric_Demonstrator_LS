# Real-data example — iris_extended (2026-08-20)

App: page `11_Real_Data_Example.py` + engine `scripts/realdata/loop_real.py`
(kept fully separate from the synthetic pages 05–10). Data:
`data/raw/iris_extended.csv` (1200 rows, 18 numeric features, 2 categoricals
species/soil_type, target leaf_area_cm2). Runner:
`iterative_data_selective_training/analysis_scripts/run_real_experiment.py` →
`real_runs.csv` (one row per setting × encoding × policy × seed × iteration,
15 seeds, K=8). Figure: `fig_realdata.png` (also in the paper repo).

## Design

- Loop mirrors the paper: warm start, fixed 200-epoch retrain on new points
  only, guided vs random from the same pool, **pool = the dataset, consumed
  without replacement** (n_eval 300, n_train 50, n_select 40/round).
- Detection without a grid: eval-point error smoothed over k=15 NN in
  standardised feature space; centre = peak; severity = kernel-weighted
  in/global error ratio (ground-truth-free, drives the adaptive policy).
- Categorical treatments: `numeric` (drop cats from model + distance),
  `onehot` (one-hot into model AND distance → cross-category "closest"),
  `stratified` (one-hot model, numeric distance, guided share drawn to match
  the weakspot members' categorical distribution via per-category quotas).
- Settings: `raw` (dataset as-is) and `deficit` (virginica fully withheld
  from initial train, 90% thinned from pool — the scarce-pool analogue).

## Results (across-loop gap, guided − random; 15 seeds)

Raw: everything ≥ 0 (wash to loss; stratified-adaptive worst +0.32 p=0.001).
Severity stays 1.1–1.5 → **no pronounced weakspot exists**, and the paper's
rule predicts exactly this.

Deficit: weakspot immediately concentrates on virginica (58/36/7% across its
soil types at round 1). Then:

| encoding   | static | dynamic | adaptive |
|------------|--------|---------|----------|
| numeric    | −0.03  | +0.03   | +0.06    |
| **onehot** | **−0.79 (p=.006, 87%)** | **−0.46 (p=.039)** | **−0.63 (p=.017, 87%)** |
| stratified | +1.64  | +0.03   | +1.29    |

- **onehot wins across all policies**: the categorical dimensions in the
  distance let the kernel harvest the rare virginica rows random almost never
  draws.
- **numeric ≈ 0**: a distance that cannot express category cannot aim at a
  categorical deficit (round-1 pick/weakspot category overlap 0.10).
- **stratified fails hard with huge variance**: quotas force the guided
  budget onto the few remaining virginica rows; once exhausted, the quota
  keeps concentrating on a depleted region — the real-data analogue of the
  over-thinned (0.995) pool, produced by policy instead of data.

Adaptive-vs-static ordering is not significant here (CIs overlap); the
severity gate keeps α≈0.5 throughout because severity grows rather than
collapses (sev₀≈1.13 → up to 1.8), which is the correct gate behaviour under
a persistent deficit.

Paper: section "A Real-Data Example" (table + 2-panel figure) inserted before
Outlook.
