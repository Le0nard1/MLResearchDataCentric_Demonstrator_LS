# Experimental details — *Selecting Data Where the Model Is Weak*

Supplementary material for the paper *Selecting Data Where the Model Is Weak:
Statistically Guided Curation for Data-Centric Training*. The paper refers to this
file for the details that are not needed to follow its argument.

## A. Experimental details

### Target and data

The target on $[0,1]^2$ is a sum of Gaussian bumps,

$$f(\mathbf{x}) = \sum_k h_k \exp\!\left(-\frac{\lVert\mathbf{x}-\mathbf{c}_k\rVert^2}{2 w_k^2}\right),$$

with, for five bumps:

| Bump | Centre $\mathbf{c}_k$ | Height $h_k$ | Width $w_k$ |
|---|---|---|---|
| 1 | (0.17, 0.17) | 3.0 | 0.10 |
| 2 | (0.25, 0.75) | 2.5 | 0.12 |
| 3 | (0.50, 0.50) | 4.0 | 0.13 |
| 4 | (0.75, 0.17) | 2.0 | 0.08 |
| 5 | (0.83, 0.83) | 3.5 | 0.10 |

Three bumps take the first three. (Source: `scripts/weakspot/datasets.py`, bump pool
rescaled from $[0,6]^2$ to the unit square in `scripts/dataselect/pipeline.py`.)

- A dataset of 2000 uniform points receives Gaussian label noise. The induced gap
  (a disc centred at $(0.5, 0.5)$, i.e. on the tallest bump) is removed from it, and
  the initial training set is drawn from the remainder.
- The candidate pool is a further 2000 uniform points with the same label noise.
- The evaluation sample is 1000 uniform, **noiseless** points. The detectors also
  operate on the initial model's errors on this sample.
- The in-weakspot error is the mean error over evaluation points inside the true gap.
- The focused experiments (Sections 4.3 and 4.4) use five bumps and noise level 0.05.

### Model and training

- scikit-learn `MLPRegressor`, two hidden layers of 64 ReLU units, standardised
  inputs (`scripts/weakspot/models.py`, `complexity = 0.5`).
- Adam, learning rate $10^{-3}$, L2 penalty $10^{-4}$, batch size $\min(200, N)$.
- Early stopping on a 10 % validation split, patience 10.
- One iteration is one epoch (`max_iter`).
- Retraining continues the initial weights (`warm_start`) for the stated number of
  epochs on the selected points only. Guided and baseline models start from the same
  initial model; the baseline is one uniform draw of equal size from the same pool.

### Weakspot identification

- Each detector maps the initial model's error to a surface on a $35\times35$ grid,
  normalised to $[0,1]$.
- The weakspot is the largest connected component above the surface's 0.85 quantile;
  its weighted centroid is the kernel centre $\hat{\mathbf{c}}$ and its $2\sigma$
  ellipse the detected extent (`scripts/weakspot/extraction.py`).
- IoU against the true gap is taken at the 0.90 quantile.

### Selection

A fraction $\alpha$ (guidance fraction) of the budget $n$ is drawn without replacement
with probability proportional to the Gaussian kernel around $\hat{\mathbf{c}}$; the
remaining $(1-\alpha)\,n$ points are drawn uniformly from the rest of the pool
(`select_by_weakspot` in `scripts/dataselect/pipeline.py`).

### Configuration of Sections 4.3 and 4.4

The fixed factors (initial model trained for 12 epochs on 100 points, 100 points added,
retraining for 400 epochs, gap radius 0.25) come from an exploratory search that
followed the broad-sweep trends towards smaller, less-trained initial models (configs
`best_search`, `best_search2`; five seeds, 42/0/7/1/3, which are also among the 50
seeds of Sections 4.3 and 4.4). They were not re-tuned for the experiments of Sections
4.3 and 4.4. Robustness of the Section 4.3 band ($\alpha \in [0.05, 0.35]$,
$\sigma = 0.1$, 50 seeds, per-seed mean difference guided − random): kNN performance
mapping −0.079 ± 0.070 ($p = 0.04$), peaks over threshold −0.066 ± 0.067
($p = 0.048$), RBF interpolation −0.051 ± 0.070 ($p = 0.16$) (configs `iso_mix_knn`,
`iso_mix`).

## B. Detector comparison

All seventeen detectors of the companion weakspot-identification paper, at the
configuration of Section 4.3 with a wide kernel and a moderate guidance fraction
($\sigma = 0.5$, $\alpha = 0.5$, gap radius 0.25, 20 seeds; config
`best_vary_detector`). $\Delta\mathrm{MAE}$ = random − guided whole-area MAE
(positive: guided better).

| Detector | ΔMAE | Distance | IoU |
|---|---:|---:|---:|
| Quantile regression (QR)† | 0.157 | 0.021 | 0.19 |
| GPR/kNN (variance-weighted) | 0.143 | 0.070 | 0.31 |
| Bayesian optimisation (EI)† | 0.129 | 0.003 | 0.01 |
| LOESS | 0.118 | 0.053 | 0.47 |
| Peaks over threshold (EVT) | 0.117 | 0.065 | 0.35 |
| RBF interpolation | 0.116 | 0.070 | 0.30 |
| kNN performance mapping | 0.115 | 0.068 | 0.32 |
| GPR/kNN (disagreement-amplified) | 0.104 | 0.069 | 0.29 |
| kNN + quantile (mean) | 0.081 | 0.072 | 0.32 |
| EVT × GPR (geometric) | 0.075 | 0.065 | 0.35 |
| QR × GPR (geometric)† | 0.071 | 0.006 | 0.21 |
| Gaussian process regression | 0.063 | 0.071 | 0.30 |
| Polynomial response surface | 0.048 | 0.111 | 0.45 |
| kNN + GPR (mean) | 0.040 | 0.070 | 0.31 |
| Anchored GPR (QR prior) | 0.019 | 0.071 | 0.30 |
| Local GPR (QR-localised) | 0.018 | 0.478 | 0.00 |
| Gated GPR (QR anchor) | 0.016 | 0.484 | 0.00 |

† Detected extent covers more than 85 % of the square, so its centroid falls on the
gap centre by symmetry and the distance is not informative.

## C. Reproducing the paper

Sweep configs live in `scripts/dataselect/sweep_configs/`; each writes
`data/experiment_results/data_selective_training/sweep__<config>.csv`.

| Paper | Config / script | Command |
|---|---|---|
| 4.2 Broad sweep, Table 1, Figure 2 | `alpha_boundary` (five detectors) | `python run_sweep.py --config alpha_boundary --workers N` |
| 4.3 Guidance fraction, Figure 3 left | `iso_mix_knn` | `python run_sweep.py --config iso_mix_knn --workers N` |
| 4.3 Kernel width, Figure 3 right | `iso_gaussian_knn` | `python run_sweep.py --config iso_gaussian_knn --workers N` |
| 4.3 Independent-diagnosis check | `iso_mix_knn_heldout` (`diag_sample = heldout`: detectors read the initial model's error on a second, independent noiseless sample from its own RNG; pool, baseline and evaluation sample identical to `iso_mix_knn`) | `python run_sweep.py --config iso_mix_knn_heldout --workers N` |
| 4.4 Spatial analysis, Figure 4 | error-landscape study | `python -m scripts.generate_error_landscape_figure --mix 0.2 --sigma 0.1` |
| 4.4 per-seed statistics | `iso_mix_knn` at $\alpha = 0.2$ | (same CSV as 4.3) |
| Appendix B above | `best_vary_detector` | `python run_sweep.py --config best_vary_detector --workers N` |

Figures 2 and 3: `plot_global_sensitivity()` and `plot_isolation()` in
`scripts/dataselect/plot_focused.py`.

Per-seed statistics in the paper average the guided–baseline difference over the
runs of each seed first (all levels of a sweep share one baseline draw per seed) and
report a $t$-based 95 % confidence interval over the 50 seeds with a Wilcoxon
signed-rank $p$-value.

## Section 4.5 — fixed-data reweighting (restricted data)

Script `scripts/dataselect/fixed_data.py` (stages in order):

| Stage | Command | Output |
|---|---|---|
| Setup pilot (diagnosis validity only, seeds 3000–3019) | `python -m scripts.dataselect.fixed_data --stage setup` | `fixed_data_setup.csv` |
| Hyperparameter pilot (same seeds, reference condition per task) | `--stage pilot` | `fixed_data_pilot.csv`, frozen `fixed_data_hp.json` |
| Main run (fresh seeds 4000–4049, all conditions) | `--stage main` | `fixed_data.csv`, `fixed_data_maps.npz` |
| Frontier (every grid setting, reported seeds, descriptive) | `--stage frontier` | `fixed_data_frontier.csv` |
| Effect-size gate ratios | `--stage ratio` | `fixed_data_ratio.csv` |

Table 2 and Figures 3–4 (+ App. B California map): `python -m scripts.dataselect.plot_fixed_data`
→ `fixed_data_summary.csv`, `figures/fig_fixed_data_{frontier,maps,california}.png`.

Gate on the Table 1 well-trained regimes: `python -m scripts.dataselect.gate` → `gate_v2.csv`,
`gate_v2_summary.csv` (replicates `baselines.run_one`'s initial model exactly; checked).

California housing is read from `~/scikit_learn_data/cal_housing_raw.npy` (the raw
`cal_housing.data` array; sklearn's own downloader crashed on this machine). Its detection
surface is masked to grid cells within 0.05 of the data (`support_dist`) — without the mask
the kNN map extrapolated coastal errors into the ocean and the detected centres fell offshore.

### Moved out of the paper's main text (condensed 2026-09-24)

**Operating point (Sections 4.2–4.4, selection protocol).** Fixed at the favourable end of
the broad-sweep trends by an exploratory search slightly beyond the grid, not re-tuned:
initial model 12 iterations on 100 points, budget 100 points, retraining 400 iterations,
gap radius 0.25 at the tallest bump (0.5, 0.5); detector kNN performance mapping (its
broad-sweep advantage lies closest to the mean of the five detectors). The initial model
is barely fitted (MAE 0.95 vs 0.77 for the constant mean), and the detector finds the same
centre with or without the gap (distance 0.068).

**Well-trained regime (Section 4.4).** 200 iterations on 400 points (MAE 0.33), gap of
radius 0.2 on the lower bump at (0.25, 0.75), which the model fits well without the gap.
Chosen on pilot seeds 2000–2019 from diagnosis validity only (`pilot_setup.py`); kernel
width for old+new retraining σ = 0.5 from `pilot_cumulative.py` on the same seeds.
Weakspot σ = 0.1 otherwise. Random-baseline MAE (initial): undertrained gap 0.557 (0.94),
no gap 0.560 (0.94); well-trained new only 0.384 (0.33), old+new 0.264 (0.33).

**Full fixed-data table (Section 4.5, paper Table 2 shows a subset).** Error reduction vs
uniform continuation in %, whole area (inside the weakspot); Wilcoxon, Holm-corrected per
row. Gate = share of seeds passing the permutation test.

| Condition | Weakspot | Gated | Loss | JTT | Density | Uniform MAE | Gate |
|---|---|---|---|---|---|---|---|
| California | +0.3 (+1.3**) | +0.0 (+0.1) | -2.3** (-2.4**) | -3.7** (-4.2**) | -0.1 (+0.1) | 0.675 | 0.08 |
| hard, gauss | -2.0** (+4.3**) | -1.9** (+4.1**) | -2.4** (+3.6**) | -0.5 (+4.8**) | +0.9* (+0.3) | 0.113 | 0.96 |
| hard, hetero | -0.7 (+3.2**) | -0.8 (+2.6**) | -3.2** (+1.9**) | -3.0** (+3.4**) | +0.8** (-0.3) | 0.130 | 0.78 |
| hard, outlier | +0.1 (+1.4**) | -0.1 (+0.1) | -33.7** (-7.7**) | -13.0** (-1.4**) | +0.6 (+0.2) | 0.195 | 0.06 |
| hole, gauss | +0.6 (+1.3) | +0.6 (+1.4) | +1.7** (+1.5) | +1.9** (+2.2) | +0.4 (+0.6) | 0.102 | 0.96 |
| hole, hetero | -1.4 (+1.2) | +0.9 (+2.0*) | -1.9* (+3.1*) | -0.5 (+4.6**) | +0.8 (+0.6) | 0.123 | 0.36 |
| hole, outlier | +0.7 (+1.7) | +0.0 (+0.0) | -39.4** (-6.8) | -12.6** (-0.4) | +0.6 (+0.7) | 0.179 | 0.00 |
| none, gauss | -0.4 | -0.5 | +1.8** | +1.3* | +1.2* | 0.060 | 0.88 |
| none, hetero | -4.3** | -0.9* | -5.8** | -3.4** | +0.8** | 0.076 | 0.20 |
| none, outlier | +0.2 | +0.0 | -53.7** | -17.4** | +1.3* | 0.149 | 0.06 |
