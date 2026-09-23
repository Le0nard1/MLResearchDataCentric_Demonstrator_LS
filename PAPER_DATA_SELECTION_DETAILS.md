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
