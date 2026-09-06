# Weakspot Research Environment

An experimental environment for studying **weakspots** in regression models — compact
regions of the input space where a trained model performs systematically worse than
elsewhere — and for using those weakspots to drive **data-centric training**.

Everything here shares one premise: model failure is not spread evenly across the input
space. It concentrates where the training data was thin. If those regions can be
*located*, they can also be *targeted*, by curating the next batch of training data
where the model is actually weak rather than uniformly at random.

The repository supports three lines of work, each backed by a paper:

| # | Approach | Question | Paper |
|---|----------|----------|-------|
| 1 | [Weakspot Identification Ensembles](#1-weakspot-identification-ensembles) | *Where* is the model weak, and which estimator finds it best? | *Advanced Methods for Weakspot Identification in Regression Models* (ICTAI) |
| 2 | [Guided Curation](#2-guided-curation) | Does selecting new data at the weakspot beat selecting it at random? | *Selecting Data Where the Model Is Weak: Statistically Guided Curation for Data-Centric Training* |
| 3 | [Data-Selective Training on Weakspots](#3-data-selective-training-on-weakspots) | What happens when you close the loop and repeat? | *Data-Centric Training on Weakspots: An Analysis* |

They build on each other: (1) establishes how to find a weakspot, (2) uses that location
to select data once, and (3) iterates the selection over many rounds.

---

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

streamlit run app/main.py       # interactive environment
```

The Streamlit app is the front door: every experiment below has a page for configuring
and running it interactively, plus a companion page for aggregating results across a
sweep. Long sweeps are better run head-less from the terminal — see each section.

```
app/pages/          Streamlit UI, one page per experiment + one per results view
scripts/weakspot/   engine for approach 1
scripts/dataselect/ engine for approach 2
scripts/iterative/  engine for approach 3
scripts/realdata/   real-dataset variant of the loop
data/experiment_results/<experiment>/   sweep result CSVs
```

Results are stored as CSVs with one row per configuration × method, each carrying a
`param_key` that identifies the configuration. Sweeps are resumable: completed
`param_key`s are skipped, so a run can be stopped and restarted freely.

---

## 1. Weakspot Identification Ensembles

### Motivation

Before a weakspot can be used, it has to be found — and it can only be *inferred*, from
a finite sample of the errors a trained model makes. That is an estimation problem, and
many statistical methods already exist for reconstructing a surface from scattered
samples: Gaussian processes, kernel interpolants, local averaging, quantile regression,
extreme-value statistics. How well any of them work when the quantity of interest is a
small, data-poor region rather than the surface everywhere had not been examined
systematically.

This approach benchmarks eight such methods as weakspot detectors, finds a structural
asymmetry between them, and builds ensembles that exploit it. The asymmetry is that a
method's ability to **locate** a weakspot's centre is largely independent of its ability
to **describe** its shape and extent. Quantile regression pins the centre but cannot
shape the region; Gaussian processes capture the shape but blur the centre. That
suggests pairing a *localizer* with a *landscape estimator*.

### Experimental setup

The difficulty in evaluating a weakspot detector is that on ordinary data there is no
independent reference — locating weakspots is the task itself. The setup therefore
constructs the ground truth deliberately:

1. **Generate a landscape.** A synthetic 2D function on the unit square, built as a sum
   of Gaussian bumps. The number of bumps controls how complex the function is.
2. **Sample it.** Draw `N` noisy observations.
3. **Induce a weakspot.** Withhold every sample inside a disk of radius `r` about a
   centre `c` *before training*, so the model never sees that region.
4. **Train.** Fit an MLP regressor to the surviving points.
5. **Measure error.** Evaluate the trained model across the whole input space and pair
   each location with its absolute prediction error. This labelled *error sample
   distribution* `(X, e)` is the only object a detector sees — it never receives the
   true centre.
6. **Detect.** Each method turns `(X, e)` into a non-negative *error-intensity surface*
   on a regular evaluation grid.
7. **Extract.** A single shared extractor makes every surface comparable: normalise it,
   threshold at its `q`-th quantile (default `q = 0.90`), take the largest connected
   component by integrated mass, and fit a 2σ ellipse to its weighted centroid and
   covariance. Because this stage is identical for every method, differences in the
   scores reflect the upstream surface alone.
8. **Score.** Compare the extracted ellipse against the induced disk.

**Is the induced gap really where the model is weak?** Not by assumption — it is
checked. Averaged over the nine gap locations, the model's mean absolute error inside
the withheld disk is 2.2× its error outside, reaching 4.6× at the centre of the domain.
Subtracting the error surface of a model trained on the *complete* data leaves a
residual that is positive almost only inside the gap, which isolates the error the
missing data alone causes.

**Metrics.** Two, because a detection can succeed on one axis and fail on the other:

- **Centroid distance** — the distance between the detected and true centre. Location
  only, ignoring shape. Lower is better.
- **IoU** — the overlap between the extracted ellipse and the induced disk. Rewards
  correct shape and extent as well as a correct centre. Higher is better.

A third, **paired** measure is used for every comparison between methods. Comparing two
methods by their mean IoU is misleading here: the spread of IoU across configurations
(σ ≈ 0.22) mostly reflects how much the *configurations* differ in difficulty, not how
much the *methods* differ. Since every method is evaluated on the same configurations,
that difficulty cancels if the difference is taken within a configuration. Reported per
comparison: the mean paired difference, a 95% bootstrap confidence interval clustered
over the 81 (landscape × noise × centre) cells, win/loss counts, and a Holm-corrected
Wilcoxon signed-rank test. The rank test rather than a *t*-test, because roughly two
thirds of per-configuration IoU values are exactly zero.

**The sweep.** All factors are crossed, giving **1,215 configurations**:

| Factor | Values | # |
|---|---|---|
| Function complexity `n_bumps` | 3, 5, 7 | 3 |
| Noise s.d. `noise_std` | 0.05, 0.15, 0.30 | 3 |
| Weakspot centre `c` | 3×3 grid on [0.25, 0.75]² | 9 |
| Exclusion radius `r` | 0.05, 0.10, 0.15 | 3 |
| Sample size `n_samples` | 500 → 1500, step 250 | 5 |

Held fixed: an MLP regressor of fixed capacity trained for 100 iterations, a 35×35
evaluation grid, and a 0.2 test split.

**A caveat worth knowing.** The extractor imposes a floor on the *absolute* IoU any
method can reach. Thresholding at a fixed quantile keeps a fixed fraction of grid cells
however large the true region is, so the fitted ellipse has a size prior of its own —
median area 5–6% of the domain, essentially independent of `r`. Against the smallest
weakspot (`r = 0.05`, area 0.8%) it is 6.7× too large, capping even a perfectly centred
detection at IoU ≈ 0.15. The ceiling rises to 0.61 at `r = 0.10` and does not bind at
`r = 0.15`. Absolute IoU is therefore **not** comparable across weakspot sizes; the
relative ranking, which shares the extractor, is unaffected.

### Methods

Seventeen detectors, registered in `DETECTION_METHODS` (`scripts/weakspot/detection.py`):

- **Base (8)** — Gaussian process regression, polynomial response surface, RBF
  interpolation, kNN performance mapping, LOESS, quantile regression, peaks-over-threshold
  (EVT), expected improvement (EI).
- **Simple ensembles (2)** — pixelwise arithmetic means, as duality-agnostic baselines.
- **Advanced ensembles (7)** — geometric consensus (QR×GPR, EVT×GPR), additive anchoring,
  multiplicative gating, variance-weighted mixture, disagreement amplification, and
  two-stage local refitting.

### Results

Mean over the 1,215 configurations. Δ IoU, win/loss and *p* are the paired comparison
against the strongest base method (peaks-over-threshold, the reference row).

| Tier | Method | Mean dist. | Mean IoU | Δ IoU | win/loss | *p* |
|---|---|---|---|---|---|---|
| Advanced | **EVT × GPR (geometric)** | 0.394 | **0.135** | +0.006 | 210/199 | 0.29 |
| Advanced | GPR/kNN (variance-weighted) | 0.381 | 0.124 | −0.004 | 184/265 | 4e−3 |
| Advanced | Anchored GPR (QR prior) | 0.380 | 0.118 | −0.011 | 181/270 | 2e−4 |
| Advanced | GPR/kNN (disagreement-amplified) | 0.378 | 0.111 | −0.018 | 162/283 | <1e−6 |
| Advanced | QR × GPR (geometric) | **0.339** | 0.070 | −0.059 | 516/327 | 1e−3 |
| Advanced | Gated GPR (QR anchor) | 0.476 | 0.043 | −0.086 | 119/340 | <1e−32 |
| Advanced | Local GPR (QR-localised) | 0.525 | 0.022 | −0.106 | 68/364 | <1e−53 |
| Simple | kNN + GPR (mean) | 0.381 | 0.124 | −0.004 | 182/267 | 3e−3 |
| Simple | kNN + Quantile (mean) | 0.423 | 0.092 | −0.037 | 110/309 | <1e−19 |
| Base | Peaks over Threshold (EVT) | 0.398 | 0.128 | ref | ref | — |
| Base | kNN Performance Mapping | 0.373 | 0.128 | −0.000 | 182/267 | 0.017 |
| Base | RBF Interpolation | 0.367 | 0.123 | −0.005 | 172/287 | 1e−3 |
| Base | Gaussian Process Regression | 0.381 | 0.117 | −0.012 | 180/271 | 2e−4 |
| Base | LOESS Local Regression | 0.419 | 0.083 | −0.045 | 135/304 | <1e−19 |
| Base | Polynomial Response Surface | 0.466 | 0.029 | −0.099 | 46/366 | <1e−56 |
| Base | Bayesian Optimization (EI) | 0.446 | 0.026 | −0.102 | 54/374 | <1e−52 |
| Base | Quantile Regression | 0.353 | 0.024 | −0.104 | 450/377 | <1e−19 |

Three findings:

1. **The gain is real but small.** EVT × GPR ranks first and is the only method with a
   positive paired difference against the strongest base detector. It beats seven of the
   eight base methods significantly; against the eighth — peaks-over-threshold, its own
   localizer component — the +0.006 margin is *not* statistically established.
2. **How the localizer is consumed decides the outcome.** Ensembles using its full
   surface stay in the leading group. The two that reduce it to an `argmax` are the worst
   methods in the study: they inherit the localizer's worst-case error directly.
3. **A localizer only contributes if its surface is sharp.** Every construction built on
   the smooth quantile-regression surface fails, `argmax` or not. A degree-2 surface over
   a box attains its maximum on the boundary unless it has an interior maximum, which is
   the edge-of-design-region degradation known for polynomial estimators.

### Scripts

| File | Role |
|---|---|
| `scripts/weakspot/datasets.py` | landscape generation, normalisation, exclusion-zone logic |
| `scripts/weakspot/models.py` | regressor construction and training |
| `scripts/weakspot/detection.py` | all 17 detectors, `DETECTION_METHODS` registry |
| `scripts/weakspot/extraction.py` | shared threshold → component → ellipse extractor, IoU |
| `scripts/weakspot/sweep.py` | `SWEEP_GRID`, `param_key`, resumable sweep driver |
| `scripts/weakspot/validation.py` | ground-truth alignment check (in/out error ratio) |
| `scripts/weakspot/plotting.py` | surface and detection plots |
| `scripts/AnalyzeExperimentWeakspotMethods.py` | analysis + figures from `sweep_results.csv` |
| `scripts/generate_paired_stats_figure.py` | paired statistics + forest plot |
| `scripts/generate_method_comparison_figures.py` | ranking and sensitivity figures |
| `scripts/generate_alignment_figure.py` | ground-truth validation figure |

**Run it.** Interactively via `app/pages/05_Weakspot_Experiment.py` (single runs and the
full sweep), with results aggregated in
`app/pages/06_Weakspot_Experiment_Visualise_Results.py`. Output lands in
`data/experiment_results/weakspot_experiment/sweep_results.csv`. Figures regenerate with:

```bash
python -m scripts.generate_paired_stats_figure
python -m scripts.generate_method_comparison_figures
python -m scripts.generate_alignment_figure
```

---

## 2. Guided Curation

### Motivation

Given a located weakspot, does it help? This approach tests the central data-centric
claim in its simplest form: acquire a fixed budget of new training points, either
*guided* — drawn near the detected weakspot — or *random*, and compare the resulting
models. One selection round, no iteration, so the effect of the guidance itself is
isolated.

The engine deliberately reuses the detectors from approach 1: a weakspot is located from
the error landscape, a selection distribution is built around it, and new points are
drawn from the pool according to that distribution. The mixing parameter `alpha` controls
how much of the budget is guided versus random, which makes the guided-vs-random contrast
a continuum rather than a binary.

### Scripts

| File | Role |
|---|---|
| `scripts/dataselect/error_landscape.py` | error surface used to place the selection |
| `scripts/dataselect/pipeline.py` | one full select → retrain → evaluate cycle |
| `scripts/dataselect/sweep.py` | JSON-configured grids, one results CSV per config |
| `scripts/dataselect/sweep_configs/*.json` | 17 grids (alpha boundary, detector variants, budget sweeps) |
| `scripts/dataselect/plots.py`, `plot_focused.py` | landscape, selection and outcome plots |
| `run_sweep.py` | head-less resumable runner for these grids |

**Run it.** Interactively via `app/pages/07_Data_Selective_Training.py`, aggregated in
`app/pages/08_Data_Selective_Training_Visualise_Results.py`. Head-less:

```bash
python run_sweep.py --config alpha_boundary --dry-run     # prints grid size and ETA
python run_sweep.py --config alpha_boundary --workers 6
```

Results land in `data/experiment_results/data_selective_training/sweep__<config>.csv`.

---

## 3. Data-Selective Training on Weakspots

### Motivation

Approach 2 selects once. This one closes the loop: detect the weakspot, select points,
retrain, then re-detect on the *updated* model and repeat. That makes the questions
dynamic. Does the weakspot move once it has been filled? Does guidance keep paying off
in later rounds, or does the advantage decay as the obvious gap closes? Is it better to
commit to the strongest signal early or to spread the budget across rounds?

Because every round writes one row per detector per iteration, a run produces a
trajectory rather than a single outcome, and the sweeps compare *schedules* — how the
guided share is allocated over rounds — as much as detectors.

### Scripts

| File | Role |
|---|---|
| `scripts/iterative/loop.py` | the iterative loop, `DEFAULTS`, `TRACKS`, `run_iterative` |
| `scripts/iterative/models.py` | model construction and warm-start retraining |
| `scripts/iterative/sweep.py` | JSON-configured grids, `param_key`, resume |
| `scripts/iterative/sweep_configs/*.json` | 45 grids (schedules, anchoring, budgets, architectures) |
| `scripts/iterative/plots.py`, `make_figures.py` | trajectory and learning-curve figures |
| `scripts/iterative/make_exploration_figures.py`, `make_perspective_figures.py` | exploratory views |

**Run it.** Interactively via `app/pages/09_Iterative_Data_Selective_Training.py`,
aggregated in `app/pages/10_Iterative_Visualise_Results.py`. Results land in
`data/experiment_results/iterative_data_selective_training/`.

### Real data

`app/pages/11_Real_Data_Example.py` and `scripts/realdata/loop_real.py` run the same loop
on a real tabular dataset, without a synthetic grid: detection works on evaluation-point
error smoothed over nearest neighbours in standardised feature space, and severity is a
ground-truth-free in/out error ratio. It exists to test whether the loop's behaviour
survives contact with data whose weakspots were not induced on purpose. See
`data/experiment_results/real_data_example/ANALYSIS_real_data.md`.

---

## Notes

- Sweep result CSVs are committed; raw datasets, trained models and generated figure
  directories are git-ignored (see `.gitignore`).
- Figure-generating scripts write directly into the corresponding paper's `figures/`
  folder, resolved relative to the script rather than the working directory, so they can
  be run from anywhere.
