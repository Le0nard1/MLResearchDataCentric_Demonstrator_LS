"""
Error-landscape view of a single data-selective-training round.

Where :mod:`scripts.dataselect.sweep` reduces every run to scalar metrics (MAE,
in/out-of-weakspot error, distance, IoU), this module keeps the **spatial** object
those scalars summarise: the model's absolute error as a function of position in the
input space. One study renders the whole story of a round on the same square:

    error before  ->  induced weakspot  ->  detected weakspot  ->
    error after guided retraining  vs.  error after the random baseline

together with the difference maps (``after - before``) that show *where* each
retraining spent its budget. The guided run repairs the weak region; the random
baseline spreads the same number of points uniformly and barely dents it.

The landscape is evaluated on a dense regular grid against the **noiseless** ground
truth, so it is genuine model error and not label noise. The scalar metrics beside
it are still computed on the standard uniform evaluation sample, so the numbers
reported here are directly comparable with the sweep's.

Everything runs on the same building blocks as the page and the sweep
(:mod:`scripts.dataselect.pipeline`), and the RNG is drawn in exactly the order
:func:`scripts.dataselect.sweep.run_one` uses, so a single-detector study at a given
seed reproduces that sweep row.

Used by
-------
* ``app/pages/07_Data_Selective_Training.py`` - the "Error Landscape" tab.
* ``scripts/generate_error_landscape_figure.py`` - the paper figure.
"""
from __future__ import annotations

import copy

import numpy as np

from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import (
    extract_weakspot, induced_mask, iou_ellipse_at_thresholds,
)
from scripts.dataselect import pipeline as P

# Resolution of the *rendering* grid. Finer than the detection grid (35) because it
# is only evaluated, never fitted on: 60x60 = 3600 forward passes per model.
LAND_RES_DEFAULT = 60
IOU_HEADLINE_Q = 0.90

# The paper's operating point with the enlarged gap (radius 0.25) - identical to the
# anchor of the ``best_vary_detector`` / ``iso_*`` sweep configs, so this study sits
# at the same point in parameter space as the isolation experiments.
OPERATING_POINT = dict(
    # setup / data
    n_bumps=5, noise_std=0.05, n_pool_total=2000, shift_strength=0.0,
    shift_center_x=0.30, shift_center_y=0.30, shift_spread=0.15,
    radius=0.25, center_x=0.5, center_y=0.5,
    # model / initial training
    model_name="MLP Neural Network", complexity=0.5,
    iters_initial=12, n_train=100, early_stopping=True,
    # evaluation / identification
    n_eval=1000, grid_res=35, extract_q=0.85,
    # selection
    sel_method=P.DEFAULT_SEL_METHOD, sel_mode="Sample ∝ weight",
    sel_sigma=0.5, n_select=100, n_candidate=2000, mix_ratio=0.5,
    # retraining
    iters_retrain=400, warm_start=True,
)

# Detectors used for the landscape study: the five the broad sweep ran, so the study
# sits at the same point in method space as the sweep and no single detector is
# cherry-picked. Averaging over them also keeps the "detected weakspot" panel honest
# - it shows where *the method* looks, not where one favourable detector happened to
# look on one seed.
BROAD_SWEEP_DETECTORS = (
    "EVT × GPR (geometric)",
    "kNN Performance Mapping",
    "Peaks over Threshold (EVT)",
    "GPR/kNN (variance-weighted)",
    "RBF Interpolation",
)


# -------------------------------------------------------------
# Error landscape of one model
# -------------------------------------------------------------
def error_landscape(model, land_flat, n_bumps: int) -> np.ndarray:
    """Absolute error of ``model`` at every point of ``land_flat``, measured against
    the noiseless ground truth. Returns a flat (M,) array."""
    y_true = P.true_function(land_flat, n_bumps=n_bumps)
    return np.abs(y_true - model.predict(land_flat))


def landscape_grid(res: int = LAND_RES_DEFAULT):
    """Rendering grid over [0,1]^2 - ``(xx, yy, flat)``, as :func:`create_grid`."""
    return create_grid(resolution=int(res))


# -------------------------------------------------------------
# One seed of the study
# -------------------------------------------------------------
def run_seed(params: dict, seed: int, detector: str, land_flat: np.ndarray) -> dict:
    """One full round (initial -> detect -> guided retrain, plus the random baseline)
    with the error landscape of all three models recorded on ``land_flat``.

    The RNG draw order mirrors :func:`scripts.dataselect.sweep.run_one` exactly
    (pool -> training subsample -> eval set -> candidate pool -> baseline pick ->
    guided pick), so this reproduces that sweep row for the same seed and detector.
    """
    p = {**OPERATING_POINT, **params}
    rng = np.random.RandomState(int(seed))
    center = np.array([float(p["center_x"]), float(p["center_y"])])
    nb, r = int(p["n_bumps"]), float(p["radius"])
    n_select, iters_retrain = int(p["n_select"]), int(p["iters_retrain"])
    es = bool(p["early_stopping"])
    warm = bool(p["warm_start"]) and AVAILABLE_MODELS[p["model_name"]] == "mlp"

    def _retrain(base, X, y):
        """Retrain on the new points only; warm-start continues ``base``'s weights.
        The guided model and the baseline both go through here, from the same initial
        model, so the head-to-head stays fair."""
        if warm:
            m = copy.deepcopy(base)
            m.named_steps["model"].max_iter = iters_retrain
        else:
            m = build_model(AVAILABLE_MODELS[p["model_name"]],
                            complexity=float(p["complexity"]),
                            iterations=iters_retrain, early_stopping=es)
        m.fit(X, y)
        return m

    # ---- setup: pool, induced gap, training subsample -------------------
    X_all = P.sample_inputs(int(p["n_pool_total"]), rng,
                            shift_strength=float(p["shift_strength"]),
                            shift_center=(float(p["shift_center_x"]),
                                          float(p["shift_center_y"])),
                            shift_spread=float(p["shift_spread"]))
    y_all = P.label(X_all, rng, n_bumps=nb, noise_std=float(p["noise_std"]))
    if r > 0:
        X_keep, y_keep, X_excl, _ = P.induce_weakspot(X_all, y_all, center, r)
    else:
        X_keep, y_keep, X_excl = X_all, y_all, X_all[:0]
    k = min(int(p["n_train"]), len(X_keep))
    tr_idx = rng.choice(len(X_keep), size=k, replace=False)
    X_tr0, y_tr0 = X_keep[tr_idx], y_keep[tr_idx]

    # ---- evaluation sample (scalar metrics) + detection grid ------------
    X_eval = P.sample_inputs(int(p["n_eval"]), rng, shift_strength=0.0)
    y_eval = P.true_function(X_eval, n_bumps=nb)
    xx, yy, grid_flat = create_grid(resolution=int(p["grid_res"]))
    gt_mask = (induced_mask(xx, yy, "Circle (radius)", center, r, None)
               if r > 0 else np.zeros(xx.shape, dtype=bool))

    def _region(err):
        return (P.region_error(X_eval, err, center, r) if r > 0
                else (float("nan"), float("nan")))

    # ---- initial model --------------------------------------------------
    model0 = build_model(AVAILABLE_MODELS[p["model_name"]],
                         complexity=float(p["complexity"]),
                         iterations=int(p["iters_initial"]),
                         warm_start=warm, early_stopping=es)
    model0.fit(X_tr0, y_tr0)
    _, err0, m_init = P.evaluate(model0, X_eval, y_eval)
    ein0, eout0 = _region(err0)

    # ---- shared candidate pool ------------------------------------------
    X_cand = P.sample_inputs(int(p["n_candidate"]), rng, shift_strength=0.0)
    y_cand = P.label(X_cand, rng, n_bumps=nb, noise_std=float(p["noise_std"]))

    # ---- random baseline (drawn first, as in the sweep) -----------------
    n_sel = min(n_select, len(X_cand))
    rand_idx = rng.choice(len(X_cand), size=n_sel, replace=False)
    modelR = _retrain(model0, X_cand[rand_idx], y_cand[rand_idx])
    _, errR, m_base = P.evaluate(modelR, X_eval, y_eval)
    einR, eoutR = _region(errR)

    # ---- weakspot identification on the initial error -------------------
    surf0 = P.normalize_surface(DETECTION_METHODS[detector](X_eval, err0, grid_flat))
    ext = extract_weakspot(surf0, xx, yy, threshold_quantile=float(p["extract_q"]))
    iou = iou_ellipse_at_thresholds(surf0, xx, yy, gt_mask, (IOU_HEADLINE_Q,))
    if ext is None:                     # no pixel above threshold -> surface argmax
        det_center = tuple(grid_flat[int(np.argmax(surf0))])
    else:
        det_center = tuple(ext["center"])
    dist = (float(np.hypot(det_center[0] - center[0], det_center[1] - center[1]))
            if r > 0 else float("nan"))

    # ---- guided selection + retrain --------------------------------------
    sel_idx, sel_w = P.select_by_weakspot(
        X_cand, det_center, float(p["sel_sigma"]), n_select, rng,
        mode=p["sel_mode"], method=p["sel_method"], ext=ext, surf=surf0,
        mix_ratio=float(p["mix_ratio"]))
    model1 = _retrain(model0, X_cand[sel_idx], y_cand[sel_idx])
    _, err1, m_guided = P.evaluate(model1, X_eval, y_eval)
    ein1, eout1 = _region(err1)

    # ---- the landscapes ---------------------------------------------------
    L0 = error_landscape(model0, land_flat, nb)
    L1 = error_landscape(model1, land_flat, nb)
    LR = error_landscape(modelR, land_flat, nb)

    return dict(
        seed=int(seed), detector=detector,
        xx=xx, yy=yy, gt_mask=gt_mask, center=center, radius=r,
        surf0=surf0, ext=ext, det_center=det_center,
        distance=dist, iou=float(iou.get(IOU_HEADLINE_Q, float("nan"))),
        X_tr0=X_tr0, y_tr0=y_tr0, X_excl=X_excl,
        X_sel=X_cand[sel_idx], X_rand=X_cand[rand_idx], sel_w=sel_w,
        land_init=L0, land_guided=L1, land_random=LR,
        m_init=m_init, m_guided=m_guided, m_base=m_base,
        err_in=(ein0, ein1, einR), err_out=(eout0, eout1, eoutR),
    )


# -------------------------------------------------------------
# The study: average the landscape over seeds
# -------------------------------------------------------------
def run_study(seeds, detectors=BROAD_SWEEP_DETECTORS, params: dict | None = None,
              land_res: int = LAND_RES_DEFAULT, progress=None) -> dict:
    """Run :func:`run_seed` over every (seed, detector) pair and average the three
    error landscapes across them.

    Averaging over seeds removes the single-run speckle (a lone initial model here is
    a 12-iteration MLP and its error map is noisy) and leaves the systematic
    structure: the gap, its repair, and what each retraining did elsewhere. Averaging
    over ``detectors`` (a single name or a sequence, by default the broad sweep's
    five) keeps the picture from resting on one favourable identification method.

    ``progress`` is an optional ``callable(done, total)`` for a UI progress bar.
    Returns the mean landscapes and difference maps, the mean detector surface,
    per-run metric arrays with their aggregates, and one representative run.
    """
    seeds = [int(s) for s in seeds]
    if isinstance(detectors, str):
        detectors = [detectors]
    detectors = list(detectors)
    p = {**OPERATING_POINT, **(params or {})}
    land_xx, land_yy, land_flat = landscape_grid(land_res)

    runs, total = [], len(seeds) * len(detectors)
    for s in seeds:
        for det in detectors:
            runs.append(run_seed(p, s, det, land_flat))
            if progress is not None:
                progress(len(runs), total)

    stack = lambda key: np.stack([r[key] for r in runs])
    L_init, L_guided, L_random = (stack("land_init"), stack("land_guided"),
                                  stack("land_random"))
    shp = land_xx.shape

    # per-run scalar metrics (evaluation sample - comparable with the sweep)
    mae = {k: np.array([r[f"m_{k}"]["MAE"] for r in runs])
           for k in ("init", "guided", "base")}
    err_in = np.array([r["err_in"] for r in runs])      # (N, 3): init, guided, base
    distance = np.array([r["distance"] for r in runs])
    iou = np.array([r["iou"] for r in runs])

    # Representative run: the one whose guided-minus-baseline gap is the median, so
    # the single-run overlays (training points, detected ellipse) are typical rather
    # than cherry-picked.
    gap = mae["guided"] - mae["base"]
    rep = runs[int(np.argsort(gap)[len(gap) // 2])]

    return dict(
        params=p, detectors=detectors, detector=", ".join(detectors),
        seeds=seeds, n_seeds=len(seeds), n_runs=len(runs),
        land_xx=land_xx, land_yy=land_yy,
        init=L_init.mean(0).reshape(shp),
        guided=L_guided.mean(0).reshape(shp),
        random=L_random.mean(0).reshape(shp),
        d_guided=(L_guided - L_init).mean(0).reshape(shp),
        d_random=(L_random - L_init).mean(0).reshape(shp),
        d_head=(L_guided - L_random).mean(0).reshape(shp),
        xx=rep["xx"], yy=rep["yy"], center=rep["center"], radius=rep["radius"],
        surf_mean=np.stack([r["surf0"] for r in runs]).mean(0),
        det_centers=np.array([r["det_center"] for r in runs]),
        rep=rep,
        mae=mae, err_in=err_in, distance=distance, iou=iou,
        gap=gap, win_rate=float(np.mean(gap < 0)),
    )


def summarise(study: dict) -> dict:
    """Scalar summary of a study - the numbers quoted in the paper's text."""
    m, ein = study["mae"], study["err_in"]
    ci = lambda a: 1.96 * float(np.std(a, ddof=1)) / np.sqrt(max(len(a), 1))
    return dict(
        n_seeds=study["n_seeds"], n_runs=study["n_runs"],
        detectors=study["detectors"],
        mae_init=float(m["init"].mean()), mae_guided=float(m["guided"].mean()),
        mae_base=float(m["base"].mean()),
        mae_guided_ci=ci(m["guided"]), mae_base_ci=ci(m["base"]),
        gap=float(study["gap"].mean()), gap_ci=ci(study["gap"]),
        win_rate=study["win_rate"],
        errin_init=float(np.nanmean(ein[:, 0])),
        errin_guided=float(np.nanmean(ein[:, 1])),
        errin_base=float(np.nanmean(ein[:, 2])),
        distance=float(np.nanmean(study["distance"])),
        iou=float(np.nanmean(study["iou"])),
    )


# -------------------------------------------------------------
# Region bookkeeping for the difference maps
# -------------------------------------------------------------
def region_means(land: np.ndarray, land_xx, land_yy, center, radius: float):
    """Mean of a landscape inside vs outside the induced weakspot circle."""
    d = np.hypot(land_xx - center[0], land_yy - center[1])
    inside = d <= float(radius)
    return float(land[inside].mean()), float(land[~inside].mean())
