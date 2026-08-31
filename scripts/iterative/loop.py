"""The iterative data-selective-training engine.

One call to :func:`run_iterative` executes a whole experiment:

    initial train  →  for k = 1 … K:
        evaluate → identify the *current* weakspot → select data around it →
        continue training → re-evaluate  (with a matched random baseline
        trained in parallel under identical conditions)

and returns every per-iteration metric. It is deliberately free of Streamlit so
the experiment page, the parameter sweep and its parallel workers all run the
*same* code path.

What the loop adds over the single-round study
----------------------------------------------
* **Regimes.** Each strategy (guided / random) can be trained under three
  regimes. The default — and the focused setup — is *new-only*: every round
  continues training on **just that round's newly selected points**, which is
  the companion paper's own conservative protocol repeated, and the regime in
  which over-concentration is punished hardest. Also available are
  *accumulative* (retrain on the whole growing set — the industrial protocol the
  paper listed as untested) and *scaled-accum*, a size-matched control that
  subsamples the accumulated pool down to the new-only budget; enabling those
  turns the run into the regime comparison. The **first** regime listed is the
  primary: its guided model is the one the detector is run against each round,
  and every headline number is reported against it.
* **Schedules.** The rehearsal mix α and kernel width σ — the two controls the
  paper found dominant — may now vary with the iteration, including adaptively
  as the detected weakspot heals. The learning rate is scheduled the same way,
  one step per loop iteration.
* **A migrating target.** The weakspot is re-identified against the current
  model each round (or pinned once, as an ablation), so detector drift and
  weakspot migration become measurable.
* **Pool exhaustion and scarcity.** The candidate pool can be redrawn each round
  (the paper's protocol) or fixed once and consumed without replacement. It can
  also be *thinned inside the induced gap* (``pool_deficit``), without which the
  deficit is a one-shot affair — a single round of uniform candidates fills the
  hole, the random arm fills it just as well, and from round two guidance has
  nothing real left to aim at.
* **Round damping.** ``round_blend`` averages each round's model with the
  previous one, damping the step *between rounds* rather than between epochs.
  Each round fits a fresh handful of points, so its own optimum is a
  high-variance estimate; taking it whole is what makes the trajectory bumpy.

Conventions
-----------
Iteration 0 is the shared initial model, so every metric series has length
``K + 1`` while the per-round series (schedules, detection) have length ``K``.
Guided and random draw from the **same** candidate pool each round, so the
head-to-head isolates the selection rule and nothing else.
"""
from __future__ import annotations

import copy

import numpy as np

from scripts.dataselect import pipeline as P
from scripts.iterative import models as M
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import (
    extract_weakspot, induced_mask, iou_ellipse_at_thresholds,
)
from scripts.weakspot.models import AVAILABLE_MODELS

IOU_HEADLINE_Q = 0.90

# Track key → (strategy, regime). ``g``/``r`` = guided / random; the suffix is
# the regime. The accumulative pair is always run — it is the primary result and
# the guided-accumulative model is the one whose weakspot drives selection.
TRACKS = {
    "gacc": ("guided", "accumulative"),
    "racc": ("random", "accumulative"),
    "gnew": ("guided", "new-only"),
    "rnew": ("random", "new-only"),
    "gsca": ("guided", "scaled-accum"),
    "rsca": ("random", "scaled-accum"),
}
REGIMES = ("accumulative", "new-only", "scaled-accum")
# Regime → its (guided, random) track pair. The primary regime's guided track is
# the "driver": the model whose error surface is searched for the weakspot, and
# the one every headline metric is reported against.
DRIVER_TRACKS = {"accumulative": ("gacc", "racc"),
                 "new-only": ("gnew", "rnew"),
                 "scaled-accum": ("gsca", "rsca")}
POOL_MODES = ("Fresh pool each iteration", "Fixed pool (consumed)")
DETECT_MODES = ("Re-detect each iteration", "Detect once (pinned target)")

# Defaults are the companion paper's operating point (Sec. "The Broad Sweep"),
# with the enlarged gap radius 0.25 the isolation experiments used so a local
# repair is large enough to register over the whole area. Everything the
# iterative study adds defaults to the *neutral* setting, so an out-of-the-box
# run reproduces the paper's configuration looped K times and each new axis can
# be switched on one at a time.
DEFAULTS: dict = dict(
    seed=42,
    # data / landscape
    n_bumps=5, noise_std=0.05, n_pool_total=2000,
    shift_strength=0.0, shift_center_x=0.30, shift_center_y=0.30, shift_spread=0.15,
    # induced weakspot
    radius=0.25, center_x=0.5, center_y=0.5,
    # initial model
    model_name="MLP Neural Network", complexity=0.5,
    arch=M.LEGACY_ARCH, activation="relu", solver="adam",
    alpha=1e-4, batch_size=0,                  # 0 → sklearn's "auto"
    iters_initial=12, n_train=100,
    # evaluation / detection
    n_eval=1000, grid_res=35, extract_q=0.85,
    detector="Quantile Regression", detect_mode=DETECT_MODES[0],
    # selection
    sel_method=P.DEFAULT_SEL_METHOD, sel_mode="Sample ∝ weight",
    sel_sigma=0.5, sigma_schedule="Constant", sigma_rate=0.5,
    mix_ratio=0.5, mix_schedule="Constant", mix_rate=0.5,
    n_select=100, n_candidate=2000, pool_mode=POOL_MODES[0], pool_deficit=0.0,
    gaussian_kernel=True,
    # retraining / loop
    # Early stopping defaults OFF here, unlike the single-round study. Holding out
    # 10% of a 100-point round leaves a 10-point validation set — far too noisy to
    # stop on — and it makes the retraining budget unknowable, since a round then
    # runs some unrecorded number of epochs rather than the iters_retrain it was
    # given. With it off, "continue for iters_retrain epochs" is literally true,
    # which matters because that budget is a swept axis. Turning it back on is
    # safe: the per-round stopping state is reset (see models.reset_stopping_state).
    warm_start=True, early_stopping=False, iters_retrain=400, n_iterations=8,
    lr_init=1e-3, lr_schedule="Constant", lr_gamma=0.7, lr_min=1e-5, lr_step=2,
    round_blend=1.0,
    # Only guided-vs-random on each round's NEW points: the companion paper's own
    # retraining protocol, now repeated. Accumulative and scaled-accum stay
    # available and are switched on for the regime comparison, but they are not
    # part of the focused setup. The first regime listed is the primary: its
    # guided model is the one the detector is run against.
    regimes=["new-only"], single_shot=False, tie_model_seed=False,
    # Freeze the initial model's first hidden layer for every later round (both
    # arms): retraining then adapts only the upper layers on a fixed feature
    # map. Warm-start MLPs only; needs at least two hidden layers.
    freeze_first=False,
)


def default_config(**overrides) -> dict:
    """A full config: the paper's operating point plus any overrides."""
    cfg = dict(DEFAULTS)
    cfg["regimes"] = list(DEFAULTS["regimes"])
    cfg.update(overrides)
    return cfg


def _metric_store() -> dict:
    return {k: [] for k in ("mae", "rmse", "r2", "err_in", "err_out", "n_train")}


def _severity(X_eval, err, center, ext) -> float:
    """Mean error inside the *detected* region ÷ mean error outside it.

    Uses only the model's own error distribution (no ground truth), so the
    adaptive schedules driven by it remain implementable in the fixed-dataset
    setting. Returns 1.0 (= "no weak region stands out") when the split is
    degenerate.
    """
    if ext is not None and ext.get("sigma_x1") and ext.get("sigma_x2"):
        r = float(np.sqrt(float(ext["sigma_x1"]) * float(ext["sigma_x2"])))
    else:
        r = 0.15
    r = float(np.clip(r, 0.03, 0.5))
    e_in, e_out = P.region_error(X_eval, err, center, r)
    if not np.isfinite(e_in) or not np.isfinite(e_out) or e_out <= 1e-12:
        return 1.0
    return float(e_in / e_out)


def _select_hard(X_pool, center, ext, sigma, mix, n_select, rng):
    """Guided selection with the Gaussian kernel switched off.

    The kernel is replaced by hard membership of the detected region: a candidate
    is either in it or not, and the guided share is drawn uniformly from those
    that are. This leaves the rehearsal mix α as the *only* control of how
    concentrated the round is, which is what makes it a clean test of whether the
    kernel width and the mix really are the substitutable coverage controls the
    single-round study took them for.

    Membership is the detected 2σ ellipse when one was extracted, and a disc of
    radius ``sigma`` about the detected centre otherwise. If too few candidates
    fall inside, the closest ones are taken instead, so a round never silently
    selects fewer points than it was asked for.
    """
    X = np.asarray(X_pool, dtype=float)
    c = np.asarray(center, dtype=float)
    n_select = int(min(n_select, len(X)))
    if ext is not None and ext.get("sigma_major") and ext.get("sigma_minor"):
        a = 2.0 * float(ext["sigma_major"]); b = 2.0 * float(ext["sigma_minor"])
        th = float(ext.get("angle_rad", 0.0))
        d = X - c
        u = d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)
        v = -d[:, 0] * np.sin(th) + d[:, 1] * np.cos(th)
        score = (u / max(a, 1e-6)) ** 2 + (v / max(b, 1e-6)) ** 2
    else:
        score = (np.linalg.norm(X - c, axis=1) / max(float(sigma), 1e-6)) ** 2
    inside = score <= 1.0
    w = inside.astype(float)                       # binary weights, for plotting

    n_guided = int(round(float(np.clip(mix, 0.0, 1.0)) * n_select))
    pool_in = np.flatnonzero(inside)
    if len(pool_in) < n_guided:                    # too few inside — take the nearest
        pool_in = np.argsort(score)[:n_guided]
    g_idx = (rng.choice(pool_in, size=n_guided, replace=False)
             if n_guided > 0 else np.array([], dtype=int))
    rest = np.setdiff1d(np.arange(len(X)), g_idx)
    n_unif = min(n_select - len(g_idx), len(rest))
    u_idx = (rng.choice(rest, size=n_unif, replace=False)
             if n_unif > 0 else np.array([], dtype=int))
    return np.concatenate([g_idx, u_idx]).astype(int), w


def run_iterative(cfg: dict, keep_rounds: bool = True, progress=None) -> dict:
    """Run one iterative experiment.

    ``cfg``          a :func:`default_config` dictionary.
    ``keep_rounds``  keep the per-iteration pools/selections/error fields needed
                     by the visualisation (off for sweeps, where only the
                     metrics are written out).
    ``progress``     optional ``callable(done, total, text)`` for a progress bar.

    Returns a dict with ``iters``, per-track metric series, the realised
    schedules, per-iteration detection quality, the optional budget-matched
    single-shot control, and (if kept) the per-round detail.
    """
    c = default_config(**cfg)
    K = int(c["n_iterations"])
    rng = np.random.RandomState(int(c["seed"]))
    # A second stream for the *optional* controls (scaled-accum subsampling,
    # single-shot) so switching them on never perturbs the core draws — the
    # primary tracks are reproducible regardless of which extras are enabled.
    rng_aux = np.random.RandomState(int(c["seed"]) + 100_003)
    # The single-shot control gets its own stream again, so it draws the same
    # points whether or not the scaled-accum regime consumed rng_aux before it.
    rng_ss = np.random.RandomState(int(c["seed"]) + 200_003)

    center = np.array([float(c["center_x"]), float(c["center_y"])])
    nb, noise = int(c["n_bumps"]), float(c["noise_std"])
    model_key = AVAILABLE_MODELS[c["model_name"]]
    warm = bool(c["warm_start"]) and model_key == "mlp"
    m_seed = int(c["seed"]) if bool(c.get("tie_model_seed")) else 42
    batch = int(c["batch_size"]) if int(c.get("batch_size", 0) or 0) > 0 else "auto"

    # Regimes keep the order they were listed in: the **first** one is the primary,
    # and its guided model is what the detector is run against each round. Which
    # model diagnoses the weakspot is not a detail — an accumulative model and a
    # new-only model have different error surfaces, so they aim the next selection
    # at different places.
    seen: set = set()
    regimes = [r for r in c["regimes"]
               if r in REGIMES and not (r in seen or seen.add(r))] or ["new-only"]
    primary = regimes[0]
    active = [k for k, (_s, reg) in TRACKS.items() if reg in regimes]
    gkey, rkey = DRIVER_TRACKS[primary]

    def _new_model(epochs: int, lr: float, warm_flag: bool):
        return M.build_iter_model(
            model_key, arch=c["arch"], complexity=float(c["complexity"]),
            iterations=int(epochs), warm_start=warm_flag,
            early_stopping=bool(c["early_stopping"]), lr_init=float(lr),
            solver=c["solver"], activation=c["activation"], alpha=float(c["alpha"]),
            batch_size=batch, random_state=m_seed)

    # ── setup ────────────────────────────────────────────────
    X_all = P.sample_inputs(int(c["n_pool_total"]), rng,
                            shift_strength=float(c["shift_strength"]),
                            shift_center=(float(c["shift_center_x"]),
                                          float(c["shift_center_y"])),
                            shift_spread=float(c["shift_spread"]))
    y_all = P.label(X_all, rng, n_bumps=nb, noise_std=noise)
    r_eff = float(c["radius"])
    if r_eff > 0:
        X_keep, y_keep, X_excl, _ = P.induce_weakspot(X_all, y_all, center, r_eff)
    else:
        X_keep, y_keep, X_excl = X_all, y_all, X_all[:0]
    k_tr = min(int(c["n_train"]), len(X_keep))
    tr_idx = rng.choice(len(X_keep), size=k_tr, replace=False)
    X_tr0, y_tr0 = X_keep[tr_idx], y_keep[tr_idx]

    X_eval = P.sample_inputs(int(c["n_eval"]), rng, shift_strength=0.0)
    y_eval = P.true_function(X_eval, n_bumps=nb)
    xx, yy, grid_flat = create_grid(resolution=int(c["grid_res"]))
    gt_mask = (induced_mask(xx, yy, "Circle (radius)", center, r_eff, None)
               if r_eff > 0 else np.zeros(xx.shape, dtype=bool))

    def _region(err):
        return (P.region_error(X_eval, err, center, r_eff) if r_eff > 0
                else (float("nan"), float("nan")))

    # ── candidate availability ───────────────────────────────
    # The single-round study withheld the gap from the *training set* but drew
    # candidates uniformly over the whole square. In a loop that makes the deficit
    # a one-shot affair: 100 uniform points fill a radius-0.25 hole in the very
    # first round, the random arm fills it just as well as the guided one, and from
    # round two there is nothing left to aim at. ``pool_deficit`` removes that
    # artefact by thinning the gap region out of the candidate pool as well, which
    # is also the more faithful reading of the fixed-dataset setting the study is
    # motivated by: a region is under-represented precisely *because* data there is
    # hard to come by, so the pool under-represents it too. At, say, 0.9 the few
    # candidates that do exist inside the gap are rare enough that uniform sampling
    # almost never picks them up, while a guided kernel concentrated there harvests
    # them — which is the actual industrial claim being tested.
    pool_deficit = float(np.clip(c.get("pool_deficit", 0.0), 0.0, 1.0))
    blend = float(np.clip(c.get("round_blend", 1.0), 0.0, 1.0))

    def _pool_inputs(n: int) -> np.ndarray:
        """``n`` uniform candidates, with the induced gap thinned by pool_deficit."""
        if pool_deficit <= 0.0 or r_eff <= 0:
            return P.sample_inputs(n, rng, shift_strength=0.0)
        kept, got = [], 0
        while got < n:                       # rejection sampling to a fixed size
            X = P.sample_inputs(int((n - got) * 1.5) + 32, rng, shift_strength=0.0)
            inside = np.linalg.norm(X - center, axis=1) <= r_eff
            drop = inside & (rng.random_sample(len(X)) < pool_deficit)
            X = X[~drop]
            kept.append(X); got += len(X)
        return np.vstack(kept)[:n]

    detect_fn = DETECTION_METHODS[c["detector"]]

    def _detect(err):
        """Detection surface, centre and extracted ellipse for an error field."""
        surf = P.normalize_surface(detect_fn(X_eval, err, grid_flat))
        ext = extract_weakspot(surf, xx, yy, threshold_quantile=float(c["extract_q"]))
        if ext is None:
            g = grid_flat[int(np.argmax(surf))]
            return surf, (float(g[0]), float(g[1])), None
        return surf, tuple(float(v) for v in ext["center"]), ext

    # ── initial (shared) model ───────────────────────────────
    m0 = _new_model(int(c["iters_initial"]), float(c["lr_init"]), warm)
    m0.fit(X_tr0, y_tr0)
    _, err0, met0 = P.evaluate(m0, X_eval, y_eval)
    ein0, eout0 = _region(err0)

    hist = {k: _metric_store() for k in active}
    for k in active:
        for name, v in (("mae", met0["MAE"]), ("rmse", met0["RMSE"]),
                        ("r2", met0["R2"]), ("err_in", ein0), ("err_out", eout0),
                        ("n_train", len(X_tr0))):
            hist[k][name].append(v)

    # Warm-start: one persistent model per track, each branching from the shared
    # initial model and *continued* every round. From-scratch: rebuilt per fit.
    # Under ``freeze_first`` each track instead continues a head model over the
    # initial model's frozen scaler + first hidden layer (see models.py); the
    # frozen features are shared, the heads are independent copies.
    if warm and bool(c.get("freeze_first")):
        track_models = {k: M.make_frozen_head(m0, X_tr0, y_tr0) for k in active}
    else:
        track_models = {k: copy.deepcopy(m0) for k in active} if warm else {}
    mg = track_models[gkey] if warm else m0

    # Accumulating training sets. Maintained even when the accumulative tracks are
    # switched off, because scaled-accum subsamples from them.
    Xg, yg = X_tr0.copy(), y_tr0.copy()
    Xr, yr = X_tr0.copy(), y_tr0.copy()

    # ── candidate pool ───────────────────────────────────────
    fixed_pool = c["pool_mode"] == POOL_MODES[1]
    if fixed_pool:
        X_pool = _pool_inputs(int(c["n_candidate"]))
        y_pool = P.label(X_pool, rng, n_bumps=nb, noise_std=noise)
        avail = {"g": np.ones(len(X_pool), dtype=bool),
                 "r": np.ones(len(X_pool), dtype=bool)}

    sched = {k: [] for k in ("lr", "mix", "sigma", "severity")}
    # Detection record per round. The ellipse geometry (major/minor axis and
    # orientation) is kept alongside the centre so the weakspot's *migration* can
    # be drawn as a path of extents rather than a path of points — a detector that
    # keeps the same centre while its region balloons is telling a very different
    # story from one that tracks a tight region across the square.
    det = {k: [] for k in ("distance", "iou", "drift", "cx", "cy",
                           "smaj", "smin", "angle", "area")}
    rounds: list[dict] = []
    pinned = None                      # detection reused when the target is pinned
    sev0 = None

    def _fit_eval(X, y, track, lr, epochs):
        """Fit one track on ``(X, y)`` and evaluate on the held-out landscape."""
        if warm:
            m = track_models[track]
            M.apply_runtime(m, lr=lr, epochs=epochs)
            snap = M.snapshot_weights(m) if blend < 1.0 else None
            m.fit(X, y)
            M.blend_weights(m, snap, blend)     # damp the step between rounds
        else:
            m = _new_model(epochs, lr, False)
            m.fit(X, y)
        _, err, met = P.evaluate(m, X_eval, y_eval)
        e_in, e_out = _region(err)
        hist[track]["mae"].append(met["MAE"])
        hist[track]["rmse"].append(met["RMSE"])
        hist[track]["r2"].append(met["R2"])
        hist[track]["err_in"].append(e_in)
        hist[track]["err_out"].append(e_out)
        hist[track]["n_train"].append(len(X))
        return m

    def _carry(track):
        """Repeat a track's previous values — used when a consumed pool leaves a
        round with nothing to select, so every series keeps length ``K + 1``."""
        for name, series in hist[track].items():
            series.append(series[-1])

    for it in range(K):
        # ---- identify the current weakspot -------------------------------
        _, err_cur, _ = P.evaluate(mg, X_eval, y_eval)
        if pinned is None or c["detect_mode"] == DETECT_MODES[0]:
            surf_g, c_g, ext_g = _detect(err_cur)
            pinned = (surf_g, c_g, ext_g)
        else:
            surf_g, c_g, ext_g = pinned

        sev = _severity(X_eval, err_cur, c_g, ext_g)
        if sev0 is None:
            sev0 = sev
        dist = (float(np.hypot(c_g[0] - center[0], c_g[1] - center[1]))
                if r_eff > 0 else float("nan"))
        iou = iou_ellipse_at_thresholds(surf_g, xx, yy, gt_mask, (IOU_HEADLINE_Q,))
        drift = (float(np.hypot(c_g[0] - det["cx"][-1], c_g[1] - det["cy"][-1]))
                 if det["cx"] else 0.0)
        det["distance"].append(dist)
        det["iou"].append(float(iou.get(IOU_HEADLINE_Q, float("nan"))))
        det["drift"].append(drift)
        det["cx"].append(c_g[0]); det["cy"].append(c_g[1])
        det["smaj"].append(float(ext_g["sigma_major"]) if ext_g else float("nan"))
        det["smin"].append(float(ext_g["sigma_minor"]) if ext_g else float("nan"))
        det["angle"].append(float(ext_g["angle_rad"]) if ext_g else 0.0)
        det["area"].append(float(ext_g["area_frac"]) if ext_g else float("nan"))

        # ---- scheduled controls for this round ---------------------------
        lr_k = M.lr_at(c["lr_schedule"], float(c["lr_init"]), it, K,
                       gamma=float(c["lr_gamma"]), lr_min=float(c["lr_min"]),
                       step=int(c["lr_step"]))
        area_k = float(ext_g["area_frac"]) if ext_g else 1.0
        mix_k = M.mix_at(c["mix_schedule"], float(c["mix_ratio"]), it, K,
                         rate=float(c["mix_rate"]), severity=sev, severity0=sev0,
                         size=area_k)
        sig_k = M.sigma_at(c["sigma_schedule"], float(c["sel_sigma"]), it, K,
                           rate=float(c["sigma_rate"]), severity=sev, severity0=sev0,
                           size=area_k)
        sched["lr"].append(lr_k); sched["mix"].append(mix_k)
        sched["sigma"].append(sig_k); sched["severity"].append(sev)

        # ---- one shared candidate pool; both strategies draw from it -----
        if fixed_pool:
            ig = np.flatnonzero(avail["g"])
            ir = np.flatnonzero(avail["r"])
            Xc_g, yc_g = X_pool[ig], y_pool[ig]
            Xc_r, yc_r = X_pool[ir], y_pool[ir]
        else:
            Xc_g = _pool_inputs(int(c["n_candidate"]))
            yc_g = P.label(Xc_g, rng, n_bumps=nb, noise_std=noise)
            Xc_r, yc_r = Xc_g, yc_g          # same pool → a tight head-to-head

        n_sel = int(c["n_select"])
        if bool(c.get("gaussian_kernel", True)):
            idx_g, w_g = P.select_by_weakspot(
                Xc_g, c_g, sig_k, min(n_sel, len(Xc_g)), rng, mode=c["sel_mode"],
                method=c["sel_method"], ext=ext_g, surf=surf_g, mix_ratio=mix_k)
        else:
            idx_g, w_g = _select_hard(Xc_g, c_g, ext_g, sig_k, mix_k,
                                      min(n_sel, len(Xc_g)), rng)
        Xsel_g, ysel_g = Xc_g[idx_g], yc_g[idx_g]

        take_r = min(n_sel, len(Xc_r))
        idx_r = (rng.choice(len(Xc_r), size=take_r, replace=False)
                 if take_r > 0 else np.array([], dtype=int))
        Xsel_r, ysel_r = Xc_r[idx_r], yc_r[idx_r]

        if fixed_pool:
            avail["g"][ig[idx_g]] = False
            avail["r"][ir[idx_r]] = False

        # ---- train every active track ------------------------------------
        Xg = np.vstack([Xg, Xsel_g]); yg = np.concatenate([yg, ysel_g])
        Xr = np.vstack([Xr, Xsel_r]); yr = np.concatenate([yr, ysel_r])
        ep = int(c["iters_retrain"])
        fitted: dict = {}
        if "gacc" in active:
            fitted["gacc"] = _fit_eval(Xg, yg, "gacc", lr_k, ep)
            fitted["racc"] = _fit_eval(Xr, yr, "racc", lr_k, ep)
        if "gnew" in active:
            if len(Xsel_g) and len(Xsel_r):
                fitted["gnew"] = _fit_eval(Xsel_g, ysel_g, "gnew", lr_k, ep)
                fitted["rnew"] = _fit_eval(Xsel_r, ysel_r, "rnew", lr_k, ep)
            else:
                _carry("gnew"); _carry("rnew")
        if "gsca" in active:
            if len(Xsel_g) and len(Xsel_r):
                # Same point BUDGET as new-only, drawn from the accumulated pool
                # so coverage is retained — isolates distribution from raw count.
                bg = rng_aux.choice(len(Xg), size=min(len(Xsel_g), len(Xg)), replace=False)
                fitted["gsca"] = _fit_eval(Xg[bg], yg[bg], "gsca", lr_k, ep)
                br = rng_aux.choice(len(Xr), size=min(len(Xsel_r), len(Xr)), replace=False)
                fitted["rsca"] = _fit_eval(Xr[br], yr[br], "rsca", lr_k, ep)
            else:
                _carry("gsca"); _carry("rsca")
        # The driver advances to the model just trained; if its round was carried
        # (an exhausted pool left nothing to select) it keeps the previous weights.
        mg = fitted.get(gkey, mg)

        if keep_rounds:
            rounds.append(dict(
                c_g=c_g, ext_g=ext_g, surf_g=surf_g, err=err_cur,
                Xc_g=Xc_g, w_g=w_g, Xsel_g=Xsel_g, Xc_r=Xc_r, Xsel_r=Xsel_r,
                n_train_acc=len(Xg), lr=lr_k, mix=mix_k, sigma=sig_k, severity=sev,
            ))
        if progress is not None:
            progress(it + 1, K, f"Iteration {it + 1}/{K}")

    # ── budget-matched single-shot control ───────────────────
    # The iterative question the companion paper could not ask: is K rounds of
    # n_select points better than ONE round of K × n_select points? Compute is
    # matched too (K × iters_retrain epochs), so only the *staging* differs.
    single = None
    if bool(c["single_shot"]) and K > 0:
        tot = int(c["n_select"]) * K
        Xc = P.sample_inputs(max(int(c["n_candidate"]), tot), rng_ss, shift_strength=0.0)
        yc = P.label(Xc, rng_ss, n_bumps=nb, noise_std=noise)
        surf_s, c_s, ext_s = _detect(err0)
        i_g, _ = P.select_by_weakspot(
            Xc, c_s, float(c["sel_sigma"]), min(tot, len(Xc)), rng_ss,
            mode=c["sel_mode"], method=c["sel_method"], ext=ext_s, surf=surf_s,
            mix_ratio=float(c["mix_ratio"]))
        i_r = rng_ss.choice(len(Xc), size=min(tot, len(Xc)), replace=False)
        epochs = int(c["iters_retrain"]) * K
        single = {}
        for tag, idx in (("guided", i_g), ("random", i_r)):
            m = copy.deepcopy(m0) if warm else _new_model(epochs, float(c["lr_init"]), False)
            if warm:
                M.apply_runtime(m, lr=float(c["lr_init"]), epochs=epochs)
            # Match the primary regime's protocol, or the control is not matched:
            # under new-only the loop never revisits the original set, so neither
            # may the single shot.
            if primary == "new-only":
                X_s, y_s = Xc[idx], yc[idx]
            else:
                X_s = np.vstack([X_tr0, Xc[idx]])
                y_s = np.concatenate([y_tr0, yc[idx]])
            m.fit(X_s, y_s)
            _, e_s, met_s = P.evaluate(m, X_eval, y_eval)
            ei, eo = _region(e_s)
            single[tag] = dict(mae=met_s["MAE"], rmse=met_s["RMSE"], r2=met_s["R2"],
                               err_in=ei, err_out=eo, n_train=len(X_s))

    return dict(
        iters=list(range(K + 1)), tracks=hist, active=active, sched=sched, det=det,
        rounds=rounds, single_shot=single, cfg=c,
        primary=primary, driver=(gkey, rkey),
        setup=dict(center=center, radius=r_eff, xx=xx, yy=yy, grid_flat=grid_flat,
                   X_eval=X_eval, err0=err0, X_tr0=X_tr0, y_tr0=y_tr0,
                   X_keep=X_keep, X_excl=X_excl,
                   true_grid=P.landscape_surface(grid_flat, n_bumps=nb),
                   warm=warm, n_params=M.n_parameters(c["arch"], float(c["complexity"]))),
    )


# ─────────────────────────────────────────────────────────────
# Derived summaries (shared by the page and the sweep)
# ─────────────────────────────────────────────────────────────
def summarise(res: dict) -> dict:
    """Headline numbers for one run, reported against the **primary** regime.

    ``auc_gap`` is the mean guided − random MAE over iterations 1…K: a single
    number for "did guidance help *across the loop*", which the final-iteration
    value alone can misrepresent when the curves cross. ``forgetting`` is the
    change in error **outside** the induced weakspot from iteration 0 — the
    direct, iterative measurement of the collapse the single-round study could
    only infer.
    """
    H = res["tracks"]
    gkey, rkey = res.get("driver", ("gacc", "racc"))
    out: dict = {"primary_regime": res.get("primary", "accumulative")}
    if gkey in H and rkey in H:
        g = np.asarray(H[gkey]["mae"], dtype=float)
        r = np.asarray(H[rkey]["mae"], dtype=float)
        out["final_gap_mae"] = float(g[-1] - r[-1])
        out["auc_gap_mae"] = float(np.mean(g[1:] - r[1:])) if len(g) > 1 else float("nan")
        out["win_rate_iters"] = float(np.mean(g[1:] < r[1:])) if len(g) > 1 else float("nan")
        gi = np.asarray(H[gkey]["err_in"], dtype=float)
        ri = np.asarray(H[rkey]["err_in"], dtype=float)
        out["final_gap_err_in"] = float(gi[-1] - ri[-1])
        go_ = np.asarray(H[gkey]["err_out"], dtype=float)
        out["forgetting_guided"] = float(go_[-1] - go_[0])
        ro_ = np.asarray(H[rkey]["err_out"], dtype=float)
        out["forgetting_random"] = float(ro_[-1] - ro_[0])
        out["best_iter_guided"] = int(np.nanargmin(g))
    if res.get("single_shot"):
        ss = res["single_shot"]
        out["single_shot_guided_mae"] = ss["guided"]["mae"]
        out["single_shot_random_mae"] = ss["random"]["mae"]
        if gkey in H:
            out["staging_gain_guided"] = float(ss["guided"]["mae"] - H[gkey]["mae"][-1])
    return out
