"""Model architectures, optimiser settings and per-iteration schedules.

The single-round study held the regression model fixed: one MLP of one shape,
sklearn's default optimiser settings, one retraining budget. In a *repeated*
loop those choices stop being incidental. Each round continues training the same
weights on a training distribution that has just been shifted towards the weak
region, so the architecture (how much capacity is available to absorb the new
region without overwriting the old one) and the learning rate (how far the
weights are allowed to move per round) become first-class controls of the
forgetting/plasticity balance the companion paper identified as the binding
constraint.

This module therefore exposes three things:

``ARCHITECTURES``    named hidden-layer shapes, from a tiny single layer to deep
                     and bottleneck variants, plus the legacy complexity-scaled
                     ``(h, h)`` shape so previous results stay reproducible.
``lr_at``            the per-iteration learning rate under a chosen schedule
                     (constant, decays, cosine, warm restarts) — the analogue of
                     an epoch-wise LR schedule, one step per *loop iteration*.
``mix_at``/``sigma_at``  schedules for the two coverage controls the paper found
                     dominant (rehearsal mix α and kernel width σ), including an
                     *adaptive* variant that relaxes the focus as the detected
                     weakspot heals — the extension named in its future work.

Everything here is plain sklearn and numpy so the sweep can pickle it into
worker processes.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.weakspot.models import AVAILABLE_MODELS, build_model  # noqa: F401

# ─────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────
# ``None`` = derive the shape from the ``complexity`` slider exactly as
# ``scripts.weakspot.models.build_model`` does, so the legacy default is
# bit-for-bit reproducible and single-round results remain comparable.
LEGACY_ARCH = "Complexity-scaled (2 × h)"

ARCHITECTURES: dict[str, tuple[int, ...] | None] = {
    LEGACY_ARCH:              None,
    "Tiny (1 × 16)":          (16,),
    "Small (1 × 64)":         (64,),
    "Standard (2 × 64)":      (64, 64),
    "Wide (2 × 256)":         (256, 256),
    "Deep (4 × 64)":          (64, 64, 64, 64),
    "Deep narrow (6 × 32)":   (32, 32, 32, 32, 32, 32),
    "Pyramid (128-64-32)":    (128, 64, 32),
    "Bottleneck (128-16-128)": (128, 16, 128),
}

ACTIVATIONS = ("relu", "tanh", "logistic")
SOLVERS = ("adam", "sgd")


def hidden_sizes(arch: str, complexity: float) -> tuple[int, ...]:
    """Hidden-layer shape for a named architecture.

    The legacy entry reproduces ``build_model``'s ``(h, h)`` with
    ``h = max(8, int(complexity * 128))``.
    """
    shape = ARCHITECTURES.get(arch, None)
    if shape is not None:
        return tuple(shape)
    h = max(8, int(float(complexity) * 128))
    return (h, h)


def n_parameters(arch: str, complexity: float, d_in: int = 2) -> int:
    """Weight+bias count for a shape — a cheap capacity axis for the analysis."""
    sizes = (d_in,) + hidden_sizes(arch, complexity) + (1,)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def build_iter_model(model_key: str, *, arch: str = LEGACY_ARCH,
                     complexity: float = 0.5, iterations: int = 100,
                     warm_start: bool = False, early_stopping: bool = True,
                     lr_init: float = 1e-3, solver: str = "adam",
                     activation: str = "relu", alpha: float = 1e-4,
                     batch_size: int | str = "auto",
                     random_state: int = 42):
    """Build the regression model for one iterative run.

    For every non-MLP algorithm this delegates unchanged to
    ``scripts.weakspot.models.build_model`` (those have no warm-start path, so
    each iteration rebuilds them from scratch). For the MLP it builds the same
    ``StandardScaler → MLPRegressor`` pipeline but exposes the architecture and
    optimiser settings that the iterative study varies.

    With ``arch=LEGACY_ARCH``, ``lr_init=1e-3``, ``solver="adam"``,
    ``activation="relu"``, ``alpha=1e-4`` and ``batch_size="auto"`` the result is
    identical to ``build_model``'s MLP, so the paper's operating point is
    reproduced exactly.
    """
    if model_key != "mlp":
        return build_model(model_key, complexity=complexity, iterations=iterations,
                           random_state=random_state, warm_start=warm_start,
                           early_stopping=early_stopping)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden_sizes(arch, complexity),
            activation=activation, solver=solver,
            alpha=float(alpha), batch_size=batch_size,
            learning_rate_init=float(lr_init),
            max_iter=int(iterations), random_state=int(random_state),
            early_stopping=bool(early_stopping), warm_start=bool(warm_start),
        )),
    ])


class FrozenFirstLayer:
    """The initial model's scaler and first hidden layer as a fixed transform.

    sklearn's MLP cannot freeze layers during backpropagation, but freezing the
    *first* hidden layer is expressible exactly: the frozen layer becomes a
    fixed feature map ``phi(x) = act(W1 . scale(x) + b1)`` taken from the
    fitted initial model, and the remaining layers become a smaller MLP trained
    on ``phi(x)``. Training that head is mathematically identical to training
    the full network with the first layer's gradients zeroed.
    """

    def __init__(self, mean, scale, W, b, activation="relu"):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.W, self.b = np.asarray(W), np.asarray(b)
        self.activation = activation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        h = ((np.asarray(X, dtype=float) - self.mean) / self.scale) @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(h, 0.0)
        if self.activation == "tanh":
            return np.tanh(h)
        if self.activation == "logistic":
            return 1.0 / (1.0 + np.exp(-h))
        return h                                            # "identity"

    def fit_transform(self, X, y=None):
        return self.transform(X)

    # Enough of the sklearn API for Pipeline; cloning is never needed here.
    def get_params(self, deep=True):
        return {"mean": self.mean, "scale": self.scale, "W": self.W,
                "b": self.b, "activation": self.activation}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


def make_frozen_head(m0, X_init, y_init):
    """A track model that continues ``m0`` with its first hidden layer frozen.

    Returns ``Pipeline([("frozen", phi), ("model", head)])`` where ``phi`` is
    the fitted initial model's scaler plus first layer and ``head`` is an MLP
    holding the remaining layers' weights, warm-started so that every later
    ``fit`` continues them. Requires at least two hidden layers (with one, the
    head would have none, which sklearn's MLP cannot represent). The helpers
    in this module (``apply_runtime``, ``snapshot_weights`` …) address the
    pipeline's ``"model"`` step, so the returned object drops into the loop
    unchanged.
    """
    scaler = m0.named_steps["scaler"]
    mlp0 = m0.named_steps["model"]
    if len(mlp0.coefs_) < 3:
        raise ValueError("freeze_first needs at least two hidden layers")
    phi = FrozenFirstLayer(scaler.mean_, scaler.scale_,
                           mlp0.coefs_[0], mlp0.intercepts_[0],
                           activation=mlp0.activation)
    head = MLPRegressor(
        hidden_layer_sizes=tuple(c.shape[1] for c in mlp0.coefs_[1:-1]),
        activation=mlp0.activation, solver=mlp0.solver, alpha=mlp0.alpha,
        batch_size=mlp0.batch_size, learning_rate_init=mlp0.learning_rate_init,
        max_iter=1, random_state=mlp0.random_state,
        early_stopping=mlp0.early_stopping, warm_start=True)
    # One throwaway epoch initialises sklearn's internal structures at the right
    # shapes; the weights are then overwritten with the initial model's upper
    # layers, so the head *continues* m0 rather than restarting.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head.fit(phi.transform(X_init), y_init)
    head.coefs_ = [c.copy() for c in mlp0.coefs_[1:]]
    head.intercepts_ = [b.copy() for b in mlp0.intercepts_[1:]]
    return Pipeline([("frozen", phi), ("model", head)])


def reset_stopping_state(est) -> None:
    """Make a warm-started estimator's stopping rule *self-contained per round*.

    sklearn only calls ``_initialize`` on the first ``fit``, so under
    ``warm_start`` the convergence bookkeeping — ``_no_improvement_count``,
    ``best_validation_score_``/``best_loss_`` and the retained ``_best_coefs`` —
    survives into every later ``fit``. In a loop that is silently fatal: the
    counter keeps climbing across rounds until it passes ``n_iter_no_change``,
    after which every round terminates after a single epoch, and with early
    stopping on, each fit ends by restoring weights selected against a *previous*
    round's validation split. The model stops learning after round one while the
    configured epoch budget goes unspent.

    Clearing the state here makes each round's stopping decision depend only on
    that round, which is the only interpretation under which "continue training
    for ``iters_retrain`` epochs" means what it says.
    """
    if not hasattr(est, "_no_improvement_count"):
        return                                   # never fitted — nothing to clear
    est._no_improvement_count = 0
    if getattr(est, "early_stopping", False):
        est.best_validation_score_ = -np.inf
        est.validation_scores_ = []
    else:
        # The same trap exists with early stopping off: there the counter is
        # driven by ``best_loss_``, which persists just as stubbornly.
        est.best_loss_ = np.inf


def snapshot_weights(model):
    """Copy an MLP's weights, or ``None`` if it has not been fitted yet."""
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(est, "coefs_"):
        return None
    return ([c.copy() for c in est.coefs_], [b.copy() for b in est.intercepts_])


def blend_weights(model, snapshot, beta: float) -> None:
    """Pull a just-trained model back towards ``snapshot``: ``β·new + (1−β)·old``.

    Round-level Polyak averaging, and the most direct answer to a bumpy loop.
    Each round fits on a fresh handful of points, so the round's own optimum is a
    high-variance estimate and taking it whole makes the trajectory lurch from
    sample to sample. Damping the *step between rounds* — rather than the step
    between epochs, which the learning rate already governs — averages that
    variance away while still letting the model move where successive rounds
    agree. ``β=1`` keeps the round's model unchanged (the default, no damping).
    """
    if snapshot is None or beta >= 1.0:
        return
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    b = float(np.clip(beta, 0.0, 1.0))
    for i, old in enumerate(snapshot[0]):
        est.coefs_[i] = b * est.coefs_[i] + (1.0 - b) * old
    for i, old in enumerate(snapshot[1]):
        est.intercepts_[i] = b * est.intercepts_[i] + (1.0 - b) * old


def apply_runtime(model, *, lr: float | None = None, epochs: int | None = None,
                  reset_stopping: bool = True) -> None:
    """Prepare an already-built model for another round of continued training.

    Used between loop iterations: sklearn rebuilds its optimiser on every
    ``fit`` call (``fit`` is not incremental even under ``warm_start``), so
    writing ``learning_rate_init`` here is what makes a per-iteration learning-
    rate schedule take effect on the *continued* weights. ``reset_stopping``
    additionally clears the cross-round convergence state — see
    :func:`reset_stopping_state`. Silently does nothing for algorithms that have
    no such knobs.
    """
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if epochs is not None and hasattr(est, "max_iter"):
        est.max_iter = int(epochs)
    if lr is not None and hasattr(est, "learning_rate_init"):
        est.learning_rate_init = float(lr)
    if reset_stopping:
        reset_stopping_state(est)


# ─────────────────────────────────────────────────────────────
# Per-iteration learning-rate schedule
# ─────────────────────────────────────────────────────────────
# One step per *loop iteration*, not per epoch: the question is how far the
# weights may travel in each round of continued training. A decaying rate lets
# the early rounds absorb the weak region and the late rounds consolidate, which
# is the natural counter to the drift the single-round study penalised.
LR_SCHEDULES = (
    "Constant",
    "Exponential decay",
    "Step decay",
    "Cosine anneal",
    "Warm restarts (SGDR)",
    "Inverse-time decay",
)


def lr_at(schedule: str, lr0: float, k: int, n_iterations: int,
          gamma: float = 0.7, lr_min: float = 1e-5, step: int = 2) -> float:
    """Learning rate for loop iteration ``k`` (0-based).

    ``gamma``  per-step multiplier (exponential/step) or decay rate (inverse).
    ``lr_min`` floor for the cosine schedules.
    ``step``   iterations per step (step decay) or per cycle (warm restarts).
    """
    lr0, k = float(lr0), max(int(k), 0)
    K = max(int(n_iterations) - 1, 1)
    if schedule == "Exponential decay":
        return max(lr0 * float(gamma) ** k, 0.0)
    if schedule == "Step decay":
        return max(lr0 * float(gamma) ** (k // max(int(step), 1)), 0.0)
    if schedule == "Cosine anneal":
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * k / K))
    if schedule == "Warm restarts (SGDR)":
        t = k % max(int(step), 1)
        T = max(int(step) - 1, 1)
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * t / T))
    if schedule == "Inverse-time decay":
        return lr0 / (1.0 + float(gamma) * k)
    return lr0                                              # "Constant"


# ─────────────────────────────────────────────────────────────
# Per-iteration coverage schedules (the paper's two dominant controls)
# ─────────────────────────────────────────────────────────────
MIX_SCHEDULES = (
    "Constant",
    "Linear decay",
    "Exponential decay",
    "Linear increase",
    "Adaptive (weakspot severity)",
    "Adaptive (weakspot size)",
)
SIGMA_SCHEDULES = (
    "Constant",
    "Linear widen",
    "Exponential widen",
    "Linear narrow",
    "Adaptive (weakspot severity)",
    "Adaptive (weakspot size)",
)

# Severity below this multiple of the ambient error counts as "healed": the
# detected region no longer stands out from the rest of the error surface, so an
# adaptive schedule hands the budget back to uniform rehearsal.
_HEALED = 1.0


def _relative_excess(severity: float, severity0: float) -> float:
    """How much of the *initial* weakspot severity is left, in [0, 1].

    ``severity`` is the model's mean error inside the detected region divided by
    its mean error outside — computed from the error distribution alone, with no
    access to the ground-truth gap, so an adaptive schedule stays implementable
    in the fixed-dataset setting the study targets.
    """
    e0 = max(float(severity0) - _HEALED, 1e-9)
    e = max(float(severity) - _HEALED, 0.0)
    return float(min(max(e / e0, 0.0), 1.0))


def _size_focus(area: float, compact: float = 0.15) -> float:
    """1 when the detected region is compact, 0 when it covers the square.

    The size-driven gate: a genuinely localised weakspot puts a small, tight
    region above the extraction threshold, whereas a model that is uniformly
    mediocre produces a near-flat error surface whose thresholded region sprawls
    across most of the input space. ``compact`` is the area fraction treated as
    fully localised — at the default extraction quantile 0.85 a single clean blob
    occupies roughly 0.15 of the grid.

    Measured caveat: on this task the signal moves once (0.95 at round 1, ~0.44
    thereafter) and then carries no trend, because round 1's model is bad
    *everywhere* — the gap is real but not yet distinguishable by contrast. The
    severity gate discriminates far better here. Kept because the two answer
    different questions: size asks "is the weak region localised", severity asks
    "is it actually worse than the rest".
    """
    a = float(min(max(area, 0.0), 1.0))
    return float(min(max((1.0 - a) / max(1.0 - compact, 1e-9), 0.0), 1.0))


def mix_at(schedule: str, a0: float, k: int, n_iterations: int, rate: float = 0.5,
           severity: float = 1.0, severity0: float = 1.0,
           size: float = 1.0) -> float:
    """Rehearsal mix α for loop iteration ``k`` (0-based), clipped to [0, 1].

    ``rate`` is the total fraction of α given up by the final iteration for the
    linear schedules and the per-iteration multiplier for the exponential one.
    The adaptive schedule scales α by how much of the initial weakspot severity
    survives, so focus relaxes towards pure rehearsal as the region heals.
    """
    a0, k = float(a0), max(int(k), 0)
    K = max(int(n_iterations) - 1, 1)
    if schedule == "Linear decay":
        v = a0 * (1.0 - float(rate) * k / K)
    elif schedule == "Exponential decay":
        v = a0 * float(rate) ** k
    elif schedule == "Linear increase":
        v = a0 + (1.0 - a0) * float(rate) * k / K
    elif schedule == "Adaptive (weakspot severity)":
        v = a0 * _relative_excess(severity, severity0)
    elif schedule == "Adaptive (weakspot size)":
        v = a0 * _size_focus(size)
    else:
        v = a0                                              # "Constant"
    return float(min(max(v, 0.0), 1.0))


def sigma_at(schedule: str, s0: float, k: int, n_iterations: int, rate: float = 0.5,
             severity: float = 1.0, severity0: float = 1.0,
             size: float = 1.0) -> float:
    """Selection-kernel width σ for loop iteration ``k`` (0-based).

    ``rate`` is the total relative widening/narrowing reached at the final
    iteration (linear) or the per-iteration growth factor − 1 (exponential).
    The adaptive schedule widens the kernel as the weakspot heals, which is the
    kernel-side expression of the same "relax the focus" idea as ``mix_at``.
    """
    s0, k = float(s0), max(int(k), 0)
    K = max(int(n_iterations) - 1, 1)
    if schedule == "Linear widen":
        v = s0 * (1.0 + float(rate) * k / K)
    elif schedule == "Exponential widen":
        v = s0 * (1.0 + float(rate)) ** k
    elif schedule == "Linear narrow":
        v = s0 * (1.0 - float(rate) * k / K)
    elif schedule == "Adaptive (weakspot severity)":
        v = s0 * (1.0 + float(rate) * (1.0 - _relative_excess(severity, severity0)))
    elif schedule == "Adaptive (weakspot size)":
        # No localised region -> widen the kernel towards uniform selection.
        v = s0 * (1.0 + float(rate) * (1.0 - _size_focus(size)))
    else:
        v = s0                                              # "Constant"
    return float(max(v, 0.01))
