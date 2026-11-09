"""
Test problems: a frozen system that can be queried at any chosen input.

A problem offers three things:

    sample(n, rs)        n inputs from the operational distribution Q (free of charge)
    failure(X, rs)       observed failure scores in [0, 1]; each row costs one query
    mean_failure(X)      true mean failure, for scoring only, never seen by a strategy

``planted`` problems define the failure function directly, so every regime of the
experiment plan can be built exactly. ``trained`` problems freeze a real MLP trained on
an uneven data distribution, which gives the three density regimes. Q is uniform on the
unit cube throughout; the location of a region is drawn per seed, so that no strategy can
profit from a fixed position (see the degenerate-centre artefact of the earlier papers).
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from scripts.dataselect.baselines import pick_kcenter  # noqa: F401  (re-exported for strategies)

PLANTED = ("no_failure", "flat","hard", "multi", "needle", "thin")
TRAINED = ("hard_dense", "data_hole", "sparse_easy")


class Problem:
    d = 2
    X_train = None                 # training inputs, for density guidance; None if unknown

    def sample(self, n, rs):
        return rs.uniform(0, 1, size=(n, self.d))

    def mean_failure(self, X):
        raise NotImplementedError

    def failure(self, X, rs):
        return (rs.uniform(size=len(X)) < self.mean_failure(X)).astype(float)

    def outcome(self, X, rs):
        """A scalar outcome scaled to [0, 1] relative to the tolerance, and the failure it
        implies. Problems without a scalar return the failure for both."""
        fail = self.failure(X, rs)
        return fail, fail

    def truth(self):
        """Centre and radius of the true region in the first two coordinates, or None."""
        return None


class Planted(Problem):
    """Failure rate ``base`` with discs of rate ``rate`` in the first two coordinates, or a
    band around a hyperplane through all coordinates (``band > 0``)."""

    def __init__(self, d, base, discs=(), rate=0.5, band=0.0, normal=None, offset=0.0):
        self.d, self.base, self.discs, self.rate = d, base, discs, rate
        self.band, self.normal, self.offset = band, normal, offset

    def truth(self):
        return self.discs[0] if len(self.discs) else None

    def mean_failure(self, X):
        f = np.full(len(X), float(self.base))
        for c, r in self.discs:
            f[((X[:, :2] - c) ** 2).sum(1) < r ** 2] = self.rate
        if self.band > 0:
            f[np.abs(X @ self.normal - self.offset) < self.band] = self.rate
        return f


def _centre(rs, margin):
    return rs.uniform(margin, 1 - margin, size=2)


def planted(regime, d, seed, n=1000, mass=0.05):
    """Regimes of the experiment plan. ``mass`` is the mass of the planted region; the
    needle's mass is a tenth of ``1/n``, below what any strategy can reach."""
    rs = np.random.RandomState(seed)
    disc = lambda m: float(np.sqrt(m / np.pi))
    r = disc(mass)
    if regime == "no_failure":
        return Planted(d, 0.01)
    if regime == "flat":
        return Planted(d, 0.20)
    if regime == "hard":
        return Planted(d, 0.01, [(_centre(rs, r), r)])
    if regime == "multi":
        r2 = disc(mass / 2)
        return Planted(d, 0.01, [(rs.uniform(r, 0.5 - r, 2), r),
                                 (rs.uniform(0.5 + r2, 1 - r2, 2), r2)])
    if regime == "needle":
        r3 = disc(0.1 / n)
        return Planted(d, 0.01, [(_centre(rs, r3), r3)], rate=1.0)
    if regime == "thin":
        w = rs.normal(size=d)
        w /= np.linalg.norm(w)
        return Planted(d, 0.01, band=mass / 2, normal=w, offset=float(w.sum() / 2))
    raise ValueError(regime)


class Steered(Problem):
    """Wraps a problem so that a strategy steers on the scalar outcome: ``failure`` returns the
    scaled scalar, and the failures it implies are logged in call order, so that every claim
    can still be made on the binary failures."""

    def __init__(self, base):
        self.base, self.d, self.X_train, self.log = base, base.d, base.X_train, []

    def sample(self, n, rs):
        return self.base.sample(n, rs)

    def mean_failure(self, X):
        return self.base.mean_failure(X)

    def truth(self):
        return self.base.truth()

    def failure(self, X, rs):
        score, fail = self.base.outcome(X, rs)
        self.log.append(fail)
        return score


class Trained(Problem):
    """A frozen MLP. A failure is an absolute error above ``t``, the model's 99th error
    percentile outside the disc, so that the background failure rate is 1% in every regime."""

    def __init__(self, regime, d, seed, n_train=2000, radius=0.2, noise=0.02):
        rs = np.random.RandomState(seed)
        self.d, self.regime = d, regime
        self.c, self.r = _centre(rs, radius), radius
        X = rs.uniform(0, 1, size=(n_train, d))
        if regime in ("data_hole", "sparse_easy"):
            keep = ~self._in_disc(X)
            if regime == "sparse_easy":
                keep |= rs.uniform(size=len(X)) < 0.05
            X = X[keep]
        y = self.target(X) + noise * rs.normal(size=len(X))
        self.X_train = X
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=seed).fit(X, y)
        Z = rs.uniform(0, 1, size=(5000, d))
        self.t = float(np.quantile(self._error(Z[~self._in_disc(Z)]), 0.99))

    def truth(self):
        return self.c, self.r

    def _in_disc(self, X):
        return ((X[:, :2] - self.c) ** 2).sum(1) < self.r ** 2

    def target(self, X):
        y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
        s = np.clip(1 - ((X[:, :2] - self.c) ** 2).sum(1) / self.r ** 2, 0, None)
        if self.regime in ("hard_dense", "data_hole"):
            y = y + 0.5 * s * np.sin(40 * X[:, 0]) * np.sin(40 * X[:, 1])
        return y

    def _error(self, X):
        return np.abs(self.target(X) - self.model.predict(X))

    def mean_failure(self, X):
        return (self._error(X) > self.t).astype(float)

    def failure(self, X, rs):
        return self.mean_failure(X)


class Rare(Problem):
    """A problem with every failure rate scaled by ``s`` and tau with it: the same structure,
    with events as rare as in logged operations (a mean rate of 0.5% for the landscapes)."""

    def __init__(self, base, s=0.1):
        self.base, self.s, self.d = base, s, base.d
        self.tau = getattr(base, "tau", 0.1) * s
        self.X_train = base.X_train

    def mean_failure(self, X):
        return self.s * self.base.mean_failure(X)


class FourBranch(Problem):
    """The four-branch series system of reliability analysis (Schueremans and Van Gemert 2005,
    k = 6, failure probability about 0.45%) on two coordinates of Q drawn per seed, mapped to
    standard normal inputs; the other coordinates are irrelevant. A failure is g(x) < 0, so
    the outcome is deterministic and the failing set is exactly known."""

    tau = 0.01

    def __init__(self, d, seed, k=6.0):
        from scipy.stats import norm
        self.d, self.k, self.ppf = d, k, norm.ppf
        self.axes = np.random.RandomState(seed).permutation(d)[:2]

    def mean_failure(self, X):
        u = np.clip(X[:, self.axes], 1e-12, 1 - 1e-12)
        x1, x2 = self.ppf(u).T
        s, q = (x1 + x2) / np.sqrt(2), 0.1 * (x1 - x2) ** 2
        g = np.minimum.reduce([3 + q - s, 3 + q + s, (x1 - x2) + self.k / np.sqrt(2),
                               (x2 - x1) + self.k / np.sqrt(2)])
        return (g < 0).astype(float)

    def failure(self, X, rs):
        return self.mean_failure(X)


def make(regime, d, seed, n=1000):
    from scripts.regional.benchmarks import BENCHMARKS, Surrogate
    from scripts.regional.landscapes import LANDSCAPES, Landscape
    if regime.endswith("_rare"):
        return Rare(make(regime[:-5], d, seed, n))
    if regime == "four_branch":
        return FourBranch(d, seed)
    if regime in BENCHMARKS:
        return Surrogate(regime, seed)
    if regime in LANDSCAPES:
        return Landscape(regime, d, seed)
    return Trained(regime, d, seed) if regime in TRAINED else planted(regime, d, seed, n)
