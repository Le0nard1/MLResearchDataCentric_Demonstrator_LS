"""
Ground-truth error landscapes: no trained model, the error landscape itself is given.

This isolates the trade-off of a test campaign. Each landscape is a standard test function of
optimisation, level set estimation and reliability analysis, read as a field of failure rates,
so that the truth is known everywhere and a test set can be judged against it. As in BBOB, an
instance is a random transformation per seed, so that no strategy profits from a fixed position.

Three structures per function:

    low     two active coordinates, chosen per seed: axis-aligned, as in the planted problems
    full    all coordinates active
    rot     two active directions of random orientation: a region no axis-aligned box fits

The field is ranked under Q and mapped to a rate ``rank ** 19``, so that the mean failure rate
is 5% and about 11% of Q fails above tau = 0.1, in every landscape and dimension. A query
returns a Bernoulli outcome of that rate.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kstest, spearmanr

from scripts.regional.campaign import interpolate
from scripts.regional.problems import Problem

DOMAINS = dict(ackley=5.0, rastrigin=5.12, rosenbrock=2.0, herbie=2.0)
CELLS, ENOUGH = 100, 36                # 36 failure-free tests clear a region at tau = 0.1


def field(name, z):
    """Higher means worse; each function is negated so that its optimum is the failing peak."""
    if name == "ackley":
        return (20 * np.exp(-0.2 * np.sqrt((z ** 2).mean(1))) + np.exp(np.cos(2 * np.pi * z).mean(1)))
    if name == "rastrigin":
        return -(z ** 2 - 10 * np.cos(2 * np.pi * z)).sum(1)
    if name == "rosenbrock":
        return -np.log1p((100 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1 - z[:, :-1]) ** 2).sum(1))
    if name == "herbie":
        w = np.exp(-(z - 1) ** 2) + np.exp(-0.8 * (z + 1) ** 2) - 0.05 * np.sin(8 * (z + 0.1))
        return np.prod(w, axis=1)
    raise ValueError(name)


LANDSCAPES = tuple(f"{name}_{s}" for name in DOMAINS for s in ("low", "full", "rot"))


class Landscape(Problem):

    def __init__(self, regime, d, seed):
        self.name, structure = regime.rsplit("_", 1)
        self.d = d
        rs = np.random.RandomState(seed)
        if structure == "low":
            self.A = np.eye(d)[rs.permutation(d)[:2]]
        elif structure == "rot":
            self.A = np.linalg.qr(rs.normal(size=(d, d)))[0][:2]
        else:
            self.A = np.eye(d)
        self.shift = rs.uniform(-0.25, 0.25, len(self.A))
        self.ranks = np.sort(self._raw(self.sample(100000, rs)))

    def _raw(self, X):
        scale = DOMAINS[self.name]
        return field(self.name, 2 * scale * ((X - 0.5) @ self.A.T + self.shift))

    def mean_failure(self, X):
        return (np.searchsorted(self.ranks, self._raw(X)) / len(self.ranks)) ** 19

    def outcome(self, X, rs):
        """A latent margin s = Phi^-1(f) + N(0, 1): it exceeds 0 with probability f, so that the
        failure has the same law as a Bernoulli query, and a near miss is visible. The scalar
        is reported as Phi(s) in [0, 1]."""
        from scipy.stats import norm
        f = np.clip(self.mean_failure(X), 1e-9, 1 - 1e-9)
        s = norm.ppf(f) + rs.normal(size=len(X))
        return norm.cdf(s), (s > 0).astype(float)


def test_set(problem, X, e, tau, seed, n_eval=20000, k=25):
    """How good the test set is against the truth, for completeness, representativeness and
    thoroughness, and how well a landscape estimated from it matches the true one. Cells are the
    Voronoi cells of ``CELLS`` points from Q, which stay comparable in any dimension."""
    rs = np.random.RandomState(seed + 2000003)
    Z = problem.sample(n_eval, rs)
    f, bad = problem.mean_failure(Z), None
    bad = f > tau
    tree = cKDTree(problem.sample(CELLS, rs))
    tests = np.bincount(tree.query(X)[1], minlength=CELLS)[tree.query(Z)[1]]
    land = interpolate(X, e, Z, min(k, len(X)))
    top = land >= np.quantile(land, 1 - bad.mean())
    return dict(reach=float((tests > 0).mean()), reach_bad=float((tests[bad] > 0).mean()),
                enough=float((tests >= ENOUGH).mean()), enough_bad=float((tests[bad] >= ENOUGH).mean()),
                tests_bad=float((problem.mean_failure(X) > tau).mean()),
                marginal_ks=float(np.mean([kstest(X[:, j], "uniform").statistic for j in range(X.shape[1])])),
                risk_gap=float(abs(e.mean() - f.mean())),
                land_spearman=float(spearmanr(land, f).statistic),
                land_overlap=float((top & bad).sum() / max(bad.sum(), 1)))
