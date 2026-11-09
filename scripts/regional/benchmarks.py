"""
Benchmark problems: a frozen MLP surrogate of a published engineering simulator.

The simulators are the standard test functions of computer experiments, in the definitions
of the Virtual Library of Simulation Experiments (Surjanovic and Bingham, sfu.ca/~ssurjano).
Each is cheap, so that any input can be queried and the truth can be computed by brute force,
and nothing about the failing set is planted: it is whatever the trained model gets wrong.

Q is uniform on the published input ranges. Two training conditions per simulator:

    iid     training inputs from Q, the usual split: test cases align with training data
    shift   training inputs concentrated around a location drawn per seed

A failure is an absolute error above the 95th percentile of the model's error under Q, so
that 5% of Q fails in every benchmark and only the shape of the failing set differs.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from scripts.regional.problems import Problem


def borehole(z):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = z.T
    log = np.log(r / rw)
    return 2 * np.pi * Tu * (Hu - Hl) / (log * (1 + 2 * L * Tu / (log * rw ** 2 * Kw) + Tu / Tl))


def otl_circuit(z):
    Rb1, Rb2, Rf, Rc1, Rc2, beta = z.T
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    b = beta * (Rc2 + 9)
    return (Vb1 + 0.74) * b / (b + Rf) + 11.35 * Rf / (b + Rf) + 0.74 * Rf * b / ((b + Rf) * Rc1)


def piston(z):
    M, S, V0, k, P0, Ta, T0 = z.T
    A = P0 * S + 19.62 * M - k * V0 / S
    V = S / (2 * k) * (np.sqrt(A ** 2 + 4 * k * (P0 * V0 / T0) * Ta) - A)
    return 2 * np.pi * np.sqrt(M / (k + S ** 2 * (P0 * V0 / T0) * (Ta / V ** 2)))


def robot_arm(z):
    angles = np.cumsum(z[:, :4], axis=1)
    L = z[:, 4:]
    return np.hypot((L * np.cos(angles)).sum(1), (L * np.sin(angles)).sum(1))


def wing_weight(z):
    Sw, Wfw, A, sweep, q, lam, tc, Nz, Wdg, Wp = z.T
    c = np.cos(np.deg2rad(sweep))
    return (0.036 * Sw ** 0.758 * Wfw ** 0.0035 * (A / c ** 2) ** 0.6 * q ** 0.006 * lam ** 0.04
            * (100 * tc / c) ** -0.3 * (Nz * Wdg) ** 0.49 + Sw * Wp)


SIMULATORS = {
    "borehole": (borehole, [(0.05, 0.15), (100, 50000), (63070, 115600), (990, 1110),
                            (63.1, 116), (700, 820), (1120, 1680), (9855, 12045)]),
    "otl": (otl_circuit, [(50, 150), (25, 70), (0.5, 3), (1.2, 2.5), (0.25, 1.2), (50, 300)]),
    "piston": (piston, [(30, 60), (0.005, 0.020), (0.002, 0.010), (1000, 5000), (90000, 110000),
                        (290, 296), (340, 360)]),
    "robot": (robot_arm, [(0, 2 * np.pi)] * 4 + [(0, 1)] * 4),
    "wing": (wing_weight, [(150, 200), (220, 300), (6, 10), (-10, 10), (16, 45), (0.5, 1),
                           (0.08, 0.18), (2.5, 6), (1700, 2500), (0.025, 0.08)]),
}
BENCHMARKS = tuple(f"{name}_{cond}" for name in SIMULATORS for cond in ("iid", "shift"))


class Surrogate(Problem):
    """Inputs of the strategies live on the unit cube and are mapped to the published ranges."""

    def __init__(self, regime, seed, n_train=2000, failing=0.05):
        name, cond = regime.rsplit("_", 1)
        self.simulator, ranges = SIMULATORS[name]
        self.low, self.high = np.array(ranges, float).T
        self.d = len(ranges)
        rs = np.random.RandomState(seed)
        X = rs.uniform(0, 1, size=(n_train, self.d))
        if cond == "shift":
            X = np.clip(rs.uniform(0.2, 0.8, self.d) + 0.2 * rs.normal(size=(n_train, self.d)), 0, 1)
        y = self.target(X)
        self.mean_y, self.std_y = y.mean(), y.std()
        self.X_train = X
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=seed)
        self.model.fit(X, (y - self.mean_y) / self.std_y)
        self.t = float(np.quantile(self._error(self.sample(20000, rs)), 1 - failing))

    def target(self, X):
        return self.simulator(self.low + X * (self.high - self.low))

    def _error(self, X):
        return np.abs(self.target(X) - (self.model.predict(X) * self.std_y + self.mean_y))

    def mean_failure(self, X):
        return (self._error(X) > self.t).astype(float)

    def failure(self, X, rs):
        return self.mean_failure(X)
