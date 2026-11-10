"""
One test campaign: spend ``n`` queries on a frozen system and report what was learnt.

Every strategy shares the same three stages and differs only in the exploration:

    explore   ``n_explore`` queries, of which a share ``kappa`` is representative
    propose   one region, chosen from the exploration only
    confirm   ``n_conf`` fresh queries from Q restricted to that region

The report has two parts: a global upper bound on the failure rate from the representative
stream, and the region with an interval and a label (flagged, cleared, undetermined). The
landscape only decides where to test, so a poor landscape costs efficiency, not validity.

Strategies (``strategy`` argument of :func:`run`):

    random         all exploration from Q, region proposed afterwards
    binning        equal allocation over a fixed grid, worst cell (the classic design)
    weakspot       representative stream, then zooming into a hard region of adaptive radius
    soft           no hard region: pool candidates are drawn with weights from the landscape
    soft_optimistic  as soft, with a bonus for candidates far from earlier tests
    soft_density   as soft, with a bonus for candidates far from the training data
    peel           hybrid with binning: an axis-aligned box trimmed cut by cut, as in PRIM
    subspace       as weakspot, zooming only on the coordinates that matter
    fixed_radius   control: zooming with a fixed region mass
    top_error      control: zooming around single failures, without the landscape
    random_centre  control: zooming around random tested points
    kcenter        guided by distance to the training data and earlier tests
    sobol          baseline: representative stream, then a scrambled Sobol sequence
    straddle       baseline: level set estimation, tests where the estimate straddles tau
    oracle         upper reference: exploration as random, region around the true centre
    <s>_scalar     weakspot, peel or subspace steering on the scalar outcome; the bound, the
                   confirmation and the label still use the binary failures
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scripts.dataselect.baselines import pick_kcenter
from scripts.dataselect.budget import landscape
from scripts.regional.bounds import cp_interval, cp_upper, label

STRATEGIES = ("random", "binning", "weakspot", "fixed_radius", "top_error", "random_centre",
              "kcenter", "oracle")
VARIANTS = ("random", "binning", "weakspot", "soft", "soft_optimistic", "soft_density", "peel",
            "subspace")
BASELINES = ("sobol", "straddle")          # space filling; level set estimation
SCALAR = ("weakspot_scalar", "peel_scalar", "subspace_scalar")   # steer on the scalar outcome


@dataclass
class Ball:
    centre: np.ndarray
    radius: float
    axes: list = None               # restrict to these coordinates

    def contains(self, X):
        Z = X if self.axes is None else X[:, self.axes]
        return np.linalg.norm(Z - self.centre, axis=1) <= self.radius


@dataclass
class Box:
    low: np.ndarray
    high: np.ndarray

    def contains(self, X):
        return ((X >= self.low) & (X < self.high)).all(1)


def draw_in(problem, region, n, rs, batch=20000, max_draws=2000000):
    """``n`` draws from Q restricted to ``region`` by rejection. Returns the inputs and the
    number of draws, which is the hidden cost of a small region."""
    kept, have, draws = [], 0, 0
    while have < n and draws < max_draws:
        X = problem.sample(batch, rs)
        hits = np.flatnonzero(region.contains(X))
        draws += int(hits[n - have - 1]) + 1 if have + len(hits) >= n else batch
        kept.append(X[hits])
        have += len(hits)
    return np.vstack(kept)[:n], draws


def mass_ball(centre, pool, mass):
    """The ball around ``centre`` that holds ``mass`` of Q, measured on the free pool:
    regions are sized by mass, not volume, because Q is uneven in general."""
    d = np.linalg.norm(pool - centre, axis=1)
    return Ball(centre, float(np.quantile(d, min(mass, 1.0))))


def interpolate(X, values, Z, k):
    d, nb = cKDTree(X).query(Z, k=k)
    w = 1.0 / np.maximum(d, 1e-12)
    return (w * values[nb]).sum(1) / w.sum(1)


def elevated(land, k):
    """Which points stand out from the region's own median by more than the noise of a
    k-neighbour mean of failure scores. Without the noise term a flat region looks uneven
    and the radius collapses. The share of elevated points is high when the region is
    flat, so that nothing tighter exists, and low when the weakspot is tight."""
    med = float(np.median(land))
    return land > med + 2 * np.sqrt(max(med, 1 / k) * (1 - min(med, 0.99)) / k)


def gap(e, inside, min_failures=3):
    """z statistic of the failure rate inside a candidate against the rest."""
    a, b = e[inside], e[~inside]
    if a.sum() < min_failures or len(b) == 0:
        return -np.inf
    p = e.mean()
    se = np.sqrt(max(p * (1 - p), 1e-9) * (1 / len(a) + 1 / len(b)))
    return (a.mean() - b.mean()) / se


def stands_out(e, inside, z=3.0):
    """Whether tested points inside a candidate fail clearly more often than the others.
    The same tests chose the candidate, so this only steers the zoom (hence the high z);
    the reported interval comes from the confirmation tests."""
    return gap(e, inside) > z


def relevant_axes(X, e, z=3.0):
    """Coordinates on which failures differ from the rest in mean or spread; all of them
    when none stands out."""
    fail = e > 0.5                                       # binary failures, or a scaled scalar
    hit, rest = X[fail], X[~fail]
    if len(hit) < 5:
        return list(range(X.shape[1]))
    n = len(hit)
    mean = np.abs(hit.mean(0) - rest.mean(0)) / (rest.std(0) / np.sqrt(n))
    spread = np.abs(np.log(hit.std(0) / rest.std(0))) * np.sqrt(2 * n)
    axes = np.flatnonzero(np.maximum(mean, spread) > z)
    return axes.tolist() if len(axes) else list(range(X.shape[1]))


def propose(X, e, pool, parent, mass, rs, k=25, min_mass=0.01, mode="weakspot", fixed_mass=0.1):
    """The next region inside ``parent``, of Q-mass ``mass``. The centre comes from the
    landscape; in the adaptive mode the mass follows from the shape of the landscape,
    otherwise it is ``fixed_mass``. Returns the region and its mass."""
    inside = np.flatnonzero(parent.contains(X)) if parent is not None else np.arange(len(X))
    k = min(k, len(X) - 1)
    land = landscape(X, e, k)
    if mode == "top_error":
        fails = inside[e[inside] > 0]
        i = rs.choice(fails) if len(fails) else rs.choice(inside)
    elif mode == "random_centre":
        i = rs.choice(inside)
    else:
        i = inside[np.argmax(land[inside])]
    if mode != "weakspot":
        return mass_ball(X[i], pool, fixed_mass), fixed_mass
    in_pool = pool[parent.contains(pool)] if parent is not None else pool
    high = elevated(interpolate(X, e, in_pool, k), k)
    share = float(high.mean())
    if share == 0 or share > 0.8:                       # flat inside: stop tightening
        return (parent, mass) if parent is not None else (mass_ball(X[i], pool, 0.25), 0.25)
    new_mass = max(mass * share, min_mass)
    near = in_pool[high]
    near = near[mass_ball(X[i], pool, new_mass).contains(near)]
    centre = near.mean(0) if len(near) else X[i]        # centre of the elevated part
    region = mass_ball(centre, pool, new_mass)
    if not stands_out(e[inside], region.contains(X[inside])):   # no evidence for a tighter region
        return (parent, mass) if parent is not None else (mass_ball(X[i], pool, 0.25), 0.25)
    return region, new_mass


def explore_grid(problem, pool, n, rs):
    """Equal allocation over the cells of a fixed grid: ``b`` bins per axis, as many as the
    budget can fill with ten tests a cell. Cell draws come from the pool."""
    d = problem.d
    b = max(int((n / 10) ** (1 / d)), 1)
    cell = np.minimum((pool * b).astype(int), b - 1)
    keys = (cell * b ** np.arange(d)).sum(1)
    uniq = np.unique(keys)
    per = max(n // len(uniq), 1)
    idx = np.concatenate([rs.permutation(np.flatnonzero(keys == u))[:per] for u in uniq])
    idx = idx[:n]
    X = pool[idx]
    e = problem.failure(X, rs)
    means = {u: e[keys[idx] == u].mean() for u in np.unique(keys[idx])}
    worst = max(means, key=means.get)
    low = np.floor(pool[keys == worst][0] * b) / b
    return X, e, Box(low, low + 1.0 / b), 1.0 / len(uniq)


def explore_soft(problem, X, e, pool, n, rounds, rs, mode, k=25, lam=0.7):
    """Per round, draw candidates of the pool with probability ``(1 - lam)`` uniform plus
    ``lam`` by landscape, with no hard boundary. Works on a fixed pool as it stands."""
    free, left = np.ones(len(pool), bool), n
    for t in range(rounds):
        m = left // (rounds - t)
        cand = np.flatnonzero(free)
        score = interpolate(X, e, pool[cand], min(k, len(X)))
        if mode == "soft_optimistic":
            d = cKDTree(X).query(pool[cand])[0]
            score = score + 0.1 * d / d.mean()
        elif mode == "soft_density" and problem.X_train is not None:
            d = cKDTree(problem.X_train).query(pool[cand])[0]
            score = score + 0.1 * d / d.mean()
        p = np.full(len(cand), (1 - lam) / len(cand))
        p = p + lam * (score / score.sum() if score.sum() > 0 else 1 / len(cand))
        pick = rs.choice(len(cand), m, replace=False, p=p / p.sum())
        free[cand[pick]] = False
        new = pool[cand[pick]]
        X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
        left -= m
    return X, e


def explore_straddle(problem, X, e, pool, n, rounds, rs, tau, k=25):
    """Level set estimation with the straddle rule (Bryan et al. 2005): per round, test the
    pool candidates where the estimate is closest to tau relative to its uncertainty,
    1.96 sd - |p - tau|. A k-neighbour rate stands in for the Gaussian process."""
    free, left = np.ones(len(pool), bool), n
    for t in range(rounds):
        m = left // (rounds - t)
        cand = np.flatnonzero(free)
        kk = min(k, len(X))
        p = interpolate(X, e, pool[cand], kk)
        dist = cKDTree(X).query(pool[cand])[0]
        sd = np.sqrt(np.maximum(p * (1 - p), 1 / kk) / kk) * (1 + dist / dist.mean())
        pick = cand[np.argsort(-(1.96 * sd - np.abs(p - tau)))[:m]]
        free[pick] = False
        new = pool[pick]
        X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
        left -= m
    return X, e


def explore_sobol(problem, X, e, n, rs):
    """Space filling: a scrambled Sobol sequence on the unit cube (Q is uniform here)."""
    from scipy.stats import qmc
    new = qmc.Sobol(problem.d, scramble=True, seed=rs.randint(2 ** 31)).random(n)
    return np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])


def peel_step(X, e, box, pool, min_mass, cuts=(0.25, 0.5, 0.75)):
    """The trim of ``box`` that best separates failures: one cut on one axis, kept side
    with the higher failure rate, at quantiles of Q in the box. None if nothing stands out."""
    inside = box.contains(X)
    in_pool = pool[box.contains(pool)]
    best, best_z = None, 3.0
    for j in range(X.shape[1]):
        for c in np.quantile(in_pool[:, j], cuts):
            for keep_low in (True, False):
                low, high = box.low.copy(), box.high.copy()
                if keep_low:
                    high[j] = c
                else:
                    low[j] = c
                child = Box(low, high)
                z = gap(e[inside], child.contains(X[inside]))
                if z > best_z and child.contains(pool).mean() >= min_mass:
                    best, best_z = child, z
    return best


def explore_peel(problem, X, e, pool, n, rounds, rs, min_mass, trace=None):
    """Start from the whole domain, as a one-cell binning, and refine only where the tests
    say so. Cells follow the input axes, so that irrelevant coordinates stay uncut."""
    box, left, draws = Box(np.zeros(problem.d), np.ones(problem.d) + 1e-9), n, 0
    for t in range(rounds):
        while True:
            child = peel_step(X, e, box, pool, min_mass)
            if child is None:
                break
            box = child
        if trace is not None:                            # for figures: box and first test of round t
            trace.append((Box(box.low.copy(), box.high.copy()), len(X)))
        new, used = draw_in(problem, box, left // (rounds - t), rs)
        draws += used
        X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
        left -= len(new)
    return X, e, box, float(box.contains(pool).mean()), draws


def explore_gp(problem, X, e, pool, n, rs, tau, rounds=10, n_cand=3000, max_fit=1000):
    """Level set estimation with a Gaussian process (Gotovos et al. 2013, straddle score of
    Bryan et al. 2005): per round, fit a GP with an RBF kernel and a noise term to the
    outcomes, and test the pool candidates with the largest 1.96 sd - |mean - tau|."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    d, left = X.shape[1], n
    kernel = ConstantKernel(0.1, (1e-3, 10)) * RBF(np.full(d, 0.3), (1e-2, 10)) + \
        WhiteKernel(0.05, (1e-4, 1))
    for t in range(rounds):
        m = left // (rounds - t)
        fit = rs.choice(len(X), min(len(X), max_fit), replace=False)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=0).fit(X[fit], e[fit])
        cand = pool[rs.choice(len(pool), n_cand, replace=False)]
        mu, sd = gp.predict(cand, return_std=True)
        new = cand[np.argsort(-(1.96 * sd - np.abs(mu - tau)))[:m]]
        X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
        left -= m
    return X, e


def explore_ce(problem, X, e, pool, n, rounds, rs, k=25, rho=0.1, smooth=0.7):
    """Cross-entropy search for failures (Rubinstein; de Boer et al. 2005): per round, fit a
    Gaussian with diagonal covariance to the elite share ``rho`` of the tests by landscape,
    smoothed with the previous fit, and draw the next tests from the pool with weights
    proportional to its density, so that every test stays an operational input."""
    mean, sd, left = pool.mean(0), pool.std(0), n
    for t in range(rounds):
        m = left // (rounds - t)
        land = interpolate(X, e, X, min(k, len(X)))
        elite = X[land >= np.quantile(land, 1 - rho)]
        mean = smooth * elite.mean(0) + (1 - smooth) * mean
        sd = smooth * np.maximum(elite.std(0), 0.02) + (1 - smooth) * sd
        logw = -0.5 * (((pool - mean) / sd) ** 2).sum(1)
        p = np.exp(logw - logw.max())
        new = pool[rs.choice(len(pool), m, replace=False, p=p / p.sum())]
        X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
        left -= m
    return X, e


def run(problem, n, strategy, seed, kappa=0.3, conf=0.25, rounds=4, delta=0.05, tau=0.1,
        min_mass=0.01, pool_size=50000, tests=False):
    """One campaign. All randomness of the system comes from ``seed``, so that strategies
    can be compared pairwise on the same problem."""
    rs = np.random.RandomState(seed)
    base = problem
    if strategy.endswith("_scalar"):                     # steer on the scalar, claim on failures
        from scripts.regional.problems import Steered
        problem, strategy = Steered(problem), strategy[: -len("_scalar")]
    pool = problem.sample(pool_size, rs)
    n_conf = int(round(conf * n))
    n_explore = n - n_conf
    X, e, region, mass, draws, n_rep = explore(problem, pool, n_explore, strategy, rs, kappa,
                                               rounds, tau, min_mass)
    if problem is not base:                              # back to the binary failures
        e = np.concatenate(problem.log)
        assert len(e) == len(X)
        problem = base
    Xc, used = draw_in(problem, region, n_conf, rs)
    draws += used
    ec = problem.failure(Xc, rs)
    lo, hi = cp_interval(int(ec.sum()), len(ec), delta)
    rep_fail = int(e[:n_rep].sum())
    report = dict(queries=len(X) + len(Xc), draws=draws, global_upper=cp_upper(rep_fail, n_rep, delta),
                  n_rep=n_rep, region_mass=mass, n_conf=len(ec), failures=int(e.sum() + ec.sum()),
                  region_mean=float(ec.mean()) if len(ec) else float("nan"), lo=lo, hi=hi,
                  label=label(lo, hi, tau))
    return (report, region, np.vstack([X, Xc]), np.concatenate([e, ec])) if tests else (report, region)


def explore(problem, pool, n_explore, strategy, rs, kappa=0.3, rounds=4, tau=0.1, min_mass=0.01,
            trace=None):
    """The exploration of a campaign: a representative stream of share ``kappa``, then the
    strategy. Returns the tests, their outcomes, the proposed region and its mass, the
    rejected draws and the size of the representative stream (the first ``n_rep`` tests)."""
    share = 1.0 if strategy in ("random", "oracle") else kappa
    n_rep = int(round(share * n_explore))
    draws = 0
    X = problem.sample(n_rep, rs)
    e = problem.failure(X, rs)

    if strategy == "binning":
        X, e, region, mass = explore_grid(problem, pool, n_explore, rs)
        n_rep = len(X)
    elif strategy == "kcenter":
        used = X if problem.X_train is None else np.vstack([problem.X_train, X])
        cand = pool[:5000]
        new = cand[pick_kcenter(cand, used, n_explore - n_rep)]
        X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy == "oracle":
        truth = problem.truth()
        if truth is None:                                # no compact region exists
            region, mass = mass_ball(pool[0], pool, 0.1), 0.1
        else:
            region = Ball(truth[0], truth[1], axes=[0, 1])
            mass = float(region.contains(pool).mean())
    elif strategy == "random":
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy == "sobol":
        X, e = explore_sobol(problem, X, e, n_explore - n_rep, rs)
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy == "straddle":
        X, e = explore_straddle(problem, X, e, pool, n_explore - n_rep, rounds, rs, tau)
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy == "gp_lse":
        X, e = explore_gp(problem, X, e, pool, n_explore - n_rep, rs, tau)
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy == "cross_entropy":
        X, e = explore_ce(problem, X, e, pool, n_explore - n_rep, rounds, rs)
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy.startswith("soft"):
        X, e = explore_soft(problem, X, e, pool, n_explore - n_rep, rounds, rs, strategy)
        region, mass = propose(X, e, pool, None, 1.0, rs, min_mass=min_mass)
    elif strategy == "peel":
        X, e, region, mass, draws = explore_peel(problem, X, e, pool, n_explore - n_rep, rounds,
                                                 rs, min_mass, trace)
    elif strategy == "subspace":
        axes, inner, mass = relevant_axes(X, e), None, 1.0
        left, per = n_explore - n_rep, (n_explore - n_rep) // rounds
        for t in range(rounds):
            inner, mass = propose(X[:, axes], e, pool[:, axes], inner, mass, rs, min_mass=min_mass)
            region = Ball(inner.centre, inner.radius, axes)
            new, used = draw_in(problem, region, per if t < rounds - 1 else left, rs)
            draws += used
            X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
            left -= len(new)
    else:                                                # zooming strategies and controls
        region, mass = None, 1.0
        left, per = n_explore - n_rep, (n_explore - n_rep) // rounds
        for t in range(rounds):
            region, mass = propose(X, e, pool, region, mass, rs, min_mass=min_mass, mode=strategy)
            if trace is not None:                        # for figures: region and first test of round t
                trace.append((region, len(X)))
            new, used = draw_in(problem, region, per if t < rounds - 1 else left, rs)
            draws += used
            X, e = np.vstack([X, new]), np.concatenate([e, problem.failure(new, rs)])
            left -= len(new)
    return X, e, region, mass, draws, n_rep


def score(problem, region, report, tau, seed, n_eval=200000):
    """Truth for one report, from a fresh evaluation sample of Q."""
    Z = problem.sample(n_eval, np.random.RandomState(seed + 1000003))
    f = problem.mean_failure(Z)
    inside, bad = region.contains(Z), f > tau
    mu = float(f[inside].mean()) if inside.any() else float("nan")
    hit = float((inside & bad).sum())
    return dict(true_risk=float(f.mean()), true_region=mu, true_mass=float(inside.mean()),
                bad_mass=float(bad.mean()),
                precision=hit / max(inside.sum(), 1), recall=hit / max(bad.sum(), 1),
                covered=bool(report["lo"] <= mu <= report["hi"]),
                global_covered=bool(f.mean() <= report["global_upper"]),
                false_flag=bool(report["label"] == "flagged" and mu <= tau),
                false_clear=bool(report["label"] == "cleared" and mu > tau))
