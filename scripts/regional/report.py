"""
The full report in the query setting: every region of a partition decided or not, as in
Definition 1 of the paper, rather than one confirmed region.

    explore    a share 1 - c of the budget, as in a campaign (representative stream + strategy)
    partition  a partition built on the exploration tests only (tree or refinement)
    allocate   the confirmation budget c n spread over the regions by expected need, chosen
               from the exploration alone
    confirm    fresh draws from Q restricted to each funded region; exact intervals with a
               Bonferroni correction over the funded regions; unfunded regions stay undetermined

Allocation is where the queries pay: a region that random testing would reach with a few
tests can be given the 36 it needs, and a region whose estimate lies close to tau, which no
affordable budget decides, is given none. Labels rest on the confirmation draws only, so
that they are valid whatever the exploration and the partition did (Proposition 1).
"""
from __future__ import annotations

import numpy as np

from scripts.regional.bounds import cp_interval, cp_upper
from scripts.regional.campaign import explore
from scripts.regional.partition import build, verdict

GRID = np.unique(np.round(np.geomspace(2, 4000, 120)).astype(int))


def need(p, tau, delta):
    """The least number of tests at which a region of rate ``p`` is decided, if its share of
    failures equals ``p``; None if no number up to the grid's end suffices."""
    for N in GRID:
        lo, hi = cp_interval(int(round(p * N)), int(N), delta)
        if hi <= tau or lo > tau:
            return int(N)
    return None


def allocate(mass, p_hat, budget, tau, delta):
    """Greedy allocation by decided mass per test. The level uses a first guess of the
    number of funded regions; the confirmation itself corrects for the actual number."""
    order, alloc = np.argsort(-mass), np.zeros(len(mass), int)
    for _ in range(2):                                   # fix point of the Bonferroni level
        m_f = max(int((alloc > 0).sum()), 1)
        needs = np.array([need(p, tau, delta / m_f) or 0 for p in p_hat])
        value = np.where(needs > 0, mass / np.maximum(needs, 1), 0.0)
        order = np.argsort(-value)
        alloc[:], left = 0, budget
        for j in order:
            if value[j] <= 0 or needs[j] > left:
                continue
            alloc[j], left = needs[j], left - needs[j]
        if alloc.sum() and left > 0:                     # spare tests: more power where funded
            alloc = alloc + np.floor(left * alloc / alloc.sum()).astype(int)
    return alloc


def draw_cells(problem, fam, alloc, rs, batch=20000, max_draws=3000000):
    """Draws from Q restricted to each funded cell, by rejection, all cells at once."""
    want = {int(c): int(a) for c, a in enumerate(alloc) if a > 0}
    got = {c: [] for c in want}
    have = {c: 0 for c in want}
    draws = 0
    while any(have[c] < want[c] for c in want) and draws < max_draws:
        X = problem.sample(batch, rs)
        draws += batch
        ids = fam.ids(X)
        for c in want:
            if have[c] >= want[c]:
                continue
            hit = X[ids == c][: want[c] - have[c]]
            got[c].append(hit)
            have[c] += len(hit)
    return {c: np.vstack(v) if v else np.zeros((0, problem.d)) for c, v in got.items()}, draws


def run_report(problem, n, strategy, partition, seed, kappa=0.3, conf=0.5, rounds=4, delta=0.05,
               tau=None, min_mass=0.01, pool_size=50000, m=5, n_eval=50000, details=False):
    """One campaign with a full report, scored against the truth."""
    tau = getattr(problem, "tau", 0.1) if tau is None else tau
    rs = np.random.RandomState(seed)
    pool = problem.sample(pool_size, rs)
    n_conf = int(round(conf * n))
    X, e, _, _, draws, n_rep = explore(problem, pool, n - n_conf, strategy, rs, kappa, rounds, tau,
                                       min_mass)
    fam = build(partition, X, e, m, tau)[0]
    cells_pool = fam.ids(pool)
    M = int(max(cells_pool.max(), fam.ids(X).max())) + 1
    mass = np.bincount(cells_pool, minlength=M) / len(pool)
    ids = fam.ids(X)
    k, nn = np.bincount(ids, weights=e, minlength=M), np.bincount(ids, minlength=M)
    p_hat = (k + 2 * e.mean()) / (nn + 2)                # shrunk towards the overall rate
    alloc = allocate(mass, p_hat, n_conf, tau, delta)
    got, used = draw_cells(problem, fam, alloc, rs)
    m_f = max(len(got), 1)
    cells = {}
    for c, Xc in got.items():
        ec = problem.failure(Xc, rs)
        lo, hi = cp_interval(int(ec.sum()), len(ec), delta / m_f)
        lab = "flagged" if lo > tau else "cleared" if hi <= tau else "undetermined"
        cells[c] = (lab, float(ec.mean()) if len(ec) else 0.0, lo, hi)
    Z = problem.sample(n_eval, np.random.RandomState(seed + 1000003))
    f = problem.mean_failure(Z)
    bad = f > tau
    v, _ = verdict([fam], [cells], Z)
    flagged, cleared = v == 2, v == 0
    hit = float((flagged & bad).mean())
    prec, rec = hit / max(flagged.mean(), 1e-12), hit / max(bad.mean(), 1e-12)
    hidden = float((cleared & bad).mean() / max(bad.mean(), 1e-12))
    und = float((v == 1).mean())
    zid = fam.ids(Z)                                     # validity of every label
    false_clear = false_flag = False
    for c, (lab, _, _, _) in cells.items():
        inside = zid == c
        if inside.sum() >= 30:
            mu = f[inside].mean()
            false_clear |= lab == "cleared" and mu > tau
            false_flag |= lab == "flagged" and mu <= tau
    out = dict(tau=tau, regions=M, funded=len(got), draws=draws + used, n_rep=n_rep,
                global_upper=cp_upper(int(e[:n_rep].sum()), n_rep, delta),
                undetermined_mass=und, flagged_mass=float(flagged.mean()),
                cleared_mass=float(cleared.mean()), bad_mass=float(bad.mean()),
                precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                hidden_bad=hidden, enough=bool(und <= 0.1),
                enough_safe=bool(und <= 0.1 and hidden <= 0.1),
                false_clear=bool(false_clear), false_flag=bool(false_flag))
    return (out, fam, cells, X, e, got) if details else out
