"""
Selection rules of iterated weakspot curation, one round at a time.

Every rule chooses ``n`` new training points from the remaining reserve. Its inputs are
the reserve in standardised inputs ``U`` (one row per candidate), the current model's
absolute errors ``e`` on the candidates, the *error landscape* ``land`` (the errors
smoothed by leave-one-out kNN, see ``scripts.dataselect.budget.landscape``) and the points
already used for training ``used`` (initial set plus earlier selections, standardised).
All rules return indices into the candidates.

The functions are independent of the experiment loop and can be reused directly::

    from scripts.dataselect.budget import landscape
    from scripts.iterative.selection import select

    land = landscape(U, e, k=10)
    pick = select("region_adapt_kc", n=200, n_focus=100, U=U, e=e, land=land,
                  used=U_train, rs=np.random.RandomState(0))

Methods (``method`` argument of :func:`select`):

    random            uniform over the reserve (the control)
    region            share ``n_focus`` uniformly from the detected weakspot (top ``q`` of
                      the landscape), the rest uniformly from the remaining reserve
    region_adapt      as ``region``, but the focus is drawn from the elevated area
    region_kc         focus from the detected weakspot, rest by k-center
    region_adapt_kc   focus + coverage: focus from the elevated area, rest by k-center
    kcenter           iterated k-center greedy over the whole batch (coverage)
    kcenter_part      share ``n_focus`` by k-center, the rest uniformly
    uniform_sched_kc  matched control of focus + coverage: share ``n_focus`` uniformly,
                      the rest by k-center
    kc_land           weakspot-weighted k-center: distance x (0.1 + landscape in [0, 1])
    toperr            the ``n`` candidates with the largest single errors
    toperr_kc         share ``n_focus`` by largest single errors, the rest by k-center
    rho_filter        reducible loss (``e - il``) inside the detected weakspot
    rho_landscape     reducible loss x landscape over the whole reserve

``n_focus`` is the focused share of the batch, ``round(alpha_t * n)`` with ``alpha_t``
from a schedule (``scripts.iterative.iter_budget.alpha_schedule``). The size-adaptive
schedule needs the weakspot size, which :func:`held_out_measures` estimates from the
errors alone, without knowing where the model is weak.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from scripts.dataselect.baselines import pick_kcenter, with_rehearsal
from scripts.dataselect.budget import landscape


def detected_weakspot(land, q):
    """Candidates in the top ``q`` fraction of the error landscape."""
    return np.flatnonzero(land >= np.quantile(land, 1 - q))


def elevated_area(land, q, kappa):
    """Candidates whose landscape exceeds ``kappa`` times its median, capped at the top
    ``q`` fraction. Empty when no candidate is clearly elevated."""
    members = detected_weakspot(land, q)
    elev = np.flatnonzero(land > kappa * np.median(land))
    return elev if len(elev) < len(members) else members


def fill_by_kcenter(focus, U, used, n):
    """Complete a focused share to ``n`` points by k-center over the other candidates."""
    rest = np.setdiff1d(np.arange(len(U)), focus)
    kc = pick_kcenter(U[rest], np.vstack([used, U[focus]]), n - len(focus))
    return np.concatenate([focus, rest[kc]]).astype(int)


def weighted_kcenter(U, used, land, n, c=0.1):
    """k-center whose farthest-point distance is scaled by ``c + land`` (land min-max
    scaled to [0, 1] on the candidates): gaps are filled first where the model is weak."""
    span = land.max() - land.min()
    w = c + ((land - land.min()) / span if span > 0 else np.zeros_like(land))
    dd = cKDTree(used).query(U)[0]
    idx = []
    for _ in range(n):
        i = int(np.argmax(dd * w))
        idx.append(i)
        dd = np.minimum(dd, np.linalg.norm(U - U[i], axis=1))
        dd[i] = -1.0
    return np.asarray(idx, dtype=int)


def select(method, n, n_focus, U, e, land, used, rs, q=0.3, kappa=1.5, il=None):
    """Choose ``n`` candidates by ``method`` (see the module docstring).

    ``rs`` is a ``numpy.random.RandomState``; its calls are part of the result, so a fixed
    state reproduces a selection exactly. ``il`` (irreducible loss per candidate) is only
    needed by the RHO-LOSS rules.
    """
    if method == "kc_land":
        return weighted_kcenter(U, used, land, n)
    if method == "toperr":
        return np.argsort(e)[::-1][:n]
    if method == "toperr_kc":
        return fill_by_kcenter(np.argsort(e)[::-1][:n_focus], U, used, n)
    if method == "uniform_sched_kc":
        u = rs.choice(len(U), n_focus, replace=False) if n_focus > 0 else np.array([], int)
        return fill_by_kcenter(u, U, used, n)
    if method == "kcenter_part":
        return with_rehearsal(pick_kcenter(U, used, n_focus), n, len(U), rs)
    if method == "kcenter":
        return pick_kcenter(U, used, n)
    if method == "random" or n_focus == 0:
        return rs.choice(len(U), n, replace=False)

    if method in ("region_adapt", "region_adapt_kc"):
        members = elevated_area(land, q, kappa)
    else:
        members = detected_weakspot(land, q)
    if len(members) == 0:                     # nothing elevated: focus over the whole reserve
        members = np.arange(len(U))
    if method in ("region", "region_adapt", "region_adapt_kc", "region_kc"):
        g = rs.choice(members, min(n_focus, len(members)), replace=False)
    elif method == "rho_filter":
        red = e - il
        g = members[np.argsort(red[members])[::-1][:min(n_focus, len(members))]]
    elif method == "rho_landscape":
        g = np.argsort(np.maximum(e - il, 0) * land)[::-1][:n_focus]
    else:
        raise ValueError(method)
    if method in ("region_adapt_kc", "region_kc"):
        return fill_by_kcenter(g, U, used, n)
    return with_rehearsal(g, n, len(U), rs)


def held_out_measures(U, e, rs, k=10, q=0.3, kappa=1.5):
    """Severity and size of the detected weakspot, measured on data it was not detected
    from: the candidates are split at random into halves A and B, the landscape is fitted
    on A and interpolated to B from the k nearest points of A.

    severity  mean error in the held-out weakspot (B above the (1-q)-quantile of the
              landscape on A) divided by the mean error in the rest of B
    p_value   one-sided Mann-Whitney test of the same two groups
    size      share of B whose landscape exceeds kappa times the median landscape on A

    Needs no ground truth, so it applies to real data as is.
    """
    half = rs.permutation(len(U)) % 2 == 0
    A, Bh = np.flatnonzero(half), np.flatnonzero(~half)
    land_A = landscape(U[A], e[A], k)
    d, nb = cKDTree(U[A]).query(U[Bh], k=k)
    w = 1.0 / np.maximum(d, 1e-12)
    land_B = (w * e[A][nb]).sum(1) / w.sum(1)
    WB = land_B >= np.quantile(land_A, 1 - q)
    inn, out = e[Bh][WB], e[Bh][~WB]
    return dict(severity=float(inn.mean() / out.mean()),
                p_value=float(mannwhitneyu(inn, out, alternative="greater").pvalue),
                size=float((land_B > kappa * np.median(land_A)).mean()))
