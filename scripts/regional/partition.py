"""
Setting B: fixed data. Where does a given data set say that more testing is needed?

The data are given (logged operations, an existing test set): ``n`` inputs from Q with a
yes/no outcome. Nothing can be queried, so the only choice is the resolution: a partition of
the space into regions, each of which is then labelled flagged, cleared or undetermined
against ``tau``. The same rule limits every method: a region must hold at least ``m``
expected events at the average rate, as a rate cannot be estimated with fewer.

Search and evidence are kept apart. Each method builds its regions on the exploration half
E and every label comes from the verification half V, with exact Clopper-Pearson bounds and
a Bonferroni correction over all regions of the method. ``weakspot_naive`` builds and
labels on all data, which shows what the split protects against.

Methods (``build`` argument):

    global          one region: the whole space (reference)
    grid3           three bins per axis, the usual expert grid
    grid_fit        the finest uniform grid whose cells hold ``m`` expected events
    twise           three bins per axis on every pair of coordinates (pairwise coverage)
    cart            classification tree on the raw outcomes, leaves of ``m`` expected events
    prim            PRIM bump hunting: up to five peeled boxes and the remainder
    weakspot        refinement guided by the landscape: split only regions that are still
                    undecided and cut where the landscape crosses tau
    weakspot_naive  weakspot, built and labelled on all data (invalid on purpose)
    weakspot_forest weakspot with the landscape estimated by a random forest (out-of-bag),
                    which, unlike the neighbour rate, ignores irrelevant coordinates
    weakspot_pca    weakspot with a supervised principal-component metric: the neighbour rate is
                    taken in the directions where the failing points differ from all points
                    (mean shift and change of covariance), so that relevant axes and oblique
                    directions weigh more and irrelevant axes less; cuts stay on the axes
    weakspot_gini   hybrid of CART and weakspot: cuts chosen on the raw outcomes as in CART,
                    but only undecided regions are refined, largest first (weakspot allocation)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts.regional.bounds import cp_interval

METHODS = ("global", "grid3", "grid_fit", "twise", "cart", "prim", "weakspot", "weakspot_naive")
HYBRIDS = ("cart", "weakspot", "weakspot_forest", "weakspot_gini", "weakspot_pca")


# ---------------------------------------------------------------- partitions
class Grid:
    """``b`` equal bins per axis on the coordinates ``axes``."""

    def __init__(self, b, axes):
        self.b, self.axes = b, list(axes)

    def ids(self, X):
        c = np.minimum((X[:, self.axes] * self.b).astype(np.int64), self.b - 1)
        return (c * self.b ** np.arange(len(self.axes), dtype=np.int64)).sum(1)


class Tree:
    def __init__(self, tree):
        self.tree = tree

    def ids(self, X):
        return self.tree.apply(X)


class Boxes:
    """Ordered boxes; a point belongs to the first box that holds it, else to the remainder."""

    def __init__(self, boxes):
        self.boxes = boxes

    def ids(self, X):
        out = np.full(len(X), len(self.boxes))
        for i in reversed(range(len(self.boxes))):
            lo, hi = self.boxes[i]
            out[((X >= lo) & (X <= hi)).all(1)] = i
        return out


class Leaves:
    """The leaves of a refinement, each a box [lo, hi)."""

    def __init__(self, leaves):
        self.leaves = leaves

    def ids(self, X):
        out = np.zeros(len(X), int)
        for i, (lo, hi) in enumerate(self.leaves):
            out[((X >= lo) & (X < hi)).all(1)] = i
        return out


# ---------------------------------------------------------------- builders
def min_leaf(y, m):
    """Points a region needs to hold ``m`` expected events at the average rate."""
    p = max(y.mean(), 1.0 / len(y))
    return int(np.ceil(m / p))


def build_cart(X, y, m):
    leaf = min(min_leaf(y, m), len(y))
    if y.sum() == 0 or leaf * 2 > len(y):
        return [Grid(1, [0])]
    return [Tree(DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(X, y))]


def build_prim(X, y, m, alpha=0.1, n_boxes=5):
    """Friedman and Fisher (1999): peel ``alpha`` of the points off one side of one axis at
    a time, keeping the peel that raises the mean most; the box of highest mean along the
    trajectory is kept, its points removed, and the search repeated."""
    leaf, d = min_leaf(y, m), X.shape[1]
    free, boxes, base = np.ones(len(X), bool), [], y.mean()
    for _ in range(n_boxes):
        idx = np.flatnonzero(free)
        if len(idx) < 2 * leaf:
            break
        lo, hi = np.zeros(d), np.ones(d)
        inside = idx
        best = (y[inside].mean(), lo.copy(), hi.copy())
        while len(inside) * (1 - alpha) >= leaf:
            cand = None
            for j in range(d):
                v = X[inside, j]
                for side, c in (("lo", np.quantile(v, alpha)), ("hi", np.quantile(v, 1 - alpha))):
                    keep = v > c if side == "lo" else v < c
                    if keep.sum() < leaf or keep.all():
                        continue
                    mu = y[inside[keep]].mean()
                    if cand is None or mu > cand[0]:
                        cand = (mu, j, side, c, keep)
            if cand is None:
                break
            mu, j, side, c, keep = cand
            if side == "lo":
                lo[j] = c
            else:
                hi[j] = c
            inside = inside[keep]
            if mu > best[0]:
                best = (mu, lo.copy(), hi.copy())
        if best[0] <= base:
            break
        boxes.append((best[1], best[2]))
        free &= ~((X >= best[1]) & (X <= best[2])).all(1)
    return [Boxes(boxes)]


def smooth(X, y, m):
    """The landscape on the data themselves: a k-neighbour rate, with k large enough to hold
    about ``2 m`` events at the average rate, so that rare events are not read as noise."""
    k = int(np.clip(np.ceil(2 * m / max(y.mean(), 1.0 / len(y))), 25, max(len(X) // 10, 25)))
    k = min(k, len(X))
    nb = cKDTree(X).query(X, k=k)[1]
    return y[nb].mean(1)


def forest(X, y, m):
    """The landscape from a random forest, out-of-bag so that no point predicts itself.
    Leaves hold the same ``m`` expected events as every region."""
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(len(y), y.mean())
    leaf = max(min(min_leaf(y, m), len(y) // 4), 5)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, oob_score=True,
                                n_jobs=1, random_state=0).fit(X, y)
    p = rf.oob_decision_function_[:, 1]
    return np.where(np.isnan(p), y.mean(), p)


def directions(X, y):
    """Supervised principal components: the eigen-directions of how the failing points differ
    from all points, M = (C1 - C)^2 + mu mu^T with mu the mean shift and C1, C the covariances
    of the failing and of all points (as in sliced average variance estimation). Returns a
    matrix W; distances in X @ W weigh each direction by the square root of its eigenvalue.
    Uniform Q has no directions of its own, so an unsupervised PCA of X would find nothing."""
    d = X.shape[1]
    if y.sum() < d + 2 or y.sum() == len(y):
        return np.eye(d)
    s = X.std(0) + 1e-12
    Z = (X - X.mean(0)) / s
    F = Z[y == 1]
    mu = F.mean(0)
    D = np.cov(F, rowvar=False) - np.eye(d)
    lam, V = np.linalg.eigh(D @ D + np.outer(mu, mu))
    lam = np.clip(lam, 0, None)
    w = np.sqrt(lam / max(lam.max(), 1e-12))
    return (V * w) / s[:, None]


def smooth_pca(X, y, m):
    return smooth(X @ directions(X, y), y, m)


def gini(a):
    p = a.mean() if len(a) else 0.0
    return p * (1 - p)


def build_weakspot(X, y, m, tau, delta=0.05, cuts=np.linspace(0.1, 0.9, 9), max_leaves=200,
                   estimate="knn", target="side"):
    """Iterative zoom on fixed data. Start from one region; in each step, among regions that
    are not yet decided, refine the one with the most undecided mass, at the cut that best
    separates the landscape into parts above and below ``tau``. A region is left as it is
    once its own data decide it and its landscape lies on one side of ``tau``."""
    d, leaf = X.shape[1], min_leaf(y, m)
    est = {"knn": smooth, "forest": forest, "pca": smooth_pca}[estimate]
    side = est(X, y, m) > tau
    goal = y.astype(float) if target == "raw" else side    # what the cuts separate

    def best_cut(idx):
        """The cut of a region, or None if the region needs no refinement."""
        if len(idx) < 2 * leaf:
            return None
        a, b = cp_interval(int(y[idx].sum()), len(idx), delta)
        if not 0 < side[idx].mean() < 1 and (b <= tau or a > tau):
            return None                                    # decided, landscape on one side
        parent, best = gini(goal[idx]) * len(idx), None
        for j in range(d):
            v = X[idx, j]
            for c in np.unique(np.quantile(v, cuts)):
                left = v < c
                nl = int(left.sum())
                if nl < leaf or len(idx) - nl < leaf:
                    continue
                gain = parent - gini(goal[idx[left]]) * nl - gini(goal[idx[~left]]) * (len(idx) - nl)
                if gain > 1e-9 and (best is None or gain > best[0]):
                    best = (gain, j, c, left)
        return best

    root = np.arange(len(X))
    leaves = [(np.zeros(d), np.ones(d) + 1e-9, root, best_cut(root))]
    while len(leaves) < max_leaves:
        open_ = [i for i, L in enumerate(leaves) if L[3] is not None]
        if not open_:
            break
        i = max(open_, key=lambda i: len(leaves[i][2]))  # most undecided mass first
        lo, hi, idx, (_, j, c, left) = leaves.pop(i)
        hl, lr = hi.copy(), lo.copy()
        hl[j], lr[j] = c, c
        for a, b, sub in ((lo, hl, idx[left]), (lr, hi, idx[~left])):
            leaves.append((a, b, sub, best_cut(sub)))
    return [Leaves([(lo, hi) for lo, hi, _, _ in leaves])]


def build(method, X, y, m, tau):
    d = X.shape[1]
    if method == "global":
        return [Grid(1, [0])]
    if method == "grid3":
        return [Grid(3, range(d))]
    if method == "grid_fit":
        b = max(int((y.sum() / m) ** (1 / d)), 1)
        return [Grid(b, range(d))]
    if method == "twise":
        return [Grid(3, [j, l]) for j in range(d) for l in range(j + 1, d)] or [Grid(3, [0])]
    if method == "cart":
        return build_cart(X, y, m)
    if method == "prim":
        return build_prim(X, y, m)
    if method in ("weakspot", "weakspot_naive"):
        return build_weakspot(X, y, m, tau)
    if method == "weakspot_forest":
        return build_weakspot(X, y, m, tau, estimate="forest")
    if method == "weakspot_pca":
        return build_weakspot(X, y, m, tau, estimate="pca")
    if method == "weakspot_gini":
        return build_weakspot(X, y, m, tau, estimate="forest", target="raw")
    raise ValueError(method)


# ---------------------------------------------------------------- labels and truth
def label_cells(families, X, y, tau, delta):
    """Counts and Bonferroni-corrected labels of every cell of every family, on (X, y)."""
    out, total = [], 0
    counts = []
    for fam in families:
        ids = fam.ids(X)
        u, inv = np.unique(ids, return_inverse=True)
        counts.append((u, np.bincount(inv), np.bincount(inv, weights=y)))
        total += len(u)
    total = max(total, 1)
    for fam, (u, n, k) in zip(families, counts):
        cells = {}
        for c, nc, kc in zip(u, n, k):
            lo, hi = cp_interval(int(kc), int(nc), delta / total)
            lab = "flagged" if lo > tau else "cleared" if hi <= tau else "undetermined"
            cells[int(c)] = (lab, kc / nc, lo, hi)
        out.append(cells)
    return out, total


def verdict(families, cells, Z):
    """Per point: 2 flagged if any family flags its cell, 0 cleared if every family clears it,
    1 undetermined otherwise. Cells without verification data are undetermined."""
    flag = np.zeros(len(Z), bool)
    clear = np.ones(len(Z), bool)
    est = np.zeros(len(Z))
    for fam, cl in zip(families, cells):
        ids = fam.ids(Z)
        lab = np.array([cl.get(int(c), ("undetermined", 0.0))[0] for c in ids]) if len(cl) < 50 else None
        if lab is None:                                    # many cells: map by lookup table
            keys = np.array(list(cl.keys()))
            labs = np.array([v[0] for v in cl.values()])
            rates = np.array([v[1] for v in cl.values()])
            order = np.argsort(keys)
            keys, labs, rates = keys[order], labs[order], rates[order]
            pos = np.clip(np.searchsorted(keys, ids), 0, len(keys) - 1)
            hit = keys[pos] == ids
            lab = np.where(hit, labs[pos], "undetermined")
            rate = np.where(hit, rates[pos], 0.0)
        else:
            rate = np.array([cl.get(int(c), ("", 0.0))[1] for c in ids])
        flag |= lab == "flagged"
        clear &= lab == "cleared"
        est = np.maximum(est, rate)
    return np.where(flag, 2, np.where(clear, 0, 1)), est


def cell_validity(families, cells, Z, f, tau, min_points=30):
    """Family-wise errors against the truth: a cleared cell whose true rate exceeds tau, or a
    flagged cell whose true rate does not; share of cells whose interval covers the truth."""
    false_clear = false_flag = False
    cover, seen = 0, 0
    for fam, cl in zip(families, cells):
        ids = fam.ids(Z)
        u, inv = np.unique(ids, return_inverse=True)
        mu = np.bincount(inv, weights=f) / np.bincount(inv)
        nz = np.bincount(inv)
        for c, m_, nc in zip(u, mu, nz):
            if nc < min_points or int(c) not in cl:
                continue
            lab, _, lo, hi = cl[int(c)]
            seen += 1
            cover += lo <= m_ <= hi
            false_clear |= lab == "cleared" and m_ > tau
            false_flag |= lab == "flagged" and m_ <= tau
    return dict(false_clear=bool(false_clear), false_flag=bool(false_flag),
                cell_cover=cover / seen if seen else float("nan"))


def evaluate(problem, method, n, seed, m=5, delta=0.05, split=0.5, n_eval=50000):
    """One fixed data set of size ``n``, one method, scored against the truth."""
    tau = getattr(problem, "tau", 0.1)
    rs = np.random.RandomState(seed + 7)
    X = problem.sample(n, rs)
    y = problem.failure(X, rs)
    if method == "weakspot_naive":
        E = V = np.arange(n)
    else:
        perm = rs.permutation(n)
        E, V = perm[: int(split * n)], perm[int(split * n):]
    families = build(method, X[E], y[E], m, tau)
    cells, total = label_cells(families, X[V], y[V], tau, delta)
    Z = problem.sample(n_eval, np.random.RandomState(seed + 1000003))
    f = problem.mean_failure(Z)
    bad = f > tau
    v, est = verdict(families, cells, Z)
    flagged, cleared = v == 2, v == 0
    hit = float((flagged & bad).mean())
    prec = hit / max(flagged.mean(), 1e-12)
    rec = hit / max(bad.mean(), 1e-12)
    return dict(tau=tau, events=int(y.sum()), cells=total,
                flagged_mass=float(flagged.mean()), cleared_mass=float(cleared.mean()),
                undetermined_mass=float((v == 1).mean()), bad_mass=float(bad.mean()),
                precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
                hidden_bad=float((cleared & bad).mean() / max(bad.mean(), 1e-12)),
                match=float(((est > tau) == bad).mean()),
                enough=bool((v == 1).mean() <= 0.1),
                # enough without hiding: a cleared region must not hide the failing set
                enough_safe=bool((v == 1).mean() <= 0.1
                                 and (cleared & bad).mean() <= 0.1 * max(bad.mean(), 1e-12)),
                **cell_validity(families, cells, Z, f, tau))
