"""
Exact binomial bounds of the campaign report.

Both claims rest on fixed-sample arguments and need no new theory: the representative
stream is i.i.d. from Q whatever the guided stream does, and the confirmation tests are
fresh draws from Q restricted to a region chosen before they were made.
"""
from __future__ import annotations

from scipy.stats import beta


def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound on a failure rate."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def cp_interval(k, n, delta):
    """Two-sided Clopper-Pearson interval at level ``1 - delta``."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(delta / 2, k, n - k + 1))
    return lo, cp_upper(k, n, delta / 2)


def label(lo, hi, tau):
    """Decision of a confirmed region against the requirement ``tau``."""
    return "flagged" if lo > tau else "cleared" if hi <= tau else "undetermined"
