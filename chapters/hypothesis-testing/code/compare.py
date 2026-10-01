"""Paired tests for two models evaluated on the same examples (Chapter 8)."""
from math import comb

import numpy as np


# tag::paired-bootstrap[]
def paired_bootstrap_delta(a_correct, b_correct, draws=2000, level=0.95, rng=None):
    """Bootstrap interval for accuracy(B) - accuracy(A) on paired examples."""
    delta = np.asarray(b_correct, dtype=np.float64) - np.asarray(a_correct,
                                                                dtype=np.float64)
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(delta)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        estimates[draw] = delta[rng.integers(0, n, size=n)].mean()
    alpha = (1.0 - level) / 2.0
    return delta.mean(), tuple(np.quantile(estimates, [alpha, 1.0 - alpha]))
# end::paired-bootstrap[]


# tag::paired-tests[]
def exact_permutation_p_value(a_correct, b_correct):
    """Two-sided sign-flip test for paired 0/1 correctness arrays."""
    delta = np.asarray(b_correct, dtype=int) - np.asarray(a_correct, dtype=int)
    signs = delta[delta != 0]
    observed = abs(signs.sum())
    if len(signs) == 0:
        return 1.0
    extreme = 0
    for wins_for_b in range(len(signs) + 1):
        total = 2 * wins_for_b - len(signs)
        if abs(total) >= observed:
            extreme += comb(len(signs), wins_for_b)
    return extreme / (2 ** len(signs))


def mcnemar_exact_p_value(a_correct, b_correct):
    """Exact two-sided McNemar p-value for discordant paired outcomes."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)
# end::paired-tests[]
