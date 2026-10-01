"""Tested snippets for Chapter 8 solutions."""
import numpy as np

from compare import exact_permutation_p_value, mcnemar_exact_p_value
from intervals import bootstrap_ci


# tag::paired-demo[]
def paired_demo():
    """A tiny benchmark where B fixes four A errors and breaks one A success."""
    a_correct = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    b_correct = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    return {
        "delta": float(b_correct.mean() - a_correct.mean()),
        "permutation_p": exact_permutation_p_value(a_correct, b_correct),
        "mcnemar_p": mcnemar_exact_p_value(a_correct, b_correct),
    }
# end::paired-demo[]


# tag::bootstrap-median[]
def median_interval(scores, rng):
    """Bootstrap a robust median score instead of an accuracy."""
    return bootstrap_ci(np.asarray(scores), np.median, draws=2000, rng=rng)
# end::bootstrap-median[]
