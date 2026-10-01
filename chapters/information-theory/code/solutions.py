"""Worked solutions to the Chapter 7 exercises."""
import numpy as np

from scratch.information import cross_entropy


# tag::empirical[]
def empirical_distribution(samples, categories):
    """The fraction of samples equal to each category: p_hat."""
    return np.bincount(samples, minlength=categories) / len(samples)


def average_nll(samples, q):
    """The maximum-likelihood objective: -(1/N) sum_i log q(x_i)."""
    return float(-np.mean(np.log(q[samples])))


def cross_entropy_of_empirical(samples, q):
    """The same number, as the cross-entropy H(p_hat, q)."""
    return float(cross_entropy(empirical_distribution(samples, len(q)), q))
# end::empirical[]


# tag::surprisal[]
def surprisal_bits(probability):
    return float(-np.log2(probability))
# end::surprisal[]
