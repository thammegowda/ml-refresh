"""Probability mass and density functions (Chapter 6)."""
import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])


# tag::densities[]
def bernoulli_pmf(x, p):
    """P(X = x) for x in {0, 1} when X ~ Bernoulli(p)."""
    return np.where(x == 1, p, 1 - p)


def gaussian_pdf(x, mean, std):
    """Density of N(mean, std^2) at x: a height, not a probability."""
    z = (x - mean) / std
    return np.exp(-0.5 * z ** 2) / (std * np.sqrt(2 * np.pi))


def gaussian_log_pdf(x, mean, std):
    """log of gaussian_pdf, computed without exponentiating."""
    z = (x - mean) / std
    return -0.5 * z ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
# end::densities[]


# tag::cdf[]
def gaussian_cdf(x, mean=0.0, std=1.0):
    """P(X <= x) for X ~ N(mean, std^2)."""
    return 0.5 * (1 + _erf((np.asarray(x) - mean) / (std * np.sqrt(2))))


def gaussian_interval(a, b, mean=0.0, std=1.0):
    """P(a < X < b): the area under the density between a and b."""
    return gaussian_cdf(b, mean, std) - gaussian_cdf(a, mean, std)
# end::cdf[]
