"""Worked solutions to the Chapter 6 exercises."""
import numpy as np


# tag::correlated-mean[]
def variance_of_correlated_mean(n, rho, trials, rng):
    """Empirical Var of the mean of n unit-variance draws, pairwise correlation rho."""
    shared = rng.standard_normal((trials, 1))           # what every draw has in common
    own = rng.standard_normal((trials, n))
    x = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own  # Var 1, Cov(x_i, x_j) = rho
    return x.mean(axis=1).var()
# end::correlated-mean[]


# tag::mle-bias[]
def average_mle_variance(n, trials, rng):
    """Average maximum-likelihood variance of many size-n samples from N(0, 1)."""
    x = rng.standard_normal((trials, n))
    return np.mean(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1))
# end::mle-bias[]


# tag::laplace[]
def laplace_regression_nll(y, prediction, scale=1.0):
    """y ~ Laplace(prediction, scale): mean absolute error / scale + log(2 scale)."""
    return np.mean(np.abs(y - prediction) / scale + np.log(2 * scale))
# end::laplace[]


# tag::baseline[]
def score_function_gradient_with_baseline(f, mean, std, n, rng, baseline):
    """Subtracting a constant from f leaves the expected gradient unchanged."""
    x = mean + std * rng.standard_normal(n)
    return np.mean((f(x) - baseline) * (x - mean) / std ** 2)
# end::baseline[]


# tag::triangle[]
def sample_triangle(n, rng):
    """Density 2x on [0, 1]: F(x) = x^2, so x = sqrt(u)."""
    return np.sqrt(rng.random(n))
# end::triangle[]
