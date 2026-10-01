"""Monte Carlo estimation and gradients of expectations (Chapter 6)."""
import numpy as np


# tag::monte-carlo[]
def monte_carlo(f, sample, n, rng):
    """Estimate E[f(X)] from n draws, and the standard error of that estimate."""
    values = f(sample(n, rng))
    return values.mean(), values.std(ddof=1) / np.sqrt(n)
# end::monte-carlo[]


# tag::gradients[]
def score_function_gradient(f, mean, std, n, rng):
    """d/d(mean) of E[f(X)], X ~ N(mean, std^2), as the average of f(x) * score(x)."""
    x = mean + std * rng.standard_normal(n)
    score = (x - mean) / std ** 2              # d log p(x) / d mean
    return np.mean(f(x) * score)


def reparameterized_gradient(df, mean, std, n, rng):
    """The same derivative through x = mean + std * eps: the average of f'(x)."""
    x = mean + std * rng.standard_normal(n)
    return np.mean(df(x))
# end::gradients[]
