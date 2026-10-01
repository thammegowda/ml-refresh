"""Turning uniform random numbers into samples (Chapter 6)."""
import numpy as np


# tag::inverse-cdf[]
def sample_exponential(rate, n, rng):
    """Invert F(x) = 1 - exp(-rate * x): x = -log(1 - u) / rate for uniform u."""
    return -np.log1p(-rng.random(n)) / rate
# end::inverse-cdf[]


# tag::gumbel[]
def sample_gumbel_max(logits, rng):
    """argmax(logits + Gumbel noise) is one draw from softmax(logits), per row."""
    u = rng.uniform(np.finfo(np.float64).tiny, 1.0, size=np.shape(logits))
    gumbel = -np.log(-np.log(u))
    return np.argmax(logits + gumbel, axis=-1)
# end::gumbel[]
