"""Sampling from categorical distributions (Chapter 6)."""
import numpy as np

__chapter__ = "probability"


# tag::categorical[]
def sample_categorical(probabilities, rng):
    """One draw per row: probabilities (..., K) -> indices (...,).

    Inverts the cumulative distribution: index i is chosen when
    cumulative[i - 1] <= u < cumulative[i] for a uniform u.
    """
    cumulative = np.cumsum(probabilities, axis=-1)
    total = cumulative[..., -1:]                      # 1 up to rounding
    u = rng.random(cumulative.shape[:-1] + (1,)) * total
    return np.sum(cumulative <= u, axis=-1)
# end::categorical[]
