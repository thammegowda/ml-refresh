"""Arrays, dtypes, views, and randomness (Appendix B)."""
import numpy as np


# tag::float-limits[]
def float_limits():
    """Machine epsilon, largest value, and smallest normal value per dtype."""
    rows = []
    for dtype in (np.float16, np.float32, np.float64):
        info = np.finfo(dtype)
        rows.append((info.dtype.name, float(info.eps), float(info.max),
                     float(info.smallest_normal)))
    return rows
# end::float-limits[]


# tag::views[]
def views_and_copies():
    X = np.zeros((3, 4), dtype=np.float32)
    row = X[0]              # basic slicing: a view that shares X's memory
    row += 1                # ...so this writes into X
    picked = X[[0, 2]]      # integer-array indexing: a copy
    picked += 100           # ...so X is unchanged here
    return X
# end::views[]


# tag::random[]
def make_data(seed=0, n=4, d=3):
    rng = np.random.default_rng(seed)                   # one generator, passed around
    X = rng.standard_normal((n, d), dtype=np.float32)
    labels = rng.integers(0, 3, size=n)
    order = rng.permutation(n)                          # a shuffled minibatch order
    return X, labels, order
# end::random[]
