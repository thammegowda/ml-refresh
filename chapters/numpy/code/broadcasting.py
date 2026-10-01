"""Broadcasting and reductions (Appendix B)."""
import numpy as np


# tag::bias[]
def add_bias(X, b):
    """X (N, d) + b (d,) -> (N, d): the same b is added to every row."""
    return X + b
# end::bias[]


# tag::distances[]
def pairwise_squared_distances(A, B):
    """Rows A (N, d) and B (M, d) -> D (N, M) with D[i, j] = ||A[i] - B[j]||^2."""
    difference = A[:, None, :] - B[None, :, :]   # (N, 1, d) - (1, M, d) -> (N, M, d)
    return np.sum(difference ** 2, axis=-1)
# end::distances[]


# tag::distances-fast[]
def pairwise_squared_distances_fast(A, B):
    """The same D without the (N, M, d) intermediate: ||a||^2 + ||b||^2 - 2 a.b."""
    squared_A = np.sum(A ** 2, axis=1)[:, None]      # (N, 1)
    squared_B = np.sum(B ** 2, axis=1)[None, :]      # (1, M)
    return squared_A + squared_B - 2 * A @ B.T       # (N, M)
# end::distances-fast[]


# tag::normalize-rows[]
def normalize_rows(X):
    """Scale each row of X (N, d) so that it sums to 1."""
    return X / np.sum(X, axis=1, keepdims=True)    # (N, d) / (N, 1)
# end::normalize-rows[]
