"""Batched matrix products, head reshapes, and einsum (Appendix B)."""
import numpy as np


# tag::heads[]
def split_heads(X, heads):
    """(B, T, H * d_h) -> (B, H, T, d_h): cut each feature vector into H chunks."""
    B, T, width = X.shape
    return X.reshape(B, T, heads, width // heads).transpose(0, 2, 1, 3)


def merge_heads(X):
    """(B, H, T, d_h) -> (B, T, H * d_h), the inverse of split_heads."""
    B, H, T, d_head = X.shape
    return X.transpose(0, 2, 1, 3).reshape(B, T, H * d_head)
# end::heads[]


# tag::scores[]
def all_pair_dot_products(Q, K):
    """Q, K (B, H, T, d_h) -> S (B, H, T, T).

    S[b, h, t, s] is the dot product of query t with key s:
    Q[b, h, t] . K[b, h, s].
    """
    return np.einsum("bhtd,bhsd->bhts", Q, K)


def all_pair_dot_products_matmul(Q, K):
    return Q @ np.swapaxes(K, -1, -2)            # (..., T, d_h) @ (..., d_h, T)
# end::scores[]
