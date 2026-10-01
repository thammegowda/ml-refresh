"""Worked solution helpers for Chapter 19."""
import numpy as np

from scratch.attention import attention_forward, causal_mask, dot_product_variance


# tag::variance-demo[]
def variance_ratios(widths=(4, 16, 64)):
    """Return Var(q dot k) / d_k for several widths."""
    return np.array([dot_product_variance(width) / width for width in widths])
# end::variance-demo[]


# tag::causal-demo[]
def causal_attention_weights():
    """Attention weights for a tiny self-attention problem with a causal mask."""
    Q = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    K = Q.copy()
    V = np.eye(3)[None, :, :]
    _output, cache = attention_forward(Q, K, V, causal_mask(3), True)
    return cache[3][0]
# end::causal-demo[]
