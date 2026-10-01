"""Scaled dot-product attention (Chapter 19)."""
import numpy as np

__chapter__ = "attention"


# tag::masks[]
def causal_mask(length):
    """True where a query position may attend to a key position."""
    positions = np.arange(length)
    return positions[:, None] >= positions[None, :]


def padding_mask(valid):
    """Convert a boolean (B, T) validity array to a (B, 1, T) key mask."""
    return np.asarray(valid, dtype=bool)[:, None, :]
# end::masks[]


def masked_softmax(scores, mask=None):
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    row_max = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - np.where(np.isfinite(row_max), row_max, 0.0)
    exp = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
    denom = exp.sum(axis=-1, keepdims=True)
    return np.divide(exp, denom, out=np.zeros_like(exp), where=denom > 0)


# tag::forward[]
def attention_forward(Q, K, V, mask=None, return_cache=False):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V."""
    scale = 1.0 / np.sqrt(Q.shape[-1])
    scores = (Q @ np.swapaxes(K, -1, -2)) * scale
    weights = masked_softmax(scores, mask)
    output = weights @ V
    if not return_cache:
        return output
    return output, (Q, K, V, weights, scale, mask)
# end::forward[]


# tag::backward[]
def attention_backward(grad_output, cache):
    """Backward pass for scaled dot-product attention."""
    Q, K, V, weights, scale, mask = cache
    grad_V = np.swapaxes(weights, -1, -2) @ grad_output
    grad_weights = grad_output @ np.swapaxes(V, -1, -2)
    row_dot = np.sum(grad_weights * weights, axis=-1, keepdims=True)
    grad_scores = weights * (grad_weights - row_dot)
    if mask is not None:
        grad_scores = np.where(mask, grad_scores, 0.0)
    grad_Q = (grad_scores @ K) * scale
    grad_K = (np.swapaxes(grad_scores, -1, -2) @ Q) * scale
    return grad_Q, grad_K, grad_V
# end::backward[]


# tag::variance[]
def dot_product_variance(width, samples=50_000, seed=19):
    """Monte Carlo Var(q dot k) for independent unit-variance entries."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((samples, width))
    K = rng.standard_normal((samples, width))
    return float(np.var(np.sum(Q * K, axis=1)))
# end::variance[]
