"""Online softmax and tiled attention helpers."""
import numpy as np

__chapter__ = "flash-attention"


# tag::naive[]
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)


def naive_attention(q, k, v, causal=False):
    """Reference attention that materializes the score matrix."""
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        pos = np.arange(q.shape[0])
        scores = np.where(pos[None, :] <= pos[:, None], scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights
# end::naive[]


# tag::online-softmax[]
def online_softmax_normalizer(blocks):
    """Return the max and normalizer for one row split into blocks."""
    m = -np.inf
    ell = 0.0
    for scores in blocks:
        block_m = np.max(scores)
        new_m = max(m, block_m)
        ell *= np.exp(m - new_m)
        ell += np.exp(block_m - new_m) * np.sum(np.exp(scores - block_m))
        m = new_m
    return m, ell
# end::online-softmax[]


# tag::flash[]
def tiled_attention(q, k, v, block_size, causal=False):
    """Attention forward pass that streams K,V blocks and never stores T by T."""
    tokens, value_dim = q.shape[0], v.shape[1]
    m = np.full(tokens, -np.inf)
    ell = np.zeros(tokens)
    numerator = np.zeros((tokens, value_dim))
    q_pos = np.arange(tokens)
    scale = np.sqrt(q.shape[-1])
    for start in range(0, k.shape[0], block_size):
        stop = min(start + block_size, k.shape[0])
        scores = q @ k[start:stop].T / scale
        if causal:
            k_pos = np.arange(start, stop)
            scores = np.where(k_pos[None, :] <= q_pos[:, None], scores, -np.inf)
        block_m = np.max(scores, axis=1)
        has_scores = np.isfinite(block_m)
        new_m = np.maximum(m, block_m)
        exp_scores = np.zeros_like(scores)
        exp_scores[has_scores] = np.exp(
            scores[has_scores] - block_m[has_scores, None]
        )
        alpha = np.exp(m - new_m)
        beta = np.zeros(tokens)
        beta[has_scores] = np.exp(block_m[has_scores] - new_m[has_scores])
        numerator *= alpha[:, None]
        numerator += beta[:, None] * (exp_scores @ v[start:stop])
        ell = alpha * ell + beta * np.sum(exp_scores, axis=1)
        m = new_m
    return numerator / ell[:, None]
# end::flash[]


# tag::memory[]
def attention_memory_elements(tokens):
    """Score storage for naive attention and row state for tiled attention."""
    return {"naive_scores": tokens * tokens, "online_state": tokens}
# end::memory[]
