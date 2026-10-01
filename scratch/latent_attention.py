"""Multi-head latent attention helpers."""
import numpy as np

__chapter__ = "latent-attention"


# tag::latent[]
def compress_kv(x, w_dkv):
    """Compress token states to one latent KV vector per token."""
    return x @ w_dkv


def up_project_latents(c, w_uk, w_uv):
    """Materialize per-head keys and values from cached latents."""
    k = np.einsum("tc,chd->thd", c, w_uk)
    v = np.einsum("tc,chd->thd", c, w_uv)
    return k, v
# end::latent[]


# tag::absorb[]
def materialized_scores(q, c, w_uk):
    """Scores from explicitly materialized keys."""
    k, _ = up_project_latents(c, w_uk, w_uk)
    return np.einsum("thd,shd->hts", q, k)


def absorbed_scores(q, c, w_uk):
    """Scores after absorbing the key up-projection into queries."""
    q_latent = np.einsum("thd,chd->thc", q, w_uk)
    return np.einsum("thc,sc->hts", q_latent, c)
# end::absorb[]


# tag::mla-attention[]
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)


def mla_attention(q, c, w_uk, w_uv, mask=None):
    k, v = up_project_latents(c, w_uk, w_uv)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask[None, :, :], scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return np.einsum("hts,shd->thd", weights, v), weights
# end::mla-attention[]


# tag::rope[]
def rotate_pairs(x, positions, theta=10_000.0):
    """Apply a small RoPE rotation to the last axis, which must be even."""
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE needs an even last dimension")
    freqs = theta ** (-np.arange(0, dim, 2) / dim)
    angles = positions[:, None] * freqs[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out
# end::rope[]


# tag::cache-table[]
def cache_size_table(layers, tokens, bytes_per_value, heads, kv_heads,
                     head_dim, latent_dim):
    """Return cache bytes for MHA, GQA, and MLA."""
    return {
        "MHA": 2 * layers * heads * head_dim * tokens * bytes_per_value,
        "GQA": 2 * layers * kv_heads * head_dim * tokens * bytes_per_value,
        "MLA": layers * latent_dim * tokens * bytes_per_value,
    }
# end::cache-table[]


# tag::sparse[]
def top_k_sparse_attention(q, k, v, index_scores, k_top):
    """Attend only to keys selected by a cheap top-k indexer."""
    selected = np.argsort(index_scores, axis=-1)[:, -k_top:]
    mask = np.zeros(index_scores.shape, dtype=bool)
    rows = np.arange(index_scores.shape[0])[:, None]
    mask[rows, selected] = True
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, mask, weights
# end::sparse[]
