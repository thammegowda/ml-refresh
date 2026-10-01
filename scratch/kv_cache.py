"""KV-cache and grouped-query attention helpers."""
import numpy as np

__chapter__ = "kv-cache"


# tag::softmax[]
def softmax(x, axis=-1):
    """Stable softmax along one axis."""
    x = np.asarray(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)
# end::softmax[]


# tag::masks[]
def causal_mask(tokens):
    """True where a query position may read a key position."""
    pos = np.arange(tokens)
    return pos[None, :] <= pos[:, None]


def sliding_window_mask(tokens, window, sinks=0):
    """Causal local attention, optionally keeping early sink tokens visible."""
    pos = np.arange(tokens)
    causal = pos[None, :] <= pos[:, None]
    recent = pos[None, :] >= pos[:, None] - window + 1
    sink = pos[None, :] < sinks
    return causal & (recent | sink)
# end::masks[]


# tag::gqa[]
def expand_kv_heads(kv, num_query_heads):
    """Repeat each KV head so it serves a group of query heads."""
    num_kv_heads = kv.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    repeats = num_query_heads // num_kv_heads
    return np.repeat(kv, repeats, axis=1)


def gqa_attention(q, k, v, mask=None):
    """Scaled dot-product attention with H query heads and G KV heads."""
    k_heads = expand_kv_heads(k, q.shape[1])
    v_heads = expand_kv_heads(v, q.shape[1])
    scale = np.sqrt(q.shape[-1])
    scores = np.einsum("thd,shd->hts", q, k_heads) / scale
    if mask is not None:
        scores = np.where(mask[None, :, :], scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return np.einsum("hts,shd->thd", weights, v_heads), weights
# end::gqa[]


# tag::cache[]
def project_heads(x, weight):
    """Project token states to per-head vectors."""
    return np.einsum("td,dhr->thr", x, weight)


def causal_self_attention(x, w_q, w_k, w_v, w_o):
    """Full prefill: project all tokens, then apply one causal attention."""
    q = project_heads(x, w_q)
    k = project_heads(x, w_k)
    v = project_heads(x, w_v)
    heads, _ = gqa_attention(q, k, v, causal_mask(x.shape[0]))
    return np.einsum("thr,hro->to", heads, w_o)


def incremental_self_attention(x, w_q, w_k, w_v, w_o):
    """Decode one token at a time, reusing cached keys and values."""
    q = project_heads(x, w_q)
    k = project_heads(x, w_k)
    v = project_heads(x, w_v)
    outs = []
    for t in range(x.shape[0]):
        heads, _ = gqa_attention(q[t:t + 1], k[:t + 1], v[:t + 1])
        outs.append(np.einsum("thr,hro->to", heads, w_o)[0])
    return np.stack(outs)
# end::cache[]


# tag::memory[]
def kv_cache_bytes(layers, num_kv_heads, head_dim, tokens, bytes_per_value):
    """Bytes for K and V caches for one sequence."""
    return 2 * layers * num_kv_heads * head_dim * tokens * bytes_per_value
# end::memory[]


# tag::qk-norm[]
def l2_normalize_last_axis(x, eps=1e-6):
    """Normalize vectors before dot products, as in QK-norm."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)
# end::qk-norm[]
