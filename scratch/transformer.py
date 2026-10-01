"""A tiny pre-norm decoder block with manual backpropagation."""
import numpy as np

from scratch.positional_encoding import apply_rope, softmax

__chapter__ = "transformer"


def init_block_params(d_model, n_heads, hidden_dim, rng, dtype=np.float32):
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    scale = 1.0 / np.sqrt(d_model)

    def normal(shape):
        return rng.normal(0.0, scale, size=shape).astype(dtype)

    return {
        "attn_norm": np.ones(d_model, dtype=dtype),
        "ffn_norm": np.ones(d_model, dtype=dtype),
        "Wq": normal((d_model, d_model)),
        "Wk": normal((d_model, d_model)),
        "Wv": normal((d_model, d_model)),
        "Wo": normal((d_model, d_model)),
        "W_gate": normal((d_model, hidden_dim)),
        "W_up": normal((d_model, hidden_dim)),
        "W_down": normal((hidden_dim, d_model)),
    }


def zeros_like_params(params):
    if isinstance(params, dict):
        return {key: zeros_like_params(value) for key, value in params.items()}
    if isinstance(params, list):
        return [zeros_like_params(value) for value in params]
    return np.zeros_like(params)


def add_grads(left, right):
    for key, value in right.items():
        left[key] = left.get(key, 0) + value
    return left


# tag::rmsnorm[]
def rmsnorm_forward(x, weight, eps=1e-5):
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_square + eps)
    normalized = x * inv_rms
    return normalized * weight, (x, weight, inv_rms, normalized)


def rmsnorm_backward(dout, cache):
    x, weight, inv_rms, normalized = cache
    dnormalized = dout * weight
    scale_grad = np.sum(dnormalized * x, axis=-1, keepdims=True)
    width = x.shape[-1]
    dx = inv_rms * dnormalized - x * inv_rms ** 3 * scale_grad / width
    dweight = np.sum(dout * normalized, axis=tuple(range(dout.ndim - 1)))
    return dx, dweight
# end::rmsnorm[]


# tag::attention[]
def causal_self_attention_forward(x, params, n_heads, rope_base=10_000.0):
    batch, length, d_model = x.shape
    head_dim = d_model // n_heads
    q = (x @ params["Wq"]).reshape(batch, length, n_heads, head_dim)
    k = (x @ params["Wk"]).reshape(batch, length, n_heads, head_dim)
    v = (x @ params["Wv"]).reshape(batch, length, n_heads, head_dim)
    positions = np.arange(length, dtype=x.dtype)
    q_rot = apply_rope(q, positions, base=rope_base)
    k_rot = apply_rope(k, positions, base=rope_base)
    scores = np.einsum("bthd,bshd->bhts", q_rot, k_rot) / np.sqrt(head_dim)
    mask = np.triu(np.ones((length, length), dtype=bool), k=1)
    weights = softmax(np.where(mask[None, None], -1e9, scores))
    context = np.einsum("bhts,bshd->bthd", weights, v)
    flat = context.reshape(batch, length, d_model)
    out = flat @ params["Wo"]
    cache = (x, params, n_heads, q_rot, k_rot, v, weights, context, mask)
    return out, cache
# end::attention[]


def causal_self_attention_backward(dout, cache):
    x, params, n_heads, q_rot, k_rot, v, weights, context, mask = cache
    batch, length, d_model = x.shape
    head_dim = d_model // n_heads
    flat_context = context.reshape(batch * length, d_model)
    flat_dout = dout.reshape(batch * length, d_model)
    grads = {"Wo": flat_context.T @ flat_dout}
    dcontext = (flat_dout @ params["Wo"].T).reshape(context.shape)
    dweights = np.einsum("bthd,bshd->bhts", dcontext, v)
    dv = np.einsum("bhts,bthd->bshd", weights, dcontext)
    centered = dweights - np.sum(dweights * weights, axis=-1, keepdims=True)
    dscores = weights * centered / np.sqrt(head_dim)
    dscores = np.where(mask[None, None], 0.0, dscores)
    dq_rot = np.einsum("bhts,bshd->bthd", dscores, k_rot)
    dk_rot = np.einsum("bhts,bthd->bshd", dscores, q_rot)
    positions = np.arange(length, dtype=x.dtype)
    dq = apply_rope(dq_rot, positions, inverse=True).reshape(batch, length, d_model)
    dk = apply_rope(dk_rot, positions, inverse=True).reshape(batch, length, d_model)
    dv = dv.reshape(batch, length, d_model)
    flat_x = x.reshape(batch * length, d_model)
    for name, delta in [("Wq", dq), ("Wk", dk), ("Wv", dv)]:
        flat_delta = delta.reshape(batch * length, d_model)
        grads[name] = flat_x.T @ flat_delta
    dx = dq @ params["Wq"].T + dk @ params["Wk"].T + dv @ params["Wv"].T
    return dx, grads


def silu(x):
    return x / (1.0 + np.exp(-x))


# tag::swiglu[]
def swiglu_forward(x, params):
    gate = x @ params["W_gate"]
    up = x @ params["W_up"]
    hidden = silu(gate) * up
    out = hidden @ params["W_down"]
    return out, (x, params, gate, up, hidden)
# end::swiglu[]


def swiglu_backward(dout, cache):
    x, params, gate, up, hidden = cache
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_dout = dout.reshape(-1, dout.shape[-1])
    grads = {"W_down": flat_hidden.T @ flat_dout}
    dhidden = (flat_dout @ params["W_down"].T).reshape(hidden.shape)
    sigma = 1.0 / (1.0 + np.exp(-gate))
    dsilu = sigma * (1.0 + gate * (1.0 - sigma))
    dgate = dhidden * up * dsilu
    dup = dhidden * silu(gate)
    flat_x = x.reshape(-1, x.shape[-1])
    grads["W_gate"] = flat_x.T @ dgate.reshape(-1, dgate.shape[-1])
    grads["W_up"] = flat_x.T @ dup.reshape(-1, dup.shape[-1])
    dx = dgate @ params["W_gate"].T + dup @ params["W_up"].T
    return dx, grads


# tag::block[]
def transformer_block_forward(x, params, n_heads, rope_base=10_000.0):
    attn_in, norm1 = rmsnorm_forward(x, params["attn_norm"])
    attn_out, attn_cache = causal_self_attention_forward(
        attn_in, params, n_heads, rope_base
    )
    residual = x + attn_out
    ffn_in, norm2 = rmsnorm_forward(residual, params["ffn_norm"])
    ffn_out, ffn_cache = swiglu_forward(ffn_in, params)
    out = residual + ffn_out
    return out, (norm1, attn_cache, residual, norm2, ffn_cache)
# end::block[]


def transformer_block_backward(dout, cache):
    norm1, attn_cache, _residual, norm2, ffn_cache = cache
    grads = {}
    dffn_in, ffn_grads = swiglu_backward(dout, ffn_cache)
    add_grads(grads, ffn_grads)
    dresidual_from_ffn, grads["ffn_norm"] = rmsnorm_backward(dffn_in, norm2)
    dresidual = dout + dresidual_from_ffn
    dattn_in, attn_grads = causal_self_attention_backward(dresidual, attn_cache)
    add_grads(grads, attn_grads)
    dx_from_attn, grads["attn_norm"] = rmsnorm_backward(dattn_in, norm1)
    dx = dresidual + dx_from_attn
    return dx, grads
