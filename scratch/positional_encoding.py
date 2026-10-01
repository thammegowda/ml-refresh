"""Position encodings and RoPE utilities for Chapter 21."""
import numpy as np

__chapter__ = "positional-encoding"


def softmax(scores):
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


# tag::attention[]
def attention(q, k, v):
    """Scaled dot-product attention without any positional information."""
    scale = np.sqrt(q.shape[-1])
    scores = q @ np.swapaxes(k, -1, -2) / scale
    return softmax(scores) @ v
# end::attention[]


# tag::absolute[]
def sinusoidal_positions(length, dim, base=10_000.0):
    """The original fixed absolute table: sin on even dims, cos on odd dims."""
    if dim % 2:
        raise ValueError("sinusoidal positions need an even dimension")
    positions = np.arange(length, dtype=np.float64)[:, None]
    theta = base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles = positions * theta[None, :]
    table = np.empty((length, dim), dtype=np.float64)
    table[:, 0::2] = np.sin(angles)
    table[:, 1::2] = np.cos(angles)
    return table


def learned_positions(length, dim, rng, scale=0.02):
    """A learned absolute position table, initialized like a small embedding."""
    return rng.normal(0.0, scale, size=(length, dim)).astype(np.float32)
# end::absolute[]


def rope_frequencies(dim, base=10_000.0):
    if dim % 2:
        raise ValueError("RoPE needs an even dimension")
    pair = np.arange(dim // 2, dtype=np.float64)
    return base ** (-2.0 * pair / dim)


# tag::rope[]
def apply_rope(x, positions, base=10_000.0, inverse=False):
    """Rotate each adjacent 2-D pair by positions * theta_i."""
    x = np.asarray(x)
    if x.shape[-1] % 2:
        raise ValueError("RoPE needs an even last dimension")
    positions = np.asarray(positions, dtype=x.dtype)
    if positions.ndim == 1 and x.ndim > 2:
        shape = [1] * (x.ndim - 1)
        shape[1] = positions.size
        positions = positions.reshape(shape)
    theta = rope_frequencies(x.shape[-1], base).astype(x.dtype)
    angles = positions[..., None] * theta
    if inverse:
        angles = -angles
    cos, sin = np.cos(angles), np.sin(angles)
    y = np.empty_like(x)
    even, odd = x[..., 0::2], x[..., 1::2]
    y[..., 0::2] = even * cos - odd * sin
    y[..., 1::2] = even * sin + odd * cos
    return y
# end::rope[]


# tag::alibi[]
def alibi_bias(length, slope):
    """Causal ALiBi scores: query t gets -slope * (t - s) for key s <= t."""
    t = np.arange(length)[:, None]
    s = np.arange(length)[None, :]
    return -float(slope) * np.maximum(t - s, 0)
# end::alibi[]
