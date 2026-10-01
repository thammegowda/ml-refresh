"""Quantization and serving calculators (Chapter 40)."""
import numpy as np

__chapter__ = "quantization"


# tag::integer[]
def quantize_absmax(x, bits=8, axis=None):
    """Symmetric signed integer quantization with an absmax scale."""
    x = np.asarray(x, dtype=np.float32)
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int32)
    return q, q.astype(np.float32) * scale, scale


def quantize_zero_point(x, bits=8, axis=None):
    """Asymmetric integer quantization with a learned zero point."""
    x = np.asarray(x, dtype=np.float32)
    qmin, qmax = 0, 2 ** bits - 1
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    scale = (xmax - xmin) / max(qmax - qmin, 1)
    scale = np.where(scale == 0, 1.0, scale)
    zero = np.clip(np.round(qmin - xmin / scale), qmin, qmax)
    q = np.clip(np.round(x / scale + zero), qmin, qmax).astype(np.int32)
    return q, (q.astype(np.float32) - zero) * scale, scale, zero
# end::integer[]


# tag::groups[]
def quantize_absmax_groups(x, bits=4, group_size=16):
    """Absmax quantization with one scale per group along the last axis."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % group_size:
        raise ValueError("last dimension must be divisible by group_size")
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    q, dequant, scale = quantize_absmax(grouped, bits=bits, axis=-1)
    return q.reshape(x.shape), dequant.reshape(x.shape), scale


def mean_squared_error(x, y):
    return float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))
# end::groups[]


# tag::lowbit-float[]
def _quantize_float(x, mantissa_bits, min_exp, max_exp, max_value=None):
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)
    exponent = np.floor(np.log2(np.maximum(ax, 2.0 ** min_exp)))
    exponent = np.clip(exponent, min_exp, max_exp)
    step = 2.0 ** (exponent - mantissa_bits)
    rounded = np.round(ax / step) * step
    if max_value is None:
        max_value = (2.0 - 2.0 ** (-mantissa_bits)) * 2.0 ** max_exp
    return sign * np.minimum(rounded, max_value).astype(np.float32)


def fp8_e4m3(x):
    return _quantize_float(x, mantissa_bits=3, min_exp=-6, max_exp=8,
                           max_value=448.0)


def fp8_e5m2(x):
    return _quantize_float(x, mantissa_bits=2, min_exp=-14, max_exp=15)


def mxfp4(x, block_size=16):
    """Emulate MXFP4: shared block scale plus nearest E2M1-like code."""
    code = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    code = np.concatenate([-code[:0:-1], code])
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % block_size:
        raise ValueError("last dimension must be divisible by block_size")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scale = np.max(np.abs(blocks), axis=-1, keepdims=True) / 6.0
    scale = np.where(scale == 0, 1.0, scale)
    normalized = blocks / scale
    nearest = code[np.argmin(np.abs(normalized[..., None] - code), axis=-1)]
    return (nearest * scale).reshape(x.shape)
# end::lowbit-float[]


# tag::smoothquant[]
def smoothquant_migrate(activations, weights, alpha=0.5):
    """Move scale from activation columns into matching weight rows."""
    activations = np.asarray(activations, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    act = np.max(np.abs(activations), axis=0)
    weight = np.max(np.abs(weights), axis=1)
    scale = (act ** alpha) / np.maximum(weight, 1e-12) ** (1.0 - alpha)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    return activations / scale, weights * scale[:, None], scale
# end::smoothquant[]


# tag::gptq[]
def gptq_quantize_vector(w, hessian, bits=2, damping=1e-8):
    """Quantize coordinates and compensate later ones with H^{-1}."""
    w = np.asarray(w, dtype=np.float64)
    hessian = np.asarray(hessian, dtype=np.float64)
    inv_h = np.linalg.inv(hessian + damping * np.eye(len(w)))
    work = w.copy()
    quantized = np.zeros_like(work)
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(w)) / qmax
    for i in range(len(w)):
        qi = np.clip(np.round(work[i] / scale), -qmax, qmax) * scale
        error = work[i] - qi
        quantized[i] = qi
        if i + 1 < len(w):
            work[i + 1:] -= error * inv_h[i + 1:, i] / inv_h[i, i]
    return quantized


def reconstruction_loss(w, q, hessian):
    error = np.asarray(w) - np.asarray(q)
    return float(error @ hessian @ error)
# end::gptq[]


# tag::serving[]
def paged_block_table(lengths, block_size):
    """Map each sequence to physical KV-cache blocks."""
    tables, next_block = [], 0
    for length in lengths:
        count = int(np.ceil(length / block_size))
        tables.append(list(range(next_block, next_block + count)))
        next_block += count
    return tables


def decoding_intensity(parameters, weight_bytes=2, kv_bytes=0):
    """Approximate FLOPs per byte for one generated token."""
    flops = 2 * parameters
    bytes_read = weight_bytes * parameters + kv_bytes
    return flops / bytes_read


def roofline_tokens_per_second(parameters, bandwidth, peak_flops,
                               weight_bytes=2, kv_bytes=0):
    bytes_read = weight_bytes * parameters + kv_bytes
    flops = 2 * parameters
    return min(peak_flops / flops, bandwidth / bytes_read)
# end::serving[]
