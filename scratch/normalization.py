"""Normalization layers and dropout in NumPy (Chapter 16)."""
import numpy as np

__chapter__ = "normalization"


# tag::layernorm[]
def layer_norm_forward(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(variance + eps)
    normalized = centered * inv_std
    out = normalized * gamma + beta
    cache = (normalized, inv_std, gamma)
    return out, cache


def layer_norm_backward(dout, cache):
    normalized, inv_std, gamma = cache
    axes = tuple(range(dout.ndim - 1))
    grad_gamma = np.sum(dout * normalized, axis=axes)
    grad_beta = np.sum(dout, axis=axes)
    grad_norm = dout * gamma
    width = dout.shape[-1]
    sum_grad = np.sum(grad_norm, axis=-1, keepdims=True)
    sum_grad_norm = np.sum(grad_norm * normalized, axis=-1, keepdims=True)
    dx = inv_std * (grad_norm - sum_grad / width
                    - normalized * sum_grad_norm / width)
    return dx, grad_gamma, grad_beta
# end::layernorm[]


# tag::rmsnorm[]
def rms_norm_forward(x, weight, eps=1e-8):
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_square + eps)
    normalized = x * inv_rms
    out = normalized * weight
    cache = (x, normalized, inv_rms, weight)
    return out, cache


def rms_norm_backward(dout, cache):
    x, normalized, inv_rms, weight = cache
    axes = tuple(range(dout.ndim - 1))
    grad_weight = np.sum(dout * normalized, axis=axes)
    grad_norm = dout * weight
    width = dout.shape[-1]
    dot = np.sum(grad_norm * x, axis=-1, keepdims=True)
    dx = grad_norm * inv_rms - x * (inv_rms ** 3) * dot / width
    return dx, grad_weight
# end::rmsnorm[]


# tag::batchnorm-dropout[]
def batch_norm_forward(x, gamma, beta, running_mean, running_var, training,
                       momentum=0.9, eps=1e-5):
    if training:
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        running_mean *= momentum
        running_mean += (1 - momentum) * mean
        running_var *= momentum
        running_var += (1 - momentum) * var
    else:
        mean = running_mean
        var = running_var
    normalized = (x - mean) / np.sqrt(var + eps)
    return normalized * gamma + beta


def inverted_dropout(x, drop_probability, rng):
    if not 0 <= drop_probability < 1:
        raise ValueError("drop_probability must be in [0, 1)")
    keep = rng.random(x.shape) >= drop_probability
    return x * keep / (1 - drop_probability), keep
# end::batchnorm-dropout[]
