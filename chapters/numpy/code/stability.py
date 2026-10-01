"""Numerically stable building blocks (Appendix B)."""
import numpy as np


def naive_logsumexp(z, axis=-1):
    return np.log(np.sum(np.exp(z), axis=axis))


# tag::logsumexp[]
def logsumexp(z, axis=-1, keepdims=False):
    """log(sum(exp(z))) along `axis`, computed without overflow."""
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0)        # all -inf rows: log(0) = -inf, not nan
    result = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def log_softmax(z, axis=-1):
    return z - logsumexp(z, axis=axis, keepdims=True)
# end::logsumexp[]


# tag::sigmoid[]
def sigmoid(x):
    """1 / (1 + exp(-x)) that never exponentiates a large positive number."""
    e = np.exp(-np.abs(x))                        # in (0, 1] for every x
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def softplus(x):
    """log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|))."""
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
# end::sigmoid[]
