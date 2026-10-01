"""Entropy, cross-entropy, and KL divergence of discrete distributions (Chapter 7)."""
import numpy as np

__chapter__ = "information-theory"


# tag::information[]
def entropy(p, axis=-1):
    """H(p) = -sum p log p, in nats, with the convention 0 log 0 = 0."""
    p = np.asarray(p, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)                  # log(1) = 0 where p = 0
    return -np.sum(p * np.log(safe), axis=axis)


def cross_entropy(p, q, axis=-1):
    """H(p, q) = -sum p log q: infinite when q = 0 somewhere that p > 0."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_q = np.where(p > 0, np.log(q), 0.0)
    return -np.sum(p * log_q, axis=axis)


def kl_divergence(p, q, axis=-1):
    """KL(p || q) = sum p log(p / q), computed directly, not as H(p, q) - H(p)."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_ratio = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) - np.log(q), 0.0)
    return np.sum(p * log_ratio, axis=axis)
# end::information[]
