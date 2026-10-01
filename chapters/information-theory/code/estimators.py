"""Estimating KL divergence from samples (Chapter 7)."""
import numpy as np


# tag::estimators[]
def kl_estimators(log_p, log_q):
    """Per-sample estimates of KL(q || p) from draws x ~ q.

    Each argument holds log p(x) or log q(x) at the same draws. With r = p(x) / q(x):
    k1 = -log r, k2 = (log r)^2 / 2, and k3 = (r - 1) - log r.
    """
    log_r = log_p - log_q
    k1 = -log_r
    k2 = 0.5 * log_r ** 2
    k3 = np.expm1(log_r) - log_r
    return k1, k2, k3
# end::estimators[]
