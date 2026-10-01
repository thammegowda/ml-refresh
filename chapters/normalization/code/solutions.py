import numpy as np

from scratch.normalization import inverted_dropout, rms_norm_forward


# tag::rms-invariance[]
def rms_scale_invariance(x, weight, factor):
    y1, _ = rms_norm_forward(x, weight)
    y2, _ = rms_norm_forward(factor * x, weight)
    return np.max(np.abs(y1 - y2))
# end::rms-invariance[]


# tag::dropout-mean[]
def dropout_mean(seed=0):
    rng = np.random.default_rng(seed)
    x = np.ones(20_000)
    y, _ = inverted_dropout(x, 0.25, rng)
    return float(y.mean())
# end::dropout-mean[]
