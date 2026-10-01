"""Intervals and sample sizes for benchmark accuracies (Chapter 8)."""
import numpy as np


# tag::accuracy-ci[]
def accuracy_standard_error(correct):
    """Standard error of the mean of 0/1 correctness indicators."""
    correct = np.asarray(correct, dtype=np.float64)
    p_hat = correct.mean()
    return np.sqrt(p_hat * (1.0 - p_hat) / correct.size)


def normal_accuracy_ci(correct, z=1.96):
    """Normal-approximation confidence interval for an accuracy."""
    correct = np.asarray(correct, dtype=np.float64)
    p_hat = correct.mean()
    half_width = z * accuracy_standard_error(correct)
    return p_hat - half_width, p_hat + half_width
# end::accuracy-ci[]


# tag::bootstrap-ci[]
def bootstrap_ci(values, statistic=np.mean, draws=2000, level=0.95, rng=None):
    """Percentile bootstrap interval for any statistic of examples."""
    values = np.asarray(values)
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(values)
    stats = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = values[rng.integers(0, n, size=n)]
        stats[draw] = statistic(sample)
    alpha = (1.0 - level) / 2.0
    return tuple(np.quantile(stats, [alpha, 1.0 - alpha]))
# end::bootstrap-ci[]


# tag::sample-size[]
def accuracy_sample_size(p, half_width, z=1.96):
    """Examples needed so a normal CI has about the requested half-width."""
    return int(np.ceil(z * z * p * (1.0 - p) / (half_width * half_width)))
# end::sample-size[]
