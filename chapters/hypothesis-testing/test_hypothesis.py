import numpy as np

from compare import (exact_permutation_p_value, mcnemar_exact_p_value,
                     paired_bootstrap_delta)
from intervals import (accuracy_sample_size, accuracy_standard_error, bootstrap_ci,
                       normal_accuracy_ci)
from solutions import median_interval, paired_demo


def rng(seed=8):
    return np.random.default_rng(seed)


def correctness(count, total):
    return np.array([1] * count + [0] * (total - count), dtype=np.int64)


def test_accuracy_standard_error_normal_ci_and_sample_size_numbers():
    correct = correctness(248, 400)
    np.testing.assert_allclose(correct.mean(), 0.62)
    np.testing.assert_allclose(accuracy_standard_error(correct), 0.024269, atol=1e-6)
    lo, hi = normal_accuracy_ci(correct)
    np.testing.assert_allclose([lo, hi], [0.572432, 0.667568], atol=1e-6)
    assert accuracy_sample_size(0.5, 0.02) == 2401
    assert accuracy_sample_size(0.7, 0.02) == 2017


def test_bootstrap_interval_is_seeded_and_contains_the_statistic():
    scores = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.float64)
    interval = bootstrap_ci(scores, np.mean, draws=2000, rng=rng())
    np.testing.assert_allclose(interval, [0.25, 1.0])
    assert interval[0] <= scores.mean() <= interval[1]
    np.testing.assert_allclose(median_interval(scores, rng()), [0.0, 1.0])


def test_exact_permutation_and_mcnemar_match_for_the_tiny_demo():
    result = paired_demo()
    np.testing.assert_allclose(result["delta"], 0.25)
    np.testing.assert_allclose(result["permutation_p"], 0.375)
    np.testing.assert_allclose(result["mcnemar_p"], 0.375)


def test_paired_bootstrap_tracks_the_mean_of_per_example_deltas():
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    b = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.int64)
    delta, interval = paired_bootstrap_delta(a, b, draws=3000, rng=rng())
    np.testing.assert_allclose(delta, (b - a).mean())
    np.testing.assert_allclose(delta, 0.2)
    np.testing.assert_allclose(interval, [-0.2, 0.6])
    assert interval[0] <= delta <= interval[1]


def test_mcnemar_edge_cases_and_asymmetry_counts():
    same = np.array([1, 0, 1, 0], dtype=np.int64)
    assert mcnemar_exact_p_value(same, same) == 1.0
    a = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    b = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    np.testing.assert_allclose(mcnemar_exact_p_value(a, b), 1.0)
    assert exact_permutation_p_value(a, b) == mcnemar_exact_p_value(a, b)
