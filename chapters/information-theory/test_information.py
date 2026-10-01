import numpy as np

from estimators import kl_estimators
from fitting import GRID, STEP, fit_gaussian, gaussian_density, kl_on_grid, two_modes
from gaussians import gaussian_kl
from sequences import bits_per_token, mutual_information, mutual_information_from_entropies, perplexity
from solutions import average_nll, cross_entropy_of_empirical, surprisal_bits
from scratch.information import cross_entropy, entropy, kl_divergence

BITS = np.log(2)


def rng(seed=7):
    return np.random.default_rng(seed)


def test_surprisal_and_entropy_of_simple_distributions():
    assert [round(surprisal_bits(p), 3) for p in (1 / 2, 1 / 6, 1 / 128_000)] == [1.0, 2.585, 16.966]
    np.testing.assert_allclose(entropy([0.5, 0.5]) / BITS, 1.0)
    np.testing.assert_allclose(entropy(np.full(6, 1 / 6)), np.log(6))
    assert entropy([1.0, 0.0, 0.0]) == 0.0
    dyadic, uniform = np.array([0.5, 0.25, 0.125, 0.125]), np.full(4, 0.25)
    np.testing.assert_allclose([entropy(dyadic) / BITS, cross_entropy(dyadic, uniform) / BITS, kl_divergence(dyadic, uniform) / BITS], [1.75, 2.0, 0.25])


def test_entropy_is_largest_for_the_uniform_distribution():
    for p in rng().dirichlet(np.ones(8), size=200):
        assert 0 <= entropy(p) <= np.log(8) + 1e-12
        np.testing.assert_allclose(np.log(8) - entropy(p), kl_divergence(p, np.full(8, 1 / 8)))


def test_kl_is_nonnegative_asymmetric_and_the_gap_between_cross_entropy_and_entropy():
    r = rng()
    p, q = r.dirichlet(np.ones(5), size=500), r.dirichlet(np.ones(5), size=500)
    divergence = kl_divergence(p, q)
    assert np.all(divergence >= 0)
    np.testing.assert_allclose(cross_entropy(p, q), entropy(p) + divergence)
    np.testing.assert_allclose(kl_divergence(p, p), 0, atol=1e-15)
    assert not np.allclose(divergence, kl_divergence(q, p))
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == np.inf
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == np.log(2)


def test_minimizing_cross_entropy_is_maximum_likelihood():
    r = rng()
    samples = r.choice(4, size=1000, p=[0.1, 0.2, 0.3, 0.4])
    q = r.dirichlet(np.ones(4))
    np.testing.assert_allclose(average_nll(samples, q), cross_entropy_of_empirical(samples, q))


def test_gaussian_kl_matches_numerical_integration():
    for parameters in [(0, 1, 0, 1), (0.5, 0.8, -1, 1.5), (2, 0.3, 0, 1)]:
        p, q = gaussian_density(GRID, *parameters[:2]), gaussian_density(GRID, *parameters[2:])
        np.testing.assert_allclose(gaussian_kl(*parameters), kl_on_grid(p, q), atol=1e-6)
    assert gaussian_kl(0, 1, 0, 1) == 0.0


def test_forward_kl_covers_both_modes_and_reverse_kl_picks_one():
    means, stds = np.arange(-4, 4.001, 0.1), np.arange(0.2, 4.001, 0.05)
    forward_mean, forward_std = fit_gaussian(two_modes, "forward", means, stds)
    reverse_mean, reverse_std = fit_gaussian(two_modes, "reverse", means, stds)
    assert abs(forward_mean) < 1e-9 and abs(forward_std - np.sqrt(0.6 ** 2 + 2 ** 2)) < 0.03
    assert abs(abs(reverse_mean) - 2) < 1e-9 and abs(reverse_std - 0.6) < 0.03
    p = two_modes(GRID)
    np.testing.assert_allclose([np.sum(p) * STEP, np.sum(GRID ** 2 * p) * STEP], [1, 4.36], rtol=1e-6)


def test_kl_estimators_bias_and_spread():
    x = rng().standard_normal(1_000_000)
    for shift, (k1_spread, k2_bias, k3_spread) in [(0.1, (20.0, 1.0, 1.42)), (1.0, (2.0, 1.25, 1.71))]:
        k1, k2, k3 = kl_estimators(-0.5 * (x - shift) ** 2, -0.5 * x ** 2)
        true = gaussian_kl(0, 1, shift, 1)
        np.testing.assert_allclose([k1.mean() / true, k3.mean() / true], 1, atol=0.02)
        np.testing.assert_allclose(k2.mean() / true, k2_bias, atol=0.02)
        np.testing.assert_allclose([k1.std() / true, k3.std() / true], [k1_spread, k3_spread], rtol=0.03)
        assert np.all(k3 >= 0) and np.any(k1 < 0)


def test_perplexity_and_bits_per_token():
    assert round(perplexity(np.full(10, np.log(1 / 50_000)))) == 50_000
    assert perplexity(np.zeros(3)) == 1.0
    np.testing.assert_allclose(bits_per_token(np.log([0.5, 0.25])), 1.5)
    np.testing.assert_allclose(perplexity(np.log([0.5, 0.25])), 2 ** 1.5)


def test_mutual_information_two_ways():
    joint = np.array([[0.30, 0.10], [0.15, 0.45]])
    np.testing.assert_allclose(mutual_information(joint), mutual_information_from_entropies(joint))
    np.testing.assert_allclose(mutual_information(joint), 0.1258, atol=1e-4)
    assert abs(mutual_information(np.outer([0.4, 0.6], [0.45, 0.55]))) < 1e-15
    identical = np.diag([0.2, 0.3, 0.5])
    np.testing.assert_allclose(mutual_information(identical), entropy([0.2, 0.3, 0.5]))
