import numpy as np

from distributions import bernoulli_pmf, gaussian_cdf, gaussian_interval, gaussian_log_pdf, gaussian_pdf
from expectation import monte_carlo, reparameterized_gradient, score_function_gradient
from joint import conditional_y_given_x, marginals, posterior
from mle import bernoulli_nll, categorical_nll, gaussian_mle, gaussian_nll, gaussian_regression_nll
from sampling import sample_exponential, sample_gumbel_max
from solutions import (average_mle_variance, laplace_regression_nll, sample_triangle,
                       score_function_gradient_with_baseline, variance_of_correlated_mean)
from scratch.gradcheck import check_gradient, numerical_gradient
from scratch.sampling import sample_categorical


def rng(seed=6):
    return np.random.default_rng(seed)


def test_densities_are_heights_and_areas_are_probabilities():
    assert bernoulli_pmf(np.array([0, 1]), 0.3).tolist() == [0.7, 0.3]
    np.testing.assert_allclose(gaussian_pdf(0.0, 0.0, 0.1), 3.9894, atol=1e-4)
    np.testing.assert_allclose(gaussian_interval(-0.05, 0.05, 0.0, 0.1), 0.3829, atol=1e-4)
    np.testing.assert_allclose([gaussian_interval(-1, 1), gaussian_interval(-2, 2)], [0.6827, 0.9545], atol=1e-4)
    x = np.linspace(-3, 4, 9)
    np.testing.assert_allclose(gaussian_log_pdf(x, 0.5, 1.5), np.log(gaussian_pdf(x, 0.5, 1.5)))
    grid = np.linspace(-10, 10, 200_001)
    np.testing.assert_allclose(np.trapezoid(gaussian_pdf(grid, 0.0, 0.1), grid), 1.0, atol=1e-9)
    np.testing.assert_allclose(gaussian_cdf(0.0), 0.5)


def test_marginals_conditionals_and_bayes():
    joint = np.array([[0.30, 0.10], [0.15, 0.45]])
    px, py = marginals(joint)
    np.testing.assert_allclose(px, [0.4, 0.6])
    np.testing.assert_allclose(py, [0.45, 0.55])
    conditional = conditional_y_given_x(joint)
    np.testing.assert_allclose(conditional.sum(axis=1), 1)
    np.testing.assert_allclose(conditional * px[:, None], joint)
    np.testing.assert_allclose(posterior(np.array([0.01, 0.99]), np.array([0.95, 0.05]))[0], 0.16102, atol=1e-5)
    np.testing.assert_allclose(posterior(np.array([0.2, 0.8]), np.array([0.95, 0.05]))[0], 0.82609, atol=1e-5)


def test_monte_carlo_estimate_and_standard_error():
    estimate, error = monte_carlo(lambda x: x ** 2, lambda n, r: r.standard_normal(n), 10_000, rng(1))
    assert abs(estimate - 1) < 3 * error
    np.testing.assert_allclose(error, np.sqrt(2 / 10_000), rtol=0.05)   # Var[X^2] = 2 for N(0, 1)


def test_variance_of_a_mean_shrinks_like_one_over_n_unless_draws_are_correlated():
    r = rng()
    means = r.standard_normal((20_000, 16)).mean(axis=1)
    np.testing.assert_allclose(means.var(), 1 / 16, rtol=0.05)
    np.testing.assert_allclose(variance_of_correlated_mean(32, 0.1, 200_000, r), 0.1 + 0.9 / 32, rtol=0.03)


def test_score_function_and_reparameterized_gradients_are_unbiased_with_known_variances():
    f, df = (lambda x: x ** 2), (lambda x: 2 * x)
    r = rng()
    trials = 400
    score = np.array([score_function_gradient(f, 1.0, 1.0, 1000, r) for _ in range(trials)])
    baseline = np.array([score_function_gradient_with_baseline(f, 1.0, 1.0, 1000, r, 2.0) for _ in range(trials)])
    reparam = np.array([reparameterized_gradient(df, 1.0, 1.0, 1000, r) for _ in range(trials)])
    for estimates, per_sample_variance in [(score, 30), (baseline, 18), (reparam, 4)]:
        standard_error = np.sqrt(per_sample_variance / 1000)
        assert abs(estimates.mean() - 2.0) < 4 * standard_error / np.sqrt(trials)
        np.testing.assert_allclose(estimates.std(), standard_error, rtol=0.12)


def test_gaussian_mle_is_a_stationary_point_of_the_nll_and_its_variance_is_biased():
    x = rng().normal(1.5, 0.7, size=500)
    mean, std = gaussian_mle(x)
    gradient = numerical_gradient(lambda theta: gaussian_nll(x, theta[0], theta[1]), [mean, std])
    np.testing.assert_allclose(gradient, 0, atol=1e-7)
    assert gaussian_nll(x, mean, std) < min(gaussian_nll(x, mean + 0.05, std), gaussian_nll(x, mean, std * 1.05))
    np.testing.assert_allclose(average_mle_variance(5, 200_000, rng()), 0.8, rtol=0.01)


def test_familiar_losses_are_negative_log_likelihoods():
    r = rng()
    y, prediction = r.standard_normal(50), r.standard_normal(50)
    mse = np.mean((y - prediction) ** 2)
    np.testing.assert_allclose(gaussian_regression_nll(y, prediction, 2.0), mse / 8 + np.log(2.0) + 0.5 * np.log(2 * np.pi))
    np.testing.assert_allclose(laplace_regression_nll(y, prediction, 1.0), np.mean(np.abs(y - prediction)) + np.log(2))
    labels, p = np.array([1, 0, 1, 1]), np.array([0.9, 0.2, 0.6, 0.99])
    np.testing.assert_allclose(bernoulli_nll(labels, p), -np.mean(np.log([0.9, 0.8, 0.6, 0.99])))
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(categorical_nll(np.array([0, 2]), probabilities), -np.log(0.7 * 0.8) / 2)
    check_gradient(lambda q: bernoulli_nll(labels, q), p, (p - labels) / (p * (1 - p)) / len(p))


def test_samplers_match_their_target_distributions():
    r = rng()
    x = sample_exponential(2.0, 200_000, r)
    np.testing.assert_allclose([x.mean(), np.mean(x > 1.0)], [0.5, np.exp(-2.0)], rtol=0.02)
    t = sample_triangle(200_000, r)
    np.testing.assert_allclose([t.mean(), np.mean(t < 0.5)], [2 / 3, 0.25], rtol=0.01)
    probabilities = np.array([0.5, 0.0, 0.2, 0.3])
    for draws in [sample_categorical(np.broadcast_to(probabilities, (200_000, 4)), r),
                  sample_gumbel_max(np.broadcast_to(np.log(np.maximum(probabilities, 1e-300)), (200_000, 4)), r)]:
        frequencies = np.bincount(draws, minlength=4) / len(draws)
        np.testing.assert_allclose(frequencies, probabilities, atol=0.004)
        assert frequencies[1] == 0


def test_temperature_scales_logits_before_the_noise():
    logits = np.array([2.0, 1.0, 0.0])
    r = rng()
    for temperature in (0.5, 2.0):
        target = np.exp(logits / temperature) / np.exp(logits / temperature).sum()
        draws = sample_gumbel_max(np.broadcast_to(logits / temperature, (200_000, 3)), r)
        np.testing.assert_allclose(np.bincount(draws, minlength=3) / len(draws), target, atol=0.004)


def test_batched_categorical_sampling_draws_one_index_per_row():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert sample_categorical(probabilities, rng()).tolist() == [0, 2]
    assert sample_categorical(np.full((2, 5, 7), 1 / 7), rng()).shape == (2, 5)
