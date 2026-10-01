import numpy as np

from scaling_laws import (
    CHINCHILLA,
    chinchilla_loss,
    chinchilla_optimal_allocation,
    chinchilla_rule_tokens,
    fit_power_law,
    transformer_training_compute,
)
from solutions import best_on_grid, warmup_stable_decay


def test_training_compute_splits_into_forward_and_backward_terms():
    total, forward, backward = transformer_training_compute(7, 11)
    assert (forward, backward, total) == (154, 308, 462)
    assert total == 6 * 7 * 11


def test_power_law_fit_recovers_coefficient_and_exponent():
    x = np.array([1, 2, 4, 8, 16], dtype=np.float64)
    y = 3.5 * x ** -0.27
    coefficient, exponent = fit_power_law(x, y)
    np.testing.assert_allclose([coefficient, exponent], [3.5, 0.27], rtol=1e-12)


def test_chinchilla_constants_match_approach_three_values():
    assert CHINCHILLA == {
        "E": 1.69,
        "A": 406.4,
        "B": 410.7,
        "alpha": 0.34,
        "beta": 0.28,
    }
    loss = chinchilla_loss(70e9, 1.4e12)
    expected = 1.69 + 406.4 / (70e9 ** 0.34) + 410.7 / (1.4e12 ** 0.28)
    np.testing.assert_allclose(loss, expected)


def test_chinchilla_optimum_satisfies_compute_constraint_and_stationarity():
    compute = 6 * 1e9 * 20e9
    parameters, tokens = chinchilla_optimal_allocation(compute)
    np.testing.assert_allclose(6 * parameters * tokens, compute)
    a, b = CHINCHILLA["alpha"], CHINCHILLA["beta"]
    A, B = CHINCHILLA["A"], CHINCHILLA["B"]
    left = a * A * parameters ** (-a)
    right = b * B * tokens ** (-b)
    np.testing.assert_allclose(left, right, rtol=1e-12)


def test_twenty_tokens_per_parameter_rule_and_grid_search():
    assert chinchilla_rule_tokens(70e9) == 1.4e12
    grid_n = np.array([0.5e9, 1e9, 2e9])
    grid_d = np.array([10e9, 20e9, 40e9])
    loss, parameters, tokens = best_on_grid(6 * 1e9 * 20e9, grid_n, grid_d)
    assert 6 * parameters * tokens <= 6 * 1e9 * 20e9
    for candidate_n in grid_n:
        for candidate_d in grid_d:
            if 6 * candidate_n * candidate_d <= 6 * 1e9 * 20e9:
                assert loss <= chinchilla_loss(candidate_n, candidate_d)


def test_warmup_stable_decay_schedule_shape():
    values = [warmup_stable_decay(step, warmup=10, stable=80, total=100)
              for step in [0, 5, 10, 79, 80, 90, 100]]
    np.testing.assert_allclose(values, [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0])
