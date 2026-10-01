import numpy as np

from quadratic import gradient_descent, quadratic_gradient, quadratic_value
from solutions import adam_first_step, clipped_demo_norm, optimal_gd_rate
from scratch.gradcheck import check_gradient
from scratch.optimization import (adam_state, adamw_step, clip_by_global_norm,
                                  cosine_decay, heavy_ball_step, momentum_state,
                                  muon_update,
                                  orthogonalize_newton_schulz, rmsprop_state,
                                  rmsprop_step, wsd_schedule)


def test_quadratic_gradient_and_condition_number_rate():
    curvature = np.array([1.0, 25.0])
    theta = np.array([1.2, -0.4])
    check_gradient(lambda x: quadratic_value(x, curvature), theta,
                   quadratic_gradient(theta, curvature))
    path = gradient_descent(np.array([1.0, 1.0]), curvature, lr=1 / 25, steps=20)
    assert abs(path[-1, 1]) < 1e-12
    np.testing.assert_allclose(path[-1, 0], (1 - 1 / 25) ** 20)
    assert optimal_gd_rate(25) == 24 / 26


def test_heavy_ball_momentum_accumulates_velocity():
    params = {"w": np.array([1.0])}
    grads = {"w": np.array([0.5])}
    state = momentum_state(params)
    heavy_ball_step(params, grads, state, lr=0.1, beta=0.9)
    np.testing.assert_allclose(state["velocity"]["w"], [-0.05])
    np.testing.assert_allclose(params["w"], [0.95])
    heavy_ball_step(params, grads, state, lr=0.1, beta=0.9)
    np.testing.assert_allclose(state["velocity"]["w"], [-0.095])
    np.testing.assert_allclose(params["w"], [0.855])


def test_rmsprop_scales_by_running_square():
    params = {"w": np.array([1.0, -1.0])}
    grads = {"w": np.array([2.0, -4.0])}
    state = rmsprop_state(params)
    rmsprop_step(params, grads, state, lr=0.1, decay=0.5, eps=0.0)
    np.testing.assert_allclose(state["square"]["w"], [2.0, 8.0])
    np.testing.assert_allclose(params["w"], [1 - 0.1 * 2 / np.sqrt(2),
                                             -1 + 0.1 * 4 / np.sqrt(8)])


def test_adam_bias_correction_and_adamw_decay():
    params = {"w": np.array([1.0, -2.0])}
    grads = {"w": np.array([0.5, -0.25])}
    state = adam_state(params)
    adamw_step(params, grads, state, lr=0.01)
    np.testing.assert_allclose(params["w"], [0.99, -1.99], rtol=1e-7)
    np.testing.assert_allclose(state["m"]["w"], 0.1 * grads["w"])
    adamw_step(params, grads, state, lr=0.01)
    np.testing.assert_allclose(state["m"]["w"], (1 - 0.9 ** 2) * grads["w"])
    assert state["t"] == 2

    decoupled = {"w": np.array([2.0])}
    coupled = {"w": np.array([2.0])}
    zero = {"w": np.array([0.0])}
    adamw_step(decoupled, zero, adam_state(decoupled), lr=0.1,
               weight_decay=0.2, decoupled=True)
    adamw_step(coupled, zero, adam_state(coupled), lr=0.1,
               weight_decay=0.2, decoupled=False)
    np.testing.assert_allclose(decoupled["w"], [1.96])
    assert coupled["w"][0] < decoupled["w"][0]
    np.testing.assert_allclose(adam_first_step(), [0.01, -0.01], rtol=1e-7)


def test_schedules_and_global_norm_clipping():
    assert wsd_schedule(0, 10, 2, 4, 1.0) == 0.5
    assert wsd_schedule(3, 10, 2, 4, 1.0) == 1.0
    assert wsd_schedule(9, 10, 2, 4, 1.0) == 0.0
    assert cosine_decay(0, 5, 0.2, 0.02) == 0.2
    np.testing.assert_allclose(cosine_decay(4, 5, 0.2, 0.02), 0.02)
    grads = {"a": np.array([3.0, 4.0]), "b": np.array([12.0])}
    clipped, norm = clip_by_global_norm(grads, 5.0)
    assert norm == 13.0
    total = np.sqrt(sum(np.sum(g * g) for g in clipped.values()))
    np.testing.assert_allclose(total, 5.0)
    np.testing.assert_allclose(clipped_demo_norm(), (13.0, 5.0))


def test_muon_newton_schulz_matches_svd_polar_factor():
    rng = np.random.default_rng(4)
    for shape in [(4, 6), (6, 4)]:
        matrix = rng.normal(size=shape)
        update = orthogonalize_newton_schulz(matrix, steps=12)
        u, _, vh = np.linalg.svd(matrix, full_matrices=False)
        target = u @ vh
        np.testing.assert_allclose(update, target, atol=3e-2, rtol=3e-2)
        assert update.shape == shape
    gradient = rng.normal(size=(5, 3))
    momentum = np.zeros_like(gradient)
    update = muon_update(gradient, momentum, beta=0.9, steps=12)
    assert update.shape == gradient.shape
    assert np.linalg.norm(momentum) > 0
    try:
        muon_update(rng.normal(size=3), rng.normal(size=3))
    except ValueError as error:
        assert "2-D" in str(error)
    else:
        raise AssertionError("Muon accepted a non-matrix parameter")
