import numpy as np

from logistic import fit_logistic_minibatch, logistic_loss_and_grads, sigmoid
from regression import (gradient_descent_linear, mse_gradients, mse_loss,
                        normal_equation, polynomial_features, predict_linear)
from scratch.gradcheck import check_gradient
from solutions import polynomial_losses
from splits import train_validation_test_split


def rng(seed=9):
    return np.random.default_rng(seed)


def test_normal_equation_recovers_tiny_linear_regression_solution():
    X = np.array([[-1.0, 0.5], [0.0, 1.0], [1.0, -0.5], [2.0, 2.0]])
    true_w = np.array([1.5, -0.7])
    y = X @ true_w + 0.25
    w, b = normal_equation(X.astype(np.float32), y.astype(np.float32))
    np.testing.assert_allclose(w, true_w, atol=1e-6)
    np.testing.assert_allclose(b, 0.25, atol=1e-6)


def test_mse_gradients_match_finite_differences():
    X = rng().normal(size=(6, 3))
    y = rng().normal(size=6)
    theta = rng().normal(size=4)

    def loss(theta_value):
        return mse_loss(X, y, theta_value[:-1], theta_value[-1])

    grad_w, grad_b = mse_gradients(X, y, theta[:-1], theta[-1])
    check_gradient(loss, theta, np.r_[grad_w, grad_b])


def test_gradient_descent_linear_decreases_loss_and_matches_closed_form():
    r = rng()
    X = r.normal(size=(40, 2)).astype(np.float32)
    y = (X @ np.array([0.8, -1.2], dtype=np.float32) + 0.3).astype(np.float32)
    w0 = np.zeros(2, dtype=np.float32)
    w_gd, b_gd, losses = gradient_descent_linear(X, y, w0, 0.0, steps=250, lr=0.08)
    w_ne, b_ne = normal_equation(X, y)
    assert losses[-1] < losses[0] * 0.02
    np.testing.assert_allclose(w_gd, w_ne, atol=1e-3)
    np.testing.assert_allclose(b_gd, b_ne, atol=1e-3)


def test_logistic_bce_gradient_is_x_t_p_minus_y_over_n():
    X = rng().normal(size=(7, 4))
    y = np.array([0, 1, 1, 0, 1, 0, 1], dtype=np.float64)
    theta = rng().normal(size=5)

    def loss(theta_value):
        return logistic_loss_and_grads(X, y, theta_value[:-1], theta_value[-1])[0]

    value, grad_w, grad_b = logistic_loss_and_grads(X, y, theta[:-1], theta[-1])
    p = sigmoid(X @ theta[:-1] + theta[-1])
    np.testing.assert_allclose(grad_w, X.T @ (p - y) / len(X))
    np.testing.assert_allclose(grad_b, np.mean(p - y))
    check_gradient(loss, theta, np.r_[grad_w, grad_b], tolerance=2e-6)
    assert value > 0


def test_minibatch_sgd_moves_logistic_regression_toward_a_separator():
    r = rng()
    X = r.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(np.float32)
    w, b, losses = fit_logistic_minibatch(
        X, y, np.zeros(2, dtype=np.float32), 0.0, 300, 0.25, 16, r)
    assert losses[-1] < losses[0] * 0.45
    preds = sigmoid(X @ w + b) >= 0.5
    assert np.mean(preds == y) > 0.9


def test_train_validation_test_split_is_disjoint_and_sized():
    train, validation, test = train_validation_test_split(50, rng=rng())
    assert [len(train), len(validation), len(test)] == [30, 10, 10]
    assert len(set(train) & set(validation)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(validation) & set(test)) == 0
    assert sorted(np.r_[train, validation, test].tolist()) == list(range(50))


def test_polynomial_degree_three_beats_underfit_and_overfit_models():
    losses = polynomial_losses()
    assert losses[3] < losses[1] * 0.2
    assert losses[3] < losses[11] * 0.05
    assert set(losses) == {1, 3, 11}
