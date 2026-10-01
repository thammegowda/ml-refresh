import numpy as np

from scratch.activations import (gelu_exact, gelu_exact_grad, gelu_tanh,
                                 gelu_tanh_grad, gated_ffn,
                                 gated_ffn_backward, leaky_relu,
                                 leaky_relu_grad, relu, relu_grad, sigmoid,
                                 sigmoid_grad, silu, silu_grad, softplus,
                                 softplus_grad, tanh_grad)
from scratch.gradcheck import check_gradient
from solutions import gated_hidden_width, gelu_tanh_max_error


def rng(seed=11):
    return np.random.default_rng(seed)


def test_sigmoid_tanh_saturation_and_bounds():
    x = np.array([-30.0, -5.0, 0.0, 5.0, 30.0])
    s = sigmoid(x)
    assert 0 < s[0] < 1e-12 and 0 <= 1 - s[-1] < 1e-12
    np.testing.assert_allclose(sigmoid_grad(0.0), 0.25)
    assert np.all(sigmoid_grad(x) <= 0.25)
    t = np.tanh(x)
    assert abs(t[0] + 1) < 1e-15 and abs(t[-1] - 1) < 1e-15
    assert np.all(tanh_grad(x) <= 1)
    np.testing.assert_allclose(tanh_grad(0.0), 1.0)


def test_elementwise_activation_derivatives_match_finite_differences():
    x = rng().normal(size=(3, 4)) + 0.2
    cases = [
        (sigmoid, sigmoid_grad),
        (np.tanh, tanh_grad),
        (relu, relu_grad),
        (leaky_relu, leaky_relu_grad),
        (gelu_exact, gelu_exact_grad),
        (gelu_tanh, gelu_tanh_grad),
        (silu, silu_grad),
        (softplus, softplus_grad),
    ]
    for forward, grad in cases:
        check_gradient(lambda z, f=forward: np.sum(f(z)), x, grad(x))


def test_relu_can_die_but_leaky_relu_keeps_a_negative_slope():
    x = np.array([-3.0, -0.2, 0.0, 2.0])
    np.testing.assert_allclose(relu_grad(x), [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(leaky_relu_grad(x, 0.05), [0.05, 0.05, 1.0, 1.0])
    assert np.all(relu(x[:2]) == 0)
    assert np.all(leaky_relu(x[:2], 0.05) < 0)


def test_gelu_tanh_approximation_error_is_measured():
    error = gelu_tanh_max_error()
    np.testing.assert_allclose(error, 0.0004732355197617192, rtol=0, atol=1e-12)
    assert error < 4.8e-4


def test_gated_ffn_parameter_count_and_gradients():
    d = 12
    h = int(gated_hidden_width(d))
    assert h == 32
    assert 3 * d * h == 2 * d * (4 * d)

    r = rng()
    x = r.normal(size=(2, d))
    W = r.normal(size=(d, h)) / np.sqrt(d)
    V = r.normal(size=(d, h)) / np.sqrt(d)
    W2 = r.normal(size=(h, d)) / np.sqrt(h)
    grad_y = r.normal(size=(2, d))
    grad_x, grad_W, grad_V, grad_W2 = gated_ffn_backward(x, W, V, W2, grad_y)

    def loss_x(value):
        return float(np.sum(gated_ffn(value, W, V, W2) * grad_y))

    def loss_W(value):
        return float(np.sum(gated_ffn(x, value, V, W2) * grad_y))

    def loss_V(value):
        return float(np.sum(gated_ffn(x, W, value, W2) * grad_y))

    def loss_W2(value):
        return float(np.sum(gated_ffn(x, W, V, value) * grad_y))

    check_gradient(loss_x, x, grad_x)
    check_gradient(loss_W, W, grad_W, tolerance=1e-5)
    check_gradient(loss_V, V, grad_V, tolerance=1e-5)
    check_gradient(loss_W2, W2, grad_W2, tolerance=1e-5)
