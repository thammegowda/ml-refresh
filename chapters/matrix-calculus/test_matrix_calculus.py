import numpy as np

from scratch.gradcheck import check_gradient
from solutions import (attention_query_gradient, broadcast_add_gradient,
                       matmul_shapes, softmax_jacobian_times_vector)
from vjps import (add_forward, add_vjp, attention_vjp, cross_entropy_forward,
                  cross_entropy_vjp, embedding_forward, embedding_vjp,
                  exp_forward, exp_vjp, layer_norm_forward, layer_norm_vjp,
                  log_forward, log_softmax_forward, log_softmax_vjp, log_vjp,
                  matmul_forward, matmul_vjp, mean_forward, mean_vjp,
                  multiply_forward, multiply_vjp, relu_forward, relu_vjp,
                  reshape_forward, reshape_vjp, rms_norm_forward, rms_norm_vjp,
                  scaled_dot_product_attention, sigmoid_forward, sigmoid_vjp,
                  softmax_forward, softmax_vjp, sum_forward, sum_vjp,
                  tanh_forward, tanh_vjp, transpose_forward, transpose_vjp)


def rng(seed=42):
    return np.random.default_rng(seed)


def check_unary(forward, vjp, x, seed=0, tolerance=1e-7):
    r = rng(seed)
    y = forward(x)
    grad_y = r.standard_normal(y.shape)
    analytic = vjp(grad_y, y)
    check_gradient(lambda z: np.sum(forward(z) * grad_y), x, analytic,
                   tolerance=tolerance)


def test_add_multiply_and_matmul_vjps():
    r = rng()
    x = r.standard_normal((2, 3))
    bias = r.standard_normal((1, 3))
    grad = r.standard_normal((2, 3))
    grad_x, grad_bias = add_vjp(grad, x, bias)
    check_gradient(lambda z: np.sum(add_forward(z, bias) * grad), x, grad_x)
    check_gradient(lambda z: np.sum(add_forward(x, z) * grad), bias, grad_bias)
    np.testing.assert_allclose(broadcast_add_gradient(grad, x, bias), grad_bias)

    grad_x, grad_bias = multiply_vjp(grad, x, bias)
    check_gradient(lambda z: np.sum(multiply_forward(z, bias) * grad), x, grad_x)
    check_gradient(lambda z: np.sum(multiply_forward(x, z) * grad), bias, grad_bias)

    a = r.standard_normal((2, 3))
    b = r.standard_normal((3, 4))
    grad = r.standard_normal((2, 4))
    grad_a, grad_b = matmul_vjp(grad, a, b)
    assert matmul_shapes(a, b, grad) == ((2, 3), (3, 4))
    check_gradient(lambda z: np.sum(matmul_forward(z, b) * grad), a, grad_a)
    check_gradient(lambda z: np.sum(matmul_forward(a, z) * grad), b, grad_b)


def test_sum_mean_and_elementwise_vjps():
    r = rng(1)
    x = r.standard_normal((2, 3, 4))
    grad = r.standard_normal((2, 4))
    check_gradient(lambda z: np.sum(sum_forward(z, axis=1) * grad), x,
                   sum_vjp(grad, x.shape, axis=1))
    check_gradient(lambda z: np.sum(mean_forward(z, axis=1) * grad), x,
                   mean_vjp(grad, x.shape, axis=1))

    check_unary(exp_forward, exp_vjp, r.standard_normal((3, 4)), seed=2)
    grad_elem = r.standard_normal((3, 4))
    positive = np.exp(r.standard_normal((3, 4)))
    check_gradient(lambda z: np.sum(log_forward(z) * grad_elem), positive,
                   log_vjp(grad_elem, positive))
    nonzero = r.standard_normal((3, 4)) + 0.25
    check_gradient(lambda z: np.sum(relu_forward(z) * grad_elem), nonzero,
                   relu_vjp(grad_elem, nonzero))
    check_unary(sigmoid_forward, sigmoid_vjp, r.standard_normal((3, 4)), seed=3)
    check_unary(tanh_forward, tanh_vjp, r.standard_normal((3, 4)), seed=4)


def test_softmax_log_softmax_and_cross_entropy_vjps():
    r = rng(5)
    x = r.standard_normal((3, 5))
    grad = r.standard_normal((3, 5))
    y = softmax_forward(x)
    analytic = softmax_vjp(grad, y)
    check_gradient(lambda z: np.sum(softmax_forward(z) * grad), x, analytic)
    np.testing.assert_allclose(softmax_jacobian_times_vector(x[0], grad[0]),
                               analytic[0])

    log_y = log_softmax_forward(x)
    analytic = log_softmax_vjp(grad, log_y)
    check_gradient(lambda z: np.sum(log_softmax_forward(z) * grad), x, analytic)

    labels = np.array([0, 3, 1])
    analytic = cross_entropy_vjp(x.copy(), labels)
    check_gradient(lambda z: cross_entropy_forward(z, labels), x, analytic)


def test_layer_norm_and_rms_norm_vjps():
    r = rng(6)
    x = r.standard_normal((2, 4))
    gamma = r.standard_normal(4)
    beta = r.standard_normal(4)
    grad = r.standard_normal((2, 4))
    grad_x, grad_gamma, grad_beta = layer_norm_vjp(grad, x, gamma, beta)
    f = lambda z: np.sum(layer_norm_forward(z, gamma, beta) * grad)
    check_gradient(f, x, grad_x, tolerance=1e-6)
    f = lambda z: np.sum(layer_norm_forward(x, z, beta) * grad)
    check_gradient(f, gamma, grad_gamma)
    f = lambda z: np.sum(layer_norm_forward(x, gamma, z) * grad)
    check_gradient(f, beta, grad_beta)

    grad_x, grad_gamma = rms_norm_vjp(grad, x, gamma)
    f = lambda z: np.sum(rms_norm_forward(z, gamma) * grad)
    check_gradient(f, x, grad_x, tolerance=1e-6)
    f = lambda z: np.sum(rms_norm_forward(x, z) * grad)
    check_gradient(f, gamma, grad_gamma)


def test_embedding_reshape_and_transpose_vjps():
    r = rng(7)
    weights = r.standard_normal((5, 3))
    ids = np.array([[0, 2], [2, 4]])
    grad = r.standard_normal((2, 2, 3))
    analytic = embedding_vjp(grad, ids, vocab_size=5)
    check_gradient(lambda z: np.sum(embedding_forward(z, ids) * grad), weights,
                   analytic)
    np.testing.assert_allclose(analytic[2], grad[0, 1] + grad[1, 0])

    x = r.standard_normal((2, 3, 4))
    grad = r.standard_normal((4, 6))
    check_gradient(lambda z: np.sum(reshape_forward(z, (4, 6)) * grad), x,
                   reshape_vjp(grad, x.shape))
    axes = (1, 0, 2)
    grad = r.standard_normal((3, 2, 4))
    check_gradient(lambda z: np.sum(transpose_forward(z, axes) * grad), x,
                   transpose_vjp(grad, axes))


def test_scaled_dot_product_attention_vjp():
    r = rng(8)
    q = r.standard_normal((1, 3, 2))
    k = r.standard_normal((1, 3, 2))
    v = r.standard_normal((1, 3, 2))
    grad = r.standard_normal((1, 3, 2))
    grad_q, grad_k, grad_v = attention_vjp(grad, q, k, v)
    f = lambda z: np.sum(scaled_dot_product_attention(z, k, v) * grad)
    check_gradient(f, q, grad_q, tolerance=1e-6)
    f = lambda z: np.sum(scaled_dot_product_attention(q, z, v) * grad)
    check_gradient(f, k, grad_k, tolerance=1e-6)
    f = lambda z: np.sum(scaled_dot_product_attention(q, k, z) * grad)
    check_gradient(f, v, grad_v, tolerance=1e-6)
    np.testing.assert_allclose(attention_query_gradient(q, k, v, grad), grad_q)
