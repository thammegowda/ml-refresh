import numpy as np

from scratch.autodiff import Tensor
from scratch.gradcheck import check_gradient
from solutions import tiny_network_loss_and_grads


def rng(seed=10):
    return np.random.default_rng(seed)


def test_scalar_expression_accumulates_shared_gradients():
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = (x * y + x * x + y.exp()).log()
    z.backward()
    value = np.log(2 * 3 + 2 ** 2 + np.exp(3))
    denom = 2 * 3 + 2 ** 2 + np.exp(3)
    np.testing.assert_allclose(z.data, value)
    np.testing.assert_allclose(x.grad, (3 + 2 * 2) / denom)
    np.testing.assert_allclose(y.grad, (2 + np.exp(3)) / denom)


def test_broadcasting_uses_unbroadcast_for_bias_gradients():
    X = Tensor(np.array([[-1.0, 2.0, 0.5], [3.0, -0.2, 1.0]]))
    b = Tensor(np.array([0.5, -1.0, 0.25]))
    y = (X + b).relu().sum()
    y.backward()
    active = (X.data + b.data) > 0
    np.testing.assert_allclose(X.grad, active.astype(np.float64))
    np.testing.assert_allclose(b.grad, active.sum(axis=0))


def test_matmul_backward_matches_matrix_calculus():
    A = Tensor(np.array([[1.0, -2.0], [0.5, 3.0]]))
    W = Tensor(np.array([[2.0, -1.0, 0.5], [1.5, 0.0, -2.0]]))
    loss = (A @ W).sum()
    loss.backward()
    upstream = np.ones((2, 3))
    np.testing.assert_allclose(A.grad, upstream @ W.data.T)
    np.testing.assert_allclose(W.grad, A.data.T @ upstream)


def test_tiny_network_gradients_match_finite_differences():
    X = np.array([[0.8, -0.4], [1.2, 0.6]], dtype=np.float64)
    W = np.array([[0.7, -0.2, 0.5], [-0.3, 0.9, 0.4]], dtype=np.float64)
    b = np.array([0.2, 0.4, -0.1], dtype=np.float64)
    sizes = [X.size, W.size, b.size]
    theta = np.r_[X.ravel(), W.ravel(), b.ravel()]

    def unpack(theta_value):
        i = sizes[0]
        j = i + sizes[1]
        return (theta_value[:i].reshape(X.shape),
                theta_value[i:j].reshape(W.shape),
                theta_value[j:].reshape(b.shape))

    def loss(theta_value):
        x_value, w_value, b_value = unpack(theta_value)
        return tiny_network_loss_and_grads(x_value, w_value, b_value)[0]

    x_grad, w_grad, b_grad = tiny_network_loss_and_grads(X, W, b)[1:]
    analytic = np.r_[x_grad.ravel(), w_grad.ravel(), b_grad.ravel()]
    check_gradient(loss, theta, analytic, tolerance=1e-7)


def test_backward_accepts_an_explicit_vector_jacobian_seed():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x * x
    y.backward(np.array([1.0, 0.0, -1.0]))
    np.testing.assert_allclose(x.grad, np.array([2.0, 0.0, -6.0]))
