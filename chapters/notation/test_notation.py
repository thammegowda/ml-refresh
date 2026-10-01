import numpy as np

from affine import affine_backward, affine_forward
from solutions import elementwise_vjp, elementwise_vjp_dense, from_pytorch_linear
from scratch.gradcheck import check_gradient

rng = np.random.default_rng(1)


def test_affine_backward_matches_finite_differences():
    X, W, b = rng.standard_normal((5, 3)), rng.standard_normal((3, 4)), rng.standard_normal(4)
    G = rng.standard_normal((5, 4))
    grad_X, grad_W, grad_b = affine_backward(G, X, W)
    assert (grad_X.shape, grad_W.shape, grad_b.shape) == (X.shape, W.shape, b.shape)
    check_gradient(lambda v: float(np.sum(G * affine_forward(v, W, b))), X, grad_X)
    check_gradient(lambda v: float(np.sum(G * affine_forward(X, v, b))), W, grad_W)
    check_gradient(lambda v: float(np.sum(G * affine_forward(X, W, v))), b, grad_b)


def test_shape_exercise():
    X = np.zeros((32, 128), dtype=np.float32)
    W1, b1 = np.zeros((128, 512), np.float32), np.zeros(512, np.float32)
    W2, b2 = np.zeros((512, 10), np.float32), np.zeros(10, np.float32)
    H = np.maximum(affine_forward(X, W1, b1), 0)
    logits = affine_forward(H, W2, b2)
    assert H.shape == (32, 512) and logits.shape == (32, 10)
    assert sum(p.size for p in (W1, b1, W2, b2)) == 71_178


def test_pytorch_weights_translate_by_transpose():
    weight, bias = rng.standard_normal((4, 3)), rng.standard_normal(4)
    x = rng.standard_normal((2, 3))
    W, b = from_pytorch_linear(weight, bias)
    np.testing.assert_allclose(affine_forward(x, W, b), x @ weight.T + bias)
    G = rng.standard_normal((2, 4))
    np.testing.assert_allclose(affine_backward(G, x, W)[1].T, G.T @ x)


def test_elementwise_vector_jacobian_product():
    x, g = rng.standard_normal(6), rng.standard_normal(6)
    np.testing.assert_allclose(elementwise_vjp(g, x, np.cos), elementwise_vjp_dense(g, x, np.cos))
    check_gradient(lambda v: float(np.sum(g * np.sin(v))), x, elementwise_vjp(g, x, np.cos))
