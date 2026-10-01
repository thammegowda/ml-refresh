import numpy as np

from scratch.attention import (
    attention_backward,
    attention_forward,
    causal_mask,
    dot_product_variance,
    padding_mask,
)
from scratch.gradcheck import check_gradient
from solutions import causal_attention_weights, variance_ratios


def test_scaling_variance_matches_width_numerically():
    widths = np.array([4, 16, 64])
    ratios = variance_ratios(widths)
    np.testing.assert_allclose(ratios, np.ones_like(ratios), rtol=0.05)
    assert abs(dot_product_variance(8) / 8 - 1.0) < 0.05


def test_causal_and_padding_masks_zero_disallowed_attention():
    rng = np.random.default_rng(19)
    Q = rng.standard_normal((1, 4, 3))
    K = rng.standard_normal((1, 4, 3))
    V = rng.standard_normal((1, 4, 2))
    _out, cache = attention_forward(Q, K, V, causal_mask(4), True)
    weights = cache[3]
    assert np.allclose(weights[0][~causal_mask(4)], 0.0)
    valid = np.array([[True, True, False, True]])
    _out, cache = attention_forward(Q, K, V, padding_mask(valid), True)
    assert np.allclose(cache[3][0, :, 2], 0.0)
    assert np.allclose(np.triu(causal_attention_weights(), 1), 0.0)


def test_attention_backward_matches_finite_differences():
    rng = np.random.default_rng(20)
    Q = rng.standard_normal((1, 3, 2))
    K = rng.standard_normal((1, 3, 2))
    V = rng.standard_normal((1, 3, 2))
    grad_output = rng.standard_normal((1, 3, 2))
    mask = causal_mask(3)
    output, cache = attention_forward(Q, K, V, mask, True)
    assert output.shape == grad_output.shape
    grad_Q, grad_K, grad_V = attention_backward(grad_output, cache)
    loss = lambda q, k, v: np.sum(attention_forward(q, k, v, mask) * grad_output)
    check_gradient(lambda q: loss(q, K, V), Q, grad_Q, tolerance=1e-7)
    check_gradient(lambda k: loss(Q, k, V), K, grad_K, tolerance=1e-7)
    check_gradient(lambda v: loss(Q, K, v), V, grad_V, tolerance=1e-7)


def test_self_and_cross_attention_shapes():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((2, 5, 4))
    self_out = attention_forward(X, X, X, causal_mask(5))
    assert self_out.shape == X.shape
    Q = rng.standard_normal((2, 3, 4))
    K = rng.standard_normal((2, 6, 4))
    V = rng.standard_normal((2, 6, 7))
    cross_out = attention_forward(Q, K, V)
    assert cross_out.shape == (2, 3, 7)
