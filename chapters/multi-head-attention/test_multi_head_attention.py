import numpy as np

from scratch.attention import causal_mask
from scratch.gradcheck import check_gradient
from scratch.multi_head_attention import (
    combine_heads,
    multi_head_attention_backward,
    multi_head_attention_forward,
    parameter_count,
    self_attention_flops,
    split_heads,
)
from solutions import first_head_causal_weights, tiny_budget


def tiny_inputs(seed=20):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, 3, 4))
    params = tuple(rng.standard_normal((4, 4)) * 0.1 for _ in range(4))
    return X, params


def test_split_and_combine_heads_round_trip():
    x = np.arange(2 * 3 * 8).reshape(2, 3, 8)
    heads = split_heads(x, 4)
    assert heads.shape == (2, 4, 3, 2)
    np.testing.assert_array_equal(combine_heads(heads), x)


def test_multi_head_attention_backward_matches_finite_differences():
    X, params = tiny_inputs()
    grad_output = np.random.default_rng(21).standard_normal((1, 3, 4))
    mask = causal_mask(3)
    output, cache = multi_head_attention_forward(X, X, X, params, 2, mask, True)
    assert output.shape == X.shape
    (grad_Xq, grad_Xk, grad_Xv), grad_params = multi_head_attention_backward(
        grad_output,
        cache,
    )
    loss = lambda x, ps: np.sum(
        multi_head_attention_forward(x, x, x, ps, 2, mask) * grad_output
    )
    check_gradient(lambda x: loss(x, params), X, grad_Xq + grad_Xk + grad_Xv,
                   tolerance=1e-7)
    for index, grad_param in enumerate(grad_params):
        def param_loss(candidate, index=index):
            edited = list(params)
            edited[index] = candidate
            return loss(X, tuple(edited))
        check_gradient(param_loss, params[index], grad_param, tolerance=1e-7)


def test_counts_and_causal_mask_across_heads():
    assert parameter_count(8) == 256
    assert tiny_budget() == (256, self_attention_flops(2, 5, 8))
    weights = first_head_causal_weights()
    assert weights.shape == (3, 3)
    assert np.allclose(np.triu(weights, 1), 0.0)


def test_cross_attention_shape():
    rng = np.random.default_rng(22)
    Xq = rng.standard_normal((2, 3, 4))
    Xk = rng.standard_normal((2, 5, 4))
    Xv = rng.standard_normal((2, 5, 4))
    params = tuple(rng.standard_normal((4, 4)) * 0.1 for _ in range(4))
    output = multi_head_attention_forward(Xq, Xk, Xv, params, 2)
    assert output.shape == (2, 3, 4)
