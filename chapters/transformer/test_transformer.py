import numpy as np

from solutions import block_parameter_count, block_scalar_loss, one_block_output
from scratch.gradcheck import check_gradient
from scratch.transformer import (
    causal_self_attention_forward,
    init_block_params,
    rmsnorm_backward,
    rmsnorm_forward,
    swiglu_backward,
    swiglu_forward,
    transformer_block_backward,
    transformer_block_forward,
)


def rng(seed=22):
    return np.random.default_rng(seed)


def flatten_params(params):
    names = list(params)
    pieces = [np.ravel(params[name]) for name in names]
    return names, [params[name].shape for name in names], np.concatenate(pieces)


def unflatten_params(names, shapes, vector):
    params = {}
    offset = 0
    for name, shape in zip(names, shapes):
        size = int(np.prod(shape))
        params[name] = vector[offset:offset + size].reshape(shape)
        offset += size
    return params


def flatten_grads(names, grads):
    return np.concatenate([np.ravel(grads[name]) for name in names])


def test_rmsnorm_backward_matches_finite_differences():
    r = rng()
    x = r.normal(size=(2, 3, 4)).astype(np.float64)
    weight = r.normal(size=4).astype(np.float64)
    upstream = r.normal(size=x.shape).astype(np.float64)
    out, cache = rmsnorm_forward(x, weight)
    dx, dweight = rmsnorm_backward(upstream, cache)
    check_gradient(lambda z: float(np.sum(rmsnorm_forward(z, weight)[0] * upstream)), x, dx)
    check_gradient(lambda w: float(np.sum(rmsnorm_forward(x, w)[0] * upstream)), weight, dweight)
    assert out.shape == x.shape


def test_swiglu_backward_matches_finite_differences():
    r = rng()
    x = r.normal(size=(1, 2, 3)).astype(np.float64)
    params = {
        "W_gate": r.normal(size=(3, 5)).astype(np.float64) * 0.2,
        "W_up": r.normal(size=(3, 5)).astype(np.float64) * 0.2,
        "W_down": r.normal(size=(5, 3)).astype(np.float64) * 0.2,
    }
    upstream = r.normal(size=x.shape).astype(np.float64)
    _out, cache = swiglu_forward(x, params)
    dx, grads = swiglu_backward(upstream, cache)
    check_gradient(lambda z: float(np.sum(swiglu_forward(z, params)[0] * upstream)), x, dx)
    names, shapes, flat = flatten_params(params)
    analytic = flatten_grads(names, grads)

    def loss(vector):
        trial = unflatten_params(names, shapes, vector)
        return float(np.sum(swiglu_forward(x, trial)[0] * upstream))

    check_gradient(loss, flat, analytic, tolerance=1e-6)


def test_causal_attention_does_not_read_future_tokens():
    r = rng()
    params = init_block_params(4, 2, 8, r, dtype=np.float64)
    x = r.normal(size=(1, 4, 4)).astype(np.float64)
    y, _ = causal_self_attention_forward(x, params, n_heads=2)
    changed = x.copy()
    changed[:, 3] += 100.0
    y_changed, _ = causal_self_attention_forward(changed, params, n_heads=2)
    np.testing.assert_allclose(y_changed[:, :3], y[:, :3], atol=1e-12)
    assert np.max(np.abs(y_changed[:, 3] - y[:, 3])) > 1e-6


def test_transformer_block_backward_matches_finite_differences_end_to_end():
    r = rng()
    params = init_block_params(4, 2, 8, r, dtype=np.float64)
    x = (r.normal(size=(1, 3, 4)) * 0.2).astype(np.float64)
    upstream = r.normal(size=x.shape).astype(np.float64)
    _y, cache = transformer_block_forward(x, params, n_heads=2)
    dx, grads = transformer_block_backward(upstream, cache)
    check_gradient(lambda z: block_scalar_loss(z, params, 2, upstream), x, dx,
                   tolerance=2e-6)
    names, shapes, flat = flatten_params(params)
    analytic = flatten_grads(names, grads)

    def loss(vector):
        trial = unflatten_params(names, shapes, vector)
        return block_scalar_loss(x, trial, 2, upstream)

    check_gradient(loss, flat, analytic, tolerance=2e-6)


def test_block_shapes_and_parameter_count_formula():
    y_shape, dx_shape, grad_keys = one_block_output()
    assert y_shape == (2, 4, 8)
    assert dx_shape == (2, 4, 8)
    assert "W_gate" in grad_keys and "attn_norm" in grad_keys
    assert block_parameter_count(8, 32) == 4 * 8 * 8 + 3 * 8 * 32 + 16
