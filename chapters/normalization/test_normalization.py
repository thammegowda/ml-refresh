import numpy as np

from precision_demo import fp16_overflows_but_bfloat16_keeps_range, loss_scaled_gradient
from solutions import dropout_mean, rms_scale_invariance
from scratch.gradcheck import check_gradient
from scratch.normalization import (batch_norm_forward, inverted_dropout,
                                   layer_norm_backward, layer_norm_forward,
                                   rms_norm_backward, rms_norm_forward)
from scratch.precision import round_to_bfloat16


def pack(*arrays):
    shapes = [array.shape for array in arrays]
    vector = np.concatenate([array.ravel() for array in arrays])
    return vector, shapes


def unpack(vector, shapes):
    arrays = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(vector[offset:offset + size].reshape(shape))
        offset += size
    return arrays


def test_layer_norm_backward_matches_finite_differences():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4))
    gamma = rng.normal(size=4)
    beta = rng.normal(size=4)
    upstream = rng.normal(size=(3, 4))
    vector, shapes = pack(x, gamma, beta)

    def loss(theta):
        xx, gg, bb = unpack(theta, shapes)
        out, _ = layer_norm_forward(xx, gg, bb)
        return float(np.sum(out * upstream))

    _, cache = layer_norm_forward(x, gamma, beta)
    dx, dgamma, dbeta = layer_norm_backward(upstream, cache)
    analytic, _ = pack(dx, dgamma, dbeta)
    check_gradient(loss, vector, analytic, tolerance=2e-7)


def test_rms_norm_backward_matches_finite_differences():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 5))
    weight = rng.normal(size=5)
    upstream = rng.normal(size=(2, 3, 5))
    vector, shapes = pack(x, weight)

    def loss(theta):
        xx, ww = unpack(theta, shapes)
        out, _ = rms_norm_forward(xx, ww)
        return float(np.sum(out * upstream))

    _, cache = rms_norm_forward(x, weight)
    dx, dweight = rms_norm_backward(upstream, cache)
    analytic, _ = pack(dx, dweight)
    check_gradient(loss, vector, analytic, tolerance=2e-7)


def test_batch_norm_training_updates_running_statistics_and_inference_uses_them():
    x = np.array([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
    gamma = np.ones(2)
    beta = np.zeros(2)
    running_mean = np.zeros(2)
    running_var = np.ones(2)
    train = batch_norm_forward(x, gamma, beta, running_mean, running_var,
                               training=True, momentum=0.5)
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(running_mean, x.mean(axis=0) * 0.5)
    inference = batch_norm_forward(x, gamma, beta, running_mean, running_var,
                                   training=False)
    assert not np.allclose(inference.mean(axis=0), 0.0)


def test_inverted_dropout_preserves_expectation():
    rng = np.random.default_rng(3)
    x = np.ones(50_000)
    dropped, mask = inverted_dropout(x, 0.2, rng)
    assert mask.dtype == np.bool_
    np.testing.assert_allclose(dropped.mean(), 1.0, atol=0.02)
    np.testing.assert_allclose(dropout_mean(), 1.0, atol=0.02)


def test_mixed_precision_range_scaling_and_master_accumulation():
    assert np.finfo(np.float16).max == 65504
    fp16, bf16 = fp16_overflows_but_bfloat16_keeps_range()
    assert np.isinf(fp16).all()
    assert np.isfinite(bf16).all()
    tiny = np.array([2 ** -20], dtype=np.float32)
    assert tiny.astype(np.float16)[0] != 0
    assert (tiny * 2 ** -10).astype(np.float16)[0] == 0
    restored = loss_scaled_gradient(np.array([2 ** -30], dtype=np.float32), 2 ** 15)
    assert restored[0] > 0
    master = np.array([1.0], dtype=np.float32)
    low = round_to_bfloat16(master)
    grad = np.array([2 ** -10], dtype=np.float32)
    for _ in range(8):
        master -= grad
        low = round_to_bfloat16(low - grad)
    assert master[0] < low[0]
    assert rms_scale_invariance(np.array([[1.0, 2.0, 3.0]]), np.ones(3), 5.0) < 1e-7
