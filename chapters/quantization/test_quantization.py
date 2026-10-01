import numpy as np

from scratch.quantization import (decoding_intensity, fp8_e4m3, fp8_e5m2,
                                  gptq_quantize_vector, mean_squared_error,
                                  mxfp4, paged_block_table,
                                  quantize_absmax, quantize_absmax_groups,
                                  quantize_zero_point, reconstruction_loss,
                                  roofline_tokens_per_second,
                                  smoothquant_migrate)
from solutions import tensor_channel_group_errors


def rng(seed=40):
    return np.random.default_rng(seed)


def test_absmax_and_zero_point_quantization_reconstruct_tiny_tensors():
    x = np.array([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=np.float32)
    q, dequant, scale = quantize_absmax(x, bits=3)
    assert q.tolist() == [-3, -1, 0, 2, 3]
    np.testing.assert_allclose(scale, [1 / 3])
    np.testing.assert_allclose(dequant, q / 3, atol=1e-7)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    qz, yz, scale, zero = quantize_zero_point(y, bits=2)
    assert qz.tolist() == [0, 2, 3]
    np.testing.assert_allclose(yz[[0, -1]], [0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose([scale.item(), zero.item()], [2 / 3, 0])


def test_per_channel_and_per_group_reduce_quantization_error():
    x = np.array([[0.02, -0.03, 0.04, 0.05, 1.0, -1.1, 0.9, -0.8],
                  [2.0, -1.8, 2.2, -2.1, 0.1, -0.1, 0.2, -0.2]],
                 dtype=np.float32)
    tensor, channel, group = tensor_channel_group_errors(x)
    assert tensor > channel > group
    _, group_dequant, _ = quantize_absmax_groups(x, bits=4, group_size=4)
    np.testing.assert_allclose(mean_squared_error(x, group_dequant), group)


def test_fp8_and_mxfp4_emulations_have_expected_tradeoffs():
    near_one = np.array([0.9, 1.1, 3.25], dtype=np.float32)
    assert mean_squared_error(near_one, fp8_e4m3(near_one)) < mean_squared_error(
        near_one, fp8_e5m2(near_one))
    np.testing.assert_allclose(fp8_e4m3([1000.0]), [448.0])
    assert fp8_e5m2([1000.0])[0] >= 1000.0
    x = np.array([[0.1, -0.2, 0.3, -0.4, 5.0, -6.0, 4.0, -3.0]],
                 dtype=np.float32)
    assert mean_squared_error(x, mxfp4(x, 4)) < mean_squared_error(x, mxfp4(x, 8))


def test_smoothquant_migration_preserves_the_layer_output():
    x = np.array([[30.0, 0.5, -0.4], [25.0, -0.3, 0.2]], dtype=np.float32)
    w = np.array([[0.1, -0.2], [2.0, 1.5], [-1.0, 0.5]], dtype=np.float32)
    xs, ws, scale = smoothquant_migrate(x, w, alpha=0.5)
    np.testing.assert_allclose(xs @ ws, x @ w, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(xs[:, 0])) < np.max(np.abs(x[:, 0]))
    assert scale[0] > scale[1]


def test_gptq_compensation_beats_round_to_nearest_on_correlated_inputs():
    r = np.random.default_rng(8)
    x = r.normal(size=(64, 6))
    x[:, 1] = 0.95 * x[:, 0] + 0.05 * r.normal(size=64)
    hessian = x.T @ x
    w = r.normal(size=6)
    _, naive, _ = quantize_absmax(w, bits=2)
    gptq = gptq_quantize_vector(w, hessian, bits=2)
    assert reconstruction_loss(w, gptq, hessian) < 0.25 * reconstruction_loss(
        w, naive, hessian)


def test_paged_cache_tables_and_roofline_calculator():
    assert paged_block_table([1, 17, 32], 16) == [[0], [1, 2], [3, 4]]
    params = 7_000_000_000
    intensity = decoding_intensity(params, weight_bytes=2, kv_bytes=0)
    np.testing.assert_allclose(intensity, 1.0)
    tokens = roofline_tokens_per_second(params, bandwidth=3e12, peak_flops=9e14)
    np.testing.assert_allclose(tokens, 3e12 / (2 * params))
