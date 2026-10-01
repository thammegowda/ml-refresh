import numpy as np

from solutions import last_row_with_sink, memory_example_mib
from scratch.kv_cache import (causal_self_attention, expand_kv_heads,
                              gqa_attention, incremental_self_attention,
                              kv_cache_bytes, l2_normalize_last_axis,
                              sliding_window_mask)


def rng(seed=24):
    return np.random.default_rng(seed)


def weights(d_model=12, num_heads=4, num_kv_heads=2, head_dim=3):
    r = rng()
    scale = 1 / np.sqrt(d_model)
    w_q = r.normal(scale=scale, size=(d_model, num_heads, head_dim))
    w_k = r.normal(scale=scale, size=(d_model, num_kv_heads, head_dim))
    w_v = r.normal(scale=scale, size=(d_model, num_kv_heads, head_dim))
    w_o = r.normal(scale=scale, size=(num_heads, head_dim, d_model))
    return w_q, w_k, w_v, w_o


def test_incremental_decode_matches_full_prefill_recomputation():
    x = rng().normal(size=(6, 12)).astype(np.float32)
    w_q, w_k, w_v, w_o = weights()
    full = causal_self_attention(x, w_q, w_k, w_v, w_o)
    cached = incremental_self_attention(x, w_q, w_k, w_v, w_o)
    np.testing.assert_allclose(cached, full, atol=1e-6)


def test_cache_memory_formula_example_is_asserted():
    bytes_used = kv_cache_bytes(32, 8, 128, 4096, 2)
    assert bytes_used == 536_870_912
    assert bytes_used // (1024 ** 2) == 512
    assert memory_example_mib() == 512


def test_g_equals_h_reduces_to_multi_head_attention():
    r = rng()
    q = r.normal(size=(3, 4, 5))
    k = r.normal(size=(3, 4, 5))
    v = r.normal(size=(3, 4, 5))
    got, got_weights = gqa_attention(q, k, v)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(5)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("hts,shd->thd", weights, v)
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(got_weights, weights)


def test_grouped_and_multi_query_head_expansion():
    kv = np.arange(2 * 2 * 3).reshape(2, 2, 3)
    expanded = expand_kv_heads(kv, num_query_heads=6)
    assert expanded.shape == (2, 6, 3)
    np.testing.assert_array_equal(expanded[:, 0], kv[:, 0])
    np.testing.assert_array_equal(expanded[:, 2], kv[:, 0])
    np.testing.assert_array_equal(expanded[:, 3], kv[:, 1])
    one_head = kv[:, :1, :]
    np.testing.assert_array_equal(expand_kv_heads(one_head, 4)[:, 3], kv[:, 0])


def test_sliding_window_mask_and_attention_sinks():
    no_sink = sliding_window_mask(tokens=6, window=3, sinks=0)
    with_sink = sliding_window_mask(tokens=6, window=3, sinks=1)
    np.testing.assert_array_equal(no_sink[5].astype(int), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(with_sink[5].astype(int), [1, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(last_row_with_sink(), [1, 0, 0, 1, 1, 1])


def test_qk_norm_removes_query_and_key_scale_from_scores():
    r = rng()
    q = r.normal(size=(2, 3, 4))
    k = r.normal(size=(2, 3, 4))
    q_scaled = l2_normalize_last_axis(7.0 * q)
    k_scaled = l2_normalize_last_axis(0.2 * k)
    q_norm = l2_normalize_last_axis(q)
    k_norm = l2_normalize_last_axis(k)
    scores_scaled = np.einsum("thd,shd->hts", q_scaled, k_scaled)
    scores_norm = np.einsum("thd,shd->hts", q_norm, k_norm)
    np.testing.assert_allclose(scores_scaled, scores_norm)
