import numpy as np

from solutions import cache_table_mib
from scratch.latent_attention import (absorbed_scores, cache_size_table,
                                      compress_kv, materialized_scores,
                                      mla_attention, rotate_pairs,
                                      top_k_sparse_attention,
                                      up_project_latents)


def rng(seed=25):
    return np.random.default_rng(seed)


def test_mla_compresses_and_up_projects_keys_and_values():
    r = rng()
    x = r.normal(size=(5, 10))
    w_dkv = r.normal(size=(10, 6))
    w_uk = r.normal(size=(6, 4, 3))
    w_uv = r.normal(size=(6, 4, 3))
    c = compress_kv(x, w_dkv)
    k, v = up_project_latents(c, w_uk, w_uv)
    assert c.shape == (5, 6)
    assert k.shape == (5, 4, 3)
    assert v.shape == (5, 4, 3)
    np.testing.assert_allclose(k[0, 2], c[0] @ w_uk[:, 2, :])


def test_weight_absorption_equals_materialized_scores():
    r = rng()
    q = r.normal(size=(4, 3, 5))
    c = r.normal(size=(4, 7))
    w_uk = r.normal(size=(7, 3, 5))
    direct = materialized_scores(q, c, w_uk)
    absorbed = absorbed_scores(q, c, w_uk)
    np.testing.assert_allclose(absorbed, direct, atol=1e-12)


def test_mla_attention_matches_manual_materialized_attention():
    r = rng()
    q = r.normal(size=(3, 2, 4))
    c = r.normal(size=(3, 5))
    w_uk = r.normal(size=(5, 2, 4))
    w_uv = r.normal(size=(5, 2, 4))
    got, weights = mla_attention(q, c, w_uk, w_uv)
    k, v = up_project_latents(c, w_uk, w_uv)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(4)
    expected_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected_weights /= expected_weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("hts,shd->thd", expected_weights, v)
    np.testing.assert_allclose(weights, expected_weights)
    np.testing.assert_allclose(got, expected)


def test_rope_is_position_dependent_and_breaks_naive_absorption():
    r = rng()
    q = r.normal(size=(4, 2, 4))
    c = r.normal(size=(4, 5))
    w_uk = r.normal(size=(5, 2, 4))
    positions = np.arange(4)
    k, _ = up_project_latents(c, w_uk, w_uk)
    rope_scores = np.einsum(
        "thd,shd->hts",
        rotate_pairs(q, positions),
        rotate_pairs(k, positions),
    )
    assert not np.allclose(rope_scores, absorbed_scores(q, c, w_uk))


def test_cache_size_table_is_computed_and_asserted():
    table = cache_size_table(32, 4096, 2, 32, 8, 128, 512)
    assert table["MHA"] == 2_147_483_648
    assert table["GQA"] == 536_870_912
    assert table["MLA"] == 134_217_728
    assert cache_table_mib() == {"MHA": 2048, "GQA": 512, "MLA": 128}


def test_top_k_sparse_attention_uses_indexer_mask():
    r = rng()
    q = r.normal(size=(4, 3))
    k = r.normal(size=(6, 3))
    v = r.normal(size=(6, 2))
    index_scores = np.array([
        [0.1, 0.7, 0.0, 0.2, 0.3, 0.4],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.9, 0.8, 0.7],
        [0.6, 0.1, 0.5, 0.2, 0.4, 0.3],
    ])
    out, mask, weights = top_k_sparse_attention(q, k, v, index_scores, k_top=2)
    assert out.shape == (4, 2)
    assert mask.sum(axis=1).tolist() == [2, 2, 2, 2]
    np.testing.assert_array_equal(mask[0], [False, True, False, False, False, True])
    np.testing.assert_allclose(weights[~mask], 0.0)
    dense, dense_mask, _ = top_k_sparse_attention(q, k, v, index_scores, k_top=6)
    np.testing.assert_allclose(dense_mask, np.ones_like(dense_mask, dtype=bool))
    assert dense.shape == (4, 2)
