import numpy as np

from solutions import memory_elements_for_4096
from scratch.flash_attention import (attention_memory_elements, naive_attention,
                                     online_softmax_normalizer,
                                     tiled_attention)


def rng(seed=26):
    return np.random.default_rng(seed)


def test_online_softmax_normalizer_matches_full_row():
    scores = rng().normal(size=17) * 3
    blocks = [scores[:4], scores[4:11], scores[11:]]
    m, ell = online_softmax_normalizer(blocks)
    expected_m = np.max(scores)
    expected_ell = np.sum(np.exp(scores - expected_m))
    assert m == expected_m
    np.testing.assert_allclose(ell, expected_ell)
    np.testing.assert_allclose(np.log(ell) + m, np.log(np.sum(np.exp(scores))))


def test_tiled_attention_equals_naive_attention():
    r = rng()
    q = r.normal(size=(9, 5))
    k = r.normal(size=(9, 5))
    v = r.normal(size=(9, 4))
    expected, _ = naive_attention(q, k, v)
    got = tiled_attention(q, k, v, block_size=3)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_tiled_causal_attention_equals_naive_causal_attention():
    r = rng()
    q = r.normal(size=(8, 6))
    k = r.normal(size=(8, 6))
    v = r.normal(size=(8, 3))
    expected, weights = naive_attention(q, k, v, causal=True)
    got = tiled_attention(q, k, v, block_size=3, causal=True)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert np.allclose(weights[np.triu_indices(8, k=1)], 0.0)


def test_tiled_attention_works_with_uneven_blocks():
    r = rng()
    q = r.normal(size=(7, 4))
    k = r.normal(size=(7, 4))
    v = r.normal(size=(7, 2))
    for block_size in (1, 2, 5, 9):
        np.testing.assert_allclose(
            tiled_attention(q, k, v, block_size),
            naive_attention(q, k, v)[0],
            atol=1e-12,
        )


def test_memory_accounting_example_is_asserted():
    assert attention_memory_elements(4096) == {
        "naive_scores": 16_777_216,
        "online_state": 4096,
    }
    assert memory_elements_for_4096()["naive_scores"] == 16_777_216
