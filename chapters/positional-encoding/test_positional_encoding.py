import numpy as np

from solutions import last_query_alibi, permutation_error, shifted_rope_dot
from scratch.positional_encoding import (
    alibi_bias,
    apply_rope,
    attention,
    learned_positions,
    rope_frequencies,
    sinusoidal_positions,
)


def rng(seed=21):
    return np.random.default_rng(seed)


def test_attention_without_positions_is_permutation_equivariant():
    r = rng()
    x = r.normal(size=(5, 4))
    wq = r.normal(size=(4, 4))
    wk = r.normal(size=(4, 4))
    wv = r.normal(size=(4, 3))
    permutation = np.array([2, 4, 1, 3, 0])
    error = permutation_error(x, wq, wk, wv, permutation)
    assert error < 1e-12


def test_absolute_position_tables_have_expected_shapes_and_values():
    table = sinusoidal_positions(3, 4, base=100.0)
    np.testing.assert_allclose(table[0], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(rope_frequencies(4, base=100.0), [1.0, 0.1])
    learned = learned_positions(3, 4, rng())
    assert learned.shape == (3, 4)
    assert learned.dtype == np.float32


def test_rope_inner_product_depends_only_on_relative_position():
    r = rng()
    q = r.normal(size=6)
    k = r.normal(size=6)
    for m, n, shift in [(1, 4, 7), (8, 3, 2), (0, 5, 11)]:
        left, shifted = shifted_rope_dot(q, k, m, n, shift)
        np.testing.assert_allclose(left, shifted, atol=1e-12)
        direct = q @ apply_rope(k, n - m)
        np.testing.assert_allclose(left, direct, atol=1e-12)


def test_rope_vectorized_implementation_and_inverse_rotation():
    r = rng()
    x = r.normal(size=(2, 4, 3, 6))
    positions = np.arange(4)
    rotated = apply_rope(x, positions)
    restored = apply_rope(rotated, positions, inverse=True)
    np.testing.assert_allclose(restored, x, atol=1e-12)


def test_alibi_is_a_linear_distance_bias_for_past_keys():
    bias = alibi_bias(5, 0.25)
    expected_last = np.array([-1.0, -0.75, -0.5, -0.25, -0.0])
    np.testing.assert_allclose(last_query_alibi(5, 0.25), expected_last)
    np.testing.assert_allclose(np.diag(bias), 0.0)
    assert np.all(bias <= 0)


def test_absolute_positions_break_permutation_equivariance():
    r = rng()
    x = r.normal(size=(5, 4))
    pos = sinusoidal_positions(5, 4)
    w = r.normal(size=(4, 4))
    permutation = np.array([2, 4, 1, 3, 0])
    original = attention((x + pos) @ w, (x + pos) @ w, x @ w)
    moved = attention((x[permutation] + pos) @ w, (x[permutation] + pos) @ w, x[permutation] @ w)
    assert np.max(np.abs(moved - original[permutation])) > 1e-3
