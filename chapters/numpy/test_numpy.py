import numpy as np

from arrays import float_limits, make_data, views_and_copies
from broadcasting import add_bias, normalize_rows, pairwise_squared_distances, pairwise_squared_distances_fast
from heads import all_pair_dot_products, all_pair_dot_products_matmul, merge_heads, split_heads
from indexing import causal_mask, embedding_backward, embedding_forward, true_class_scores
from solutions import (bfloat16_vs_float32, embedding_backward_buggy, embedding_backward_loop, embedding_backward_one_hot,
                       normalize_rows_wrong, running_sum, split_heads_wrong)
from stability import log_softmax, logsumexp, naive_logsumexp, sigmoid, softplus
from scratch.arrays import unbroadcast
from scratch.precision import round_to_bfloat16
from scratch.gradcheck import check_gradient

rng = np.random.default_rng(2026)


def test_float_limits_match_the_table_in_the_text():
    rows = {name: rest for name, *rest in float_limits()}
    assert rows["float16"] == [2 ** -10, 65504.0, 2 ** -14]
    assert rows["float32"] == [2 ** -23, float(np.finfo(np.float32).max), 2 ** -126]
    assert rows["float64"][0] == 2 ** -52
    np.testing.assert_allclose([np.log(65504.0), np.log(rows["float32"][1])], [11.09, 88.72], atol=0.005)


def test_basic_slices_are_views_and_integer_indexing_copies():
    X = views_and_copies()
    np.testing.assert_array_equal(X, [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])


def test_one_generator_makes_runs_reproducible():
    first, second = make_data(seed=7), make_data(seed=7)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert first[0].dtype == np.float32 and sorted(first[2]) == list(range(4))


def test_broadcast_shapes_exercise():
    A, B, C, D = (4, 1, 3), (5, 1), (3,), (4, 3)
    assert np.broadcast_shapes(A, B) == (4, 5, 3)
    assert np.broadcast_shapes(A, C) == (4, 1, 3)
    assert np.broadcast_shapes(B, C) == (5, 3)
    assert np.broadcast_shapes(A, D) == (4, 4, 3)
    try:
        np.broadcast_shapes(D, B)
    except ValueError:
        pass
    else:
        raise AssertionError("(4, 3) and (5, 1) should not broadcast")


def test_bias_and_pairwise_distances():
    X = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(add_bias(X, np.array([10, 20, 30], dtype=np.float32)), [[10, 21, 32], [13, 24, 35]])
    A, B = rng.standard_normal((5, 3)), rng.standard_normal((7, 3))
    D = pairwise_squared_distances(A, B)
    assert D.shape == (5, 7)
    np.testing.assert_allclose(D, pairwise_squared_distances_fast(A, B), atol=1e-12)
    np.testing.assert_allclose(D[2, 4], np.sum((A[2] - B[4]) ** 2))
    np.testing.assert_allclose(np.diag(pairwise_squared_distances_fast(A, A)), 0, atol=1e-12)


def test_keepdims_exercise():
    X = rng.random((4, 3)) + 0.1
    np.testing.assert_allclose(normalize_rows(X).sum(axis=1), 1)
    try:
        normalize_rows_wrong(X)
    except ValueError:
        pass
    else:
        raise AssertionError("(4, 3) / (4,) should not broadcast")
    square = rng.random((3, 3)) + 0.1
    wrong = normalize_rows_wrong(square)
    np.testing.assert_allclose(wrong, square / square.sum(axis=1)[None, :])
    assert not np.allclose(wrong.sum(axis=1), 1)


def test_bias_gradient_is_the_sum_over_broadcast_rows():
    X, G = rng.standard_normal((6, 4)), rng.standard_normal((6, 4))
    b = rng.standard_normal(4)
    gradient = unbroadcast(G, b.shape)
    np.testing.assert_allclose(gradient, G.sum(axis=0))
    check_gradient(lambda v: float(np.sum(G * (X + v))), b, gradient)


def test_gather_embedding_gradient_and_the_scatter_bug():
    logits, labels = rng.standard_normal((4, 5)), np.array([4, 0, 0, 2])
    np.testing.assert_array_equal(true_class_scores(logits, labels), [logits[0, 4], logits[1, 0], logits[2, 0], logits[3, 2]])
    table, ids = rng.standard_normal((6, 3)), np.array([[1, 4, 1], [1, 0, 5]])
    upstream = rng.standard_normal((2, 3, 3))
    assert embedding_forward(table, ids).shape == (2, 3, 3)
    gradient = embedding_backward(upstream, ids, 6)
    np.testing.assert_allclose(gradient, embedding_backward_loop(upstream, ids, 6))
    np.testing.assert_allclose(gradient, embedding_backward_one_hot(upstream, ids, 6))
    check_gradient(lambda t: float(np.sum(upstream * t[ids])), table, gradient)
    np.testing.assert_allclose(gradient[1], upstream[0, 0] + upstream[0, 2] + upstream[1, 0])
    assert not np.allclose(embedding_backward_buggy(upstream, ids, 6)[1], gradient[1])
    np.testing.assert_allclose(embedding_backward_buggy(upstream, ids, 6)[[0, 4, 5]], gradient[[0, 4, 5]])


def test_causal_mask_allows_only_the_past():
    mask = causal_mask(4)
    for t in range(4):
        for s in range(4):
            assert mask[t, s] == (s <= t)


def test_head_reshapes_and_batched_products():
    X = rng.standard_normal((2, 5, 12))
    heads = split_heads(X, 3)
    assert heads.shape == (2, 3, 5, 4)
    np.testing.assert_array_equal(heads[1, 2, 3], X[1, 3, 8:12])
    np.testing.assert_array_equal(merge_heads(heads), X)
    assert not np.array_equal(split_heads_wrong(X, 3), heads)
    Q, K = split_heads(X, 3), split_heads(rng.standard_normal((2, 5, 12)), 3)
    S = all_pair_dot_products(Q, K)
    np.testing.assert_allclose(S, all_pair_dot_products_matmul(Q, K))
    np.testing.assert_allclose(S[1, 2, 3, 4], Q[1, 2, 3] @ K[1, 2, 4])


def test_logsumexp_is_stable_bounded_and_differentiates_to_softmax():
    z = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    with np.errstate(over="ignore"):
        assert np.isinf(naive_logsumexp(z))
    np.testing.assert_allclose(logsumexp(z), 1002.4076, rtol=1e-6)
    Z = rng.standard_normal((5, 7)) * 10
    lse = logsumexp(Z, axis=1)
    np.testing.assert_allclose(lse, np.log(np.sum(np.exp(Z), axis=1)))
    np.testing.assert_allclose(logsumexp(Z + 3.5, axis=1), lse + 3.5)
    assert np.all(Z.max(axis=1) <= lse) and np.all(lse <= Z.max(axis=1) + np.log(7) + 1e-12)
    with np.errstate(divide="ignore"):
        assert logsumexp(np.full(3, -np.inf)) == -np.inf
    np.testing.assert_allclose(np.exp(log_softmax(Z)).sum(axis=1), 1)
    x = rng.standard_normal(6)
    check_gradient(lambda v: float(logsumexp(v)), x, np.exp(log_softmax(x)))


def test_sigmoid_and_softplus_never_overflow():
    x = np.array([-1000.0, -30.0, -1.0, 0.0, 1.0, 30.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s, p = sigmoid(x), softplus(x)
    np.testing.assert_allclose(s[1:-1], 1 / (1 + np.exp(-x[1:-1])))
    assert s[0] == 0.0 and s[-1] == 1.0 and s[3] == 0.5
    np.testing.assert_allclose(p[1:-1], np.log1p(np.exp(x[1:-1])))
    assert p[-1] == 1000.0 and softplus(np.float32(100)) == np.float32(100)
    v = rng.standard_normal(5)
    check_gradient(lambda u: float(np.sum(softplus(u))), v, sigmoid(v))


def test_central_differences_are_second_order_and_forward_differences_first_order():
    f, x, exact = np.sin, 1.0, np.cos(1.0)
    central = [abs((f(x + h) - f(x - h)) / (2 * h) - exact) for h in (1e-2, 1e-3)]
    forward = [abs((f(x + h) - f(x)) / h - exact) for h in (1e-2, 1e-3)]
    np.testing.assert_allclose(central[0] / central[1], 100, rtol=0.01)
    np.testing.assert_allclose(forward[0] / forward[1], 10, rtol=0.1)


def test_low_precision_accumulation_stalls():
    bf16, fp32 = bfloat16_vs_float32()
    assert bf16 == 0.5
    np.testing.assert_allclose(fp32, 10.0, rtol=1e-4)
    assert running_sum(1e-3, 10_000, np.float16) == 4.0
    assert running_sum(1e-3, 382, round_to_bfloat16) < 0.5
    assert running_sum(1e-3, 383, round_to_bfloat16) == 0.5
    assert float(np.float16(1e-3)) > 2.0 ** -9 / 2 and float(round_to_bfloat16(np.float32(1e-3))) > 2.0 ** -9 / 2
