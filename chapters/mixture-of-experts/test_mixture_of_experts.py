import numpy as np

from solutions import dense_reference_moe, dropped_fraction, skewed_router_logits
from scratch.mixture_of_experts import (
    router_z_loss,
    simulate_bias_balancing,
    sparse_moe_forward,
    switch_load_balancing_loss,
    top_k_router,
)


def rng(seed=27):
    return np.random.default_rng(seed)


def parameters(num_experts=4, d_model=3, d_hidden=5, d_out=2):
    r = rng()
    w1 = r.normal(size=(num_experts, d_model, d_hidden)).astype(np.float32)
    b1 = r.normal(size=(num_experts, d_hidden)).astype(np.float32)
    w2 = r.normal(size=(num_experts, d_hidden, d_out)).astype(np.float32)
    b2 = r.normal(size=(num_experts, d_out)).astype(np.float32)
    return w1, b1, w2, b2


def test_top_k_router_renormalizes_weights_and_keeps_bias_out_of_probabilities():
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 4.0, 3.0]], dtype=np.float64)
    experts, weights, probabilities = top_k_router(logits, 2)
    assert experts.tolist() == [[0, 1], [1, 2]]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    expected = probabilities[0, [0, 1]] / probabilities[0, [0, 1]].sum()
    np.testing.assert_allclose(weights[0], expected)
    biased, biased_weights, biased_probs = top_k_router(logits, 1, [0.0, 0.0, 5.0])
    assert biased[0, 0] == 2
    np.testing.assert_allclose(biased_probs, probabilities)
    np.testing.assert_allclose(biased_weights, 1.0)
    exercise_probabilities = np.array([0.5, 0.3, 0.2])
    selected = exercise_probabilities[[1, 2]]
    np.testing.assert_allclose(selected / selected.sum(), [0.6, 0.4])


def test_sparse_forward_matches_dense_token_loop():
    r = rng()
    x = r.normal(size=(7, 3)).astype(np.float32)
    logits = r.normal(size=(7, 4)).astype(np.float32)
    w1, b1, w2, b2 = parameters()
    experts, weights, _ = top_k_router(logits, 2)
    sparse, dropped = sparse_moe_forward(x, w1, b1, w2, b2, experts, weights)
    dense = dense_reference_moe(x, w1, b1, w2, b2, logits, 2)
    assert not dropped.any()
    np.testing.assert_allclose(sparse, dense, rtol=1e-6, atol=1e-6)


def test_switch_loss_is_smallest_at_uniform_when_load_matches_probability():
    uniform = np.full((8, 4), 0.25)
    chosen = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    loss, load, mean_probability = switch_load_balancing_loss(uniform, chosen)
    np.testing.assert_allclose([loss, *load, *mean_probability], [1.0] + [0.25] * 8)
    collapsed = np.tile([1.0, 0.0, 0.0, 0.0], (8, 1))
    collapsed_loss, collapsed_load, _ = switch_load_balancing_loss(
        collapsed, np.zeros(8, dtype=int)
    )
    np.testing.assert_allclose(collapsed_loss, 4.0)
    assert collapsed_load.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_capacity_drops_assignments_after_each_expert_is_full():
    logits = np.array([[9.0, 0.0], [8.0, 0.0], [7.0, 0.0], [0.0, 3.0]])
    assert dropped_fraction(logits, capacity=2) == 0.25
    r = rng()
    x = r.normal(size=(4, 3)).astype(np.float32)
    w1, b1, w2, b2 = parameters(num_experts=2)
    experts, weights, _ = top_k_router(logits, 1)
    _, dropped = sparse_moe_forward(x, w1, b1, w2, b2, experts, weights, capacity=2)
    assert dropped[:, 0].tolist() == [False, False, True, False]


def test_router_z_loss_penalizes_large_log_partitions():
    small = np.zeros((3, 4))
    large = np.full((3, 4), 5.0)
    np.testing.assert_allclose(router_z_loss(small), np.log(4) ** 2)
    assert router_z_loss(large) > router_z_loss(small)


def test_auxiliary_loss_free_bias_balances_a_skewed_router():
    logits = skewed_router_logits()
    history, bias = simulate_bias_balancing(logits, steps=80, rate=0.05)
    start_gap = history[0].max() - history[0].min()
    end = history[-10:].mean(axis=0)
    end_gap = end.max() - end.min()
    assert start_gap > 0.45
    assert end_gap < 0.10
    assert bias[0] < bias[1:].mean()
