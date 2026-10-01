"""Tiny Mixture-of-Experts utilities for Chapter 27."""
import numpy as np

__chapter__ = "mixture-of-experts"


# tag::router[]
def softmax(logits):
    """Row-wise softmax."""
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def top_k_router(logits, k, selection_bias=None):
    """Return top-k expert indices and renormalized combine weights."""
    logits = np.asarray(logits)
    scores = logits if selection_bias is None else logits + selection_bias
    order = np.argsort(scores, axis=-1)[:, ::-1]
    experts = order[:, :k]
    probabilities = softmax(logits)
    chosen = np.take_along_axis(probabilities, experts, axis=-1)
    weights = chosen / chosen.sum(axis=-1, keepdims=True)
    return experts, weights, probabilities
# end::router[]


# tag::forward[]
def expert_mlp(x, w1, b1, w2, b2):
    """One ReLU feed-forward expert."""
    hidden = np.maximum(x @ w1 + b1, 0)
    return hidden @ w2 + b2


def sparse_moe_forward(x, w1, b1, w2, b2, experts, weights, capacity=None):
    """Apply selected experts; optionally drop assignments beyond capacity."""
    num_tokens, _ = x.shape
    num_experts, _, out_dim = w2.shape
    y = np.zeros((num_tokens, out_dim), dtype=x.dtype)
    dropped = np.zeros(experts.shape, dtype=bool)
    for expert in range(num_experts):
        token, slot = np.nonzero(experts == expert)
        if capacity is not None and len(token) > capacity:
            dropped[token[capacity:], slot[capacity:]] = True
            token, slot = token[:capacity], slot[:capacity]
        if len(token) == 0:
            continue
        out = expert_mlp(x[token], w1[expert], b1[expert], w2[expert], b2[expert])
        y[token] += weights[token, slot, None] * out
    return y, dropped
# end::forward[]


# tag::losses[]
def switch_load_balancing_loss(probabilities, chosen_expert):
    """Switch loss E * sum_i f_i P_i for top-1 routing."""
    probabilities = np.asarray(probabilities)
    num_experts = probabilities.shape[-1]
    counts = np.bincount(chosen_expert, minlength=num_experts)
    load = counts / chosen_expert.size
    mean_probability = probabilities.mean(axis=0)
    loss = num_experts * np.sum(load * mean_probability)
    return float(loss), load, mean_probability


def router_z_loss(logits):
    """Mean squared log-partition of router logits."""
    logits = np.asarray(logits)
    maximum = logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(logits - maximum).sum(axis=-1)) + maximum[:, 0]
    return float(np.mean(log_z ** 2))
# end::losses[]


# tag::bias[]
def update_selection_bias(bias, load, target, rate):
    """Lower overloaded experts and raise underloaded experts."""
    return bias - rate * np.sign(load - target)


def simulate_bias_balancing(base_logits, steps=80, rate=0.05):
    """Balance a fixed skewed router by changing only its selection bias."""
    bias = np.zeros(base_logits.shape[1], dtype=base_logits.dtype)
    target = np.full(base_logits.shape[1], 1 / base_logits.shape[1])
    history = []
    for _ in range(steps):
        experts, _, probabilities = top_k_router(base_logits, 1, bias)
        _, load, _ = switch_load_balancing_loss(probabilities, experts[:, 0])
        history.append(load)
        bias = update_selection_bias(bias, load, target, rate)
    return np.array(history), bias
# end::bias[]
