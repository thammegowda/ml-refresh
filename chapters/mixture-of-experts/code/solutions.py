import numpy as np

from scratch.mixture_of_experts import sparse_moe_forward, top_k_router


# tag::dense-loop[]
def dense_reference_moe(x, w1, b1, w2, b2, logits, k):
    """Token-by-token reference used to test the sparse implementation."""
    experts, weights, _ = top_k_router(logits, k)
    out_dim = w2.shape[-1]
    y = np.zeros((x.shape[0], out_dim), dtype=x.dtype)
    for token in range(x.shape[0]):
        for slot, expert in enumerate(experts[token]):
            hidden = np.maximum(x[token] @ w1[expert] + b1[expert], 0)
            y[token] += weights[token, slot] * (hidden @ w2[expert] + b2[expert])
    return y
# end::dense-loop[]


# tag::skewed-logits[]
def skewed_router_logits(num_tokens=240, num_experts=4, seed=0):
    """Synthetic logits: each expert has tokens, but expert 0 starts advantaged."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=0.05, size=(num_tokens, num_experts))
    preferred = np.arange(num_tokens) % num_experts
    logits[np.arange(num_tokens), preferred] += 0.8
    logits[:, 0] += 1.2
    return logits.astype(np.float32)
# end::skewed-logits[]


# tag::capacity-demo[]
def dropped_fraction(logits, capacity):
    experts, weights, _ = top_k_router(logits, 1)
    x = np.ones((logits.shape[0], 1), dtype=np.float32)
    w1 = np.ones((logits.shape[1], 1, 1), dtype=np.float32)
    b1 = np.zeros((logits.shape[1], 1), dtype=np.float32)
    w2 = np.ones((logits.shape[1], 1, 1), dtype=np.float32)
    b2 = np.zeros((logits.shape[1], 1), dtype=np.float32)
    _, dropped = sparse_moe_forward(x, w1, b1, w2, b2, experts, weights, capacity)
    return dropped.mean()
# end::capacity-demo[]
