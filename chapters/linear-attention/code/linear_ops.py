import numpy as np


# tag::linear-attention[]
def feature_map(x):
    """Positive ELU+1 feature map."""
    x = np.asarray(x)
    return np.where(x > 0, x + 1, np.exp(x))


def parallel_linear_attention(q, k, v, eps=1e-8):
    """Causal linear attention computed from the full lower triangle."""
    q_phi, k_phi = feature_map(q), feature_map(k)
    scores = q_phi @ k_phi.T
    scores *= np.tri(q.shape[0], dtype=scores.dtype)
    numerator = scores @ v
    denominator = scores.sum(axis=-1, keepdims=True)
    return numerator / np.maximum(denominator, eps)


def recurrent_linear_attention(q, k, v, eps=1e-8):
    """Causal linear attention using the recurrent state S and normalizer c."""
    q_phi, k_phi = feature_map(q), feature_map(k)
    state = np.zeros((k_phi.shape[1], v.shape[1]), dtype=q.dtype)
    normalizer = np.zeros(k_phi.shape[1], dtype=q.dtype)
    outputs = []
    for qt, kt, vt in zip(q_phi, k_phi, v):
        state += np.outer(kt, vt)
        normalizer += kt
        numerator = qt @ state
        denominator = qt @ normalizer
        outputs.append(numerator / max(denominator, eps))
    return np.array(outputs)
# end::linear-attention[]


# tag::delta[]
def delta_step(state, key_feature, value, beta=1.0):
    """Error-correcting associative-memory update."""
    prediction = key_feature @ state
    error = value - prediction
    next_state = state + beta * np.outer(key_feature, error)
    return next_state, error


def gated_delta_step(state, key_feature, value, beta=1.0, gate=0.95):
    """Forget part of the old state, then write the current prediction error."""
    decayed = gate * state
    prediction = key_feature @ decayed
    error = value - prediction
    next_state = decayed + beta * np.outer(key_feature, error)
    return next_state, error
# end::delta[]


# tag::ssm[]
def selective_state_space(x, a, b, c, delta):
    """A tiny diagonal selective SSM recurrence."""
    state = np.zeros_like(a, dtype=x.dtype)
    outputs = []
    for xt, bt, ct, dt in zip(x, b, c, delta):
        decay = np.exp(dt * a)
        state = decay * state + dt * bt * xt
        outputs.append(ct @ state)
    return np.array(outputs)
# end::ssm[]
