import numpy as np

from linear_ops import recurrent_linear_attention


# tag::decode[]
def decode_one(q_t, k_t, v_t, state, normalizer, feature_map, eps=1e-8):
    """One-token update for linear-attention decoding."""
    qt, kt = feature_map(q_t), feature_map(k_t)
    state = state + np.outer(kt, v_t)
    normalizer = normalizer + kt
    y_t = (qt @ state) / max(qt @ normalizer, eps)
    return y_t, state, normalizer
# end::decode[]


# tag::constant-space[]
def decoding_state_size(d_feature, d_value):
    """Number of scalars cached by one linear-attention layer."""
    return d_feature * d_value + d_feature
# end::constant-space[]


# tag::prefix-check[]
def last_token_from_prefix(q, k, v):
    """The recurrent form can process a prefix and keep only its final state."""
    return recurrent_linear_attention(q, k, v)[-1]
# end::prefix-check[]
