import numpy as np

from scratch.positional_encoding import alibi_bias, apply_rope, attention


# tag::equivariance[]
def permutation_error(x, wq, wk, wv, permutation):
    q, k, v = x @ wq, x @ wk, x @ wv
    original = attention(q, k, v)
    xp = x[permutation]
    permuted = attention(xp @ wq, xp @ wk, xp @ wv)
    return np.max(np.abs(permuted - original[permutation]))
# end::equivariance[]


# tag::relative[]
def shifted_rope_dot(q, k, m, n, shift):
    left = apply_rope(q, m) @ apply_rope(k, n)
    shifted = apply_rope(q, m + shift) @ apply_rope(k, n + shift)
    return left, shifted
# end::relative[]


# tag::alibi-row[]
def last_query_alibi(length, slope):
    return alibi_bias(length, slope)[-1]
# end::alibi-row[]
