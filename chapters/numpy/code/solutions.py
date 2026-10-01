"""Worked solutions to the Appendix B exercises."""
import numpy as np

from scratch.precision import round_to_bfloat16


# tag::keepdims[]
def normalize_rows_wrong(X):
    return X / np.sum(X, axis=1)          # (N, d) / (N,): aligns N with the d axis
# end::keepdims[]


# tag::scatter[]
def embedding_backward_loop(upstream, ids, vocabulary_size):
    gradient = np.zeros((vocabulary_size, upstream.shape[-1]), dtype=upstream.dtype)
    for token, row in zip(ids.reshape(-1), upstream.reshape(-1, upstream.shape[-1])):
        gradient[token] += row
    return gradient


def embedding_backward_one_hot(upstream, ids, vocabulary_size):
    one_hot = np.eye(vocabulary_size, dtype=upstream.dtype)[ids.reshape(-1)]   # (n, V)
    return one_hot.T @ upstream.reshape(-1, upstream.shape[-1])                 # (V, d)


def embedding_backward_buggy(upstream, ids, vocabulary_size):
    gradient = np.zeros((vocabulary_size, upstream.shape[-1]), dtype=upstream.dtype)
    # Repeated ids collide: only the last write to each row survives.
    gradient[ids.reshape(-1)] += upstream.reshape(-1, upstream.shape[-1])
    return gradient
# end::scatter[]


# tag::heads-wrong[]
def split_heads_wrong(X, heads):
    B, T, width = X.shape
    # Reinterprets memory in order, mixing tokens and heads.
    return X.reshape(B, heads, T, width // heads)
# end::heads-wrong[]


# tag::accumulate[]
def running_sum(value, steps, round_to):
    total = round_to(np.float32(0))
    for _ in range(steps):
        total = round_to(total + round_to(np.float32(value)))
    return float(total)


def bfloat16_vs_float32(value=1e-3, steps=10_000):
    bfloat16_total = running_sum(value, steps, round_to_bfloat16)
    float32_total = running_sum(value, steps, np.float32)
    return bfloat16_total, float32_total
# end::accumulate[]
