"""Gather, scatter, and masks (Appendix B)."""
import numpy as np


# tag::gather[]
def true_class_scores(logits, labels):
    """logits (N, C), integer labels (N,) -> (N,) holding logits[i, labels[i]]."""
    return logits[np.arange(len(labels)), labels]
# end::gather[]


# tag::embedding[]
def embedding_forward(table, ids):
    """table (V, d) and integer ids of any shape S -> vectors of shape S + (d,)."""
    return table[ids]


def embedding_backward(upstream, ids, vocabulary_size):
    """Gradient of the table: add each upstream vector into the row of its token."""
    gradient = np.zeros((vocabulary_size, upstream.shape[-1]), dtype=upstream.dtype)
    np.add.at(gradient, ids.reshape(-1), upstream.reshape(-1, upstream.shape[-1]))
    return gradient
# end::embedding[]


# tag::causal-mask[]
def causal_mask(T):
    """mask[t, s] is True when position t may look at position s, i.e. s <= t."""
    return np.tril(np.ones((T, T), dtype=bool))
# end::causal-mask[]
