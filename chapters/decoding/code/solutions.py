"""Solutions for Chapter 39."""
import numpy as np

from scratch.decoding import filtered_distribution, mask_logits, softmax


# tag::digits[]
def digit_only_distribution(logits, vocab):
    """Mask a vocabulary so only tokens made entirely of digits can be sampled."""
    allowed = np.array([token.isdigit() for token in vocab])
    return softmax(mask_logits(logits, allowed))
# end::digits[]


# tag::tiny-grammar[]
def tiny_json_number_mask(prefix, vocab):
    """Allowed tokens for a tiny grammar: '[' digit (',' digit)* ']'."""
    if not prefix:
        return np.array([token == "[" for token in vocab])
    if prefix[-1] in {"[", ","}:
        return np.array([token.isdigit() for token in vocab])
    if prefix[-1].isdigit():
        return np.array([token in {"]", ","} for token in vocab])
    return np.zeros(len(vocab), dtype=bool)


def grammar_step(logits, prefix, vocab):
    return softmax(mask_logits(logits, tiny_json_number_mask(prefix, vocab)))
# end::tiny-grammar[]


# tag::acceptance-proof[]
def speculative_output_probability(p, q):
    """The proof in code: accept mass plus corrected reject mass equals p."""
    accept_as = np.minimum(p, q)
    reject = 1.0 - accept_as.sum()
    residual = np.maximum(0.0, p - q)
    return accept_as + reject * residual / residual.sum()
# end::acceptance-proof[]
