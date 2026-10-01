"""Decoding and speculative sampling utilities (Chapter 39)."""
import numpy as np

from scratch.sampling import sample_categorical

__chapter__ = "decoding"


# tag::logits[]
def softmax(logits):
    """Stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def mask_logits(logits, allowed):
    """Set disallowed tokens to -inf before softmax or argmax."""
    logits = np.asarray(logits, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)
    return np.where(allowed, logits, -np.inf)


def filtered_distribution(logits, temperature=1.0, top_k=None, top_p=None,
                          min_p=None, allowed=None):
    """Temperature, top-k, top-p, and min-p as probability transforms."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    if allowed is not None:
        z = mask_logits(z, allowed)
    p = softmax(z)
    keep = np.ones_like(p, dtype=bool)
    if top_k is not None:
        cutoff = np.partition(p, -top_k)[-top_k]
        keep &= p >= cutoff
    if top_p is not None:
        order = np.argsort(-p)
        cumulative = np.cumsum(p[order])
        chosen = np.zeros_like(keep)
        stop = int(np.searchsorted(cumulative, top_p, side="left"))
        chosen[order[:stop + 1]] = True
        keep &= chosen
    if min_p is not None:
        keep &= p >= min_p * np.max(p)
    return np.where(keep, p, 0.0) / np.sum(np.where(keep, p, 0.0))
# end::logits[]


# tag::beam[]
def tiny_beam_search(next_logits, prompt, steps, beam_size=2):
    """Keep the highest log-probability prefixes under a next-logits function."""
    beams = [(tuple(prompt), 0.0)]
    for _ in range(steps):
        candidates = []
        for prefix, score in beams:
            probs = softmax(next_logits(prefix))
            for token, prob in enumerate(probs):
                candidates.append((prefix + (token,), score + np.log(prob)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]
    return beams
# end::beam[]


# tag::speculative[]
def speculative_sample_step(p, q, rng):
    """One exact speculative-sampling step for target p and draft q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    draft = int(sample_categorical(q, rng))
    if rng.random() < min(1.0, p[draft] / max(q[draft], 1e-300)):
        return draft, True
    residual = np.maximum(0.0, p - q)
    return int(sample_categorical(residual / residual.sum(), rng)), False


def speculative_sample(p, q, n, rng):
    """Draw n tokens from p by proposing with q and correcting rejections."""
    draws = np.empty(n, dtype=np.int64)
    accepted = 0
    for i in range(n):
        draws[i], ok = speculative_sample_step(p, q, rng)
        accepted += int(ok)
    return draws, accepted / n
# end::speculative[]


# tag::acceptance[]
def acceptance_probability(p, q):
    """Probability that one proposed token is accepted: sum min(p_i, q_i)."""
    return float(np.minimum(p, q).sum())


def expected_accepted_prefix(acceptance_probs):
    """Expected accepted draft tokens before the first rejection."""
    expected = 0.0
    prefix = 1.0
    for alpha in acceptance_probs:
        prefix *= alpha
        expected += prefix
    return expected
# end::acceptance[]
