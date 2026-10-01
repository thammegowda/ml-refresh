"""Worked solution helpers for Chapter 18."""
import numpy as np

from language_models import (
    SYNTHETIC_TEXT,
    average_nll_from_probs,
    make_bigrams,
    smoothed_bigram_probs,
    train_neural_bigram,
    word_ids,
)


# tag::count-example[]
def smoothed_tiny_loss(alpha=1.0):
    ids, vocab = word_ids(SYNTHETIC_TEXT)
    previous, target = make_bigrams(ids)
    probs = smoothed_bigram_probs(ids, len(vocab), alpha)
    return average_nll_from_probs(probs, previous, target)
# end::count-example[]


# tag::training-example[]
def trained_tiny_losses():
    ids, vocab = word_ids(SYNTHETIC_TEXT)
    previous, target = make_bigrams(ids)
    _W, losses = train_neural_bigram(previous, target, len(vocab))
    return np.array([losses[0], losses[-1]])
# end::training-example[]
