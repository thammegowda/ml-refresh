import numpy as np

from language_models import (
    SYNTHETIC_TEXT,
    average_nll_from_probs,
    make_bigrams,
    make_contexts,
    mlp_logits,
    neural_bigram_loss_and_grad,
    perplexity,
    smoothed_bigram_probs,
    softmax,
    train_neural_bigram,
    word_ids,
)
from scratch.gradcheck import check_gradient
from solutions import smoothed_tiny_loss, trained_tiny_losses


def test_autoregressive_bigrams_and_smoothed_counts():
    ids, vocab = word_ids(SYNTHETIC_TEXT)
    previous, target = make_bigrams(ids)
    probs = smoothed_bigram_probs(ids, len(vocab), alpha=1.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(vocab)))
    assert np.all(probs > 0)
    nll = average_nll_from_probs(probs, previous, target)
    np.testing.assert_allclose(nll, smoothed_tiny_loss())


def test_neural_bigram_gradient_matches_finite_differences():
    ids, vocab = word_ids(SYNTHETIC_TEXT)
    previous, target = make_bigrams(ids)
    rng = np.random.default_rng(18)
    W = 0.01 * rng.standard_normal((len(vocab), len(vocab)))
    loss, grad = neural_bigram_loss_and_grad(W.copy(), previous, target)
    assert np.isfinite(loss)
    check_gradient(
        lambda candidate: neural_bigram_loss_and_grad(candidate, previous, target)[0],
        W,
        grad,
        tolerance=1e-7,
    )


def test_training_reduces_cross_entropy_and_perplexity():
    ids, vocab = word_ids(SYNTHETIC_TEXT)
    previous, target = make_bigrams(ids)
    _W, losses = train_neural_bigram(previous, target, len(vocab))
    assert losses[-1] < losses[0] / 2
    returned = trained_tiny_losses()
    np.testing.assert_allclose(returned, [losses[0], losses[-1]])
    assert perplexity(losses[-1]) < perplexity(losses[0])


def test_mlp_context_window_shapes_and_softmax():
    ids, vocab = word_ids(SYNTHETIC_TEXT)
    contexts, targets = make_contexts(ids, width=3)
    rng = np.random.default_rng(19)
    embed, hidden, vocab_size = 5, 7, len(vocab)
    params = (
        rng.standard_normal((vocab_size, embed)) * 0.1,
        rng.standard_normal((3 * embed, hidden)) * 0.1,
        np.zeros(hidden),
        rng.standard_normal((hidden, vocab_size)) * 0.1,
        np.zeros(vocab_size),
    )
    logits = mlp_logits(params, contexts)
    assert logits.shape == (len(targets), vocab_size)
    np.testing.assert_allclose(softmax(logits).sum(axis=1), np.ones(len(targets)))
