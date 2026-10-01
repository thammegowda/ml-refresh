"""Tiny language-modeling utilities for Chapter 18."""
import numpy as np

SYNTHETIC_TEXT = "time flies like time fruit flies like fruit time flies"


def word_ids(text=SYNTHETIC_TEXT):
    words = text.split()
    vocab = {word: i for i, word in enumerate(sorted(set(words)))}
    return np.array([vocab[word] for word in words], dtype=np.int64), vocab


def make_bigrams(ids):
    return np.asarray(ids[:-1]), np.asarray(ids[1:])


# tag::count-bigram[]
def smoothed_bigram_probs(ids, vocab_size, alpha=1.0):
    """Estimate p(next | previous) with add-alpha smoothing."""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    previous, target = make_bigrams(ids)
    np.add.at(counts, (previous, target), 1.0)
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * vocab_size)


def average_nll_from_probs(probs, previous, target):
    """Mean negative log-probability assigned to observed bigrams."""
    return -np.mean(np.log(probs[previous, target]))
# end::count-bigram[]


# tag::neural-bigram[]
def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def neural_bigram_loss_and_grad(W, previous, target):
    """Cross-entropy loss and gradient for logits W[previous]."""
    logits = W[previous]
    probs = softmax(logits)
    n = len(target)
    loss = -np.mean(np.log(probs[np.arange(n), target]))
    grad_logits = probs
    grad_logits[np.arange(n), target] -= 1.0
    grad_logits /= n
    grad_W = np.zeros_like(W)
    np.add.at(grad_W, previous, grad_logits)
    return loss, grad_W
# end::neural-bigram[]


# tag::train-bigram[]
def train_neural_bigram(previous, target, vocab_size, steps=200, lr=2.0, seed=18):
    rng = np.random.default_rng(seed)
    W = (0.01 * rng.standard_normal((vocab_size, vocab_size))).astype(np.float32)
    losses = []
    for _step in range(steps):
        loss, grad_W = neural_bigram_loss_and_grad(W, previous, target)
        W -= lr * grad_W.astype(np.float32)
        losses.append(float(loss))
    return W, losses
# end::train-bigram[]


# tag::mlp[]
def make_contexts(ids, width):
    """Return fixed-width histories and next-token targets."""
    ids = np.asarray(ids, dtype=np.int64)
    contexts = np.stack([ids[i:i + width] for i in range(len(ids) - width)])
    targets = ids[width:]
    return contexts, targets


def mlp_logits(params, contexts):
    """Bengio-style fixed-window MLP language model forward pass."""
    C, W1, b1, W2, b2 = params
    embedded = C[contexts].reshape(contexts.shape[0], -1)
    hidden = np.tanh(embedded @ W1 + b1)
    return hidden @ W2 + b2
# end::mlp[]


def perplexity(nll):
    return float(np.exp(nll))
