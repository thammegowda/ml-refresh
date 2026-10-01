"""Perplexity and mutual information (Chapter 7)."""
import numpy as np

from scratch.information import entropy, kl_divergence


# tag::perplexity[]
def perplexity(token_log_probs):
    """exp of the average negative log-likelihood per token (natural logs)."""
    return float(np.exp(-np.mean(token_log_probs)))


def bits_per_token(token_log_probs):
    """The same average, measured in bits."""
    return float(-np.mean(token_log_probs) / np.log(2))
# end::perplexity[]


# tag::mutual-information[]
def mutual_information(joint):
    """I(X; Y) = KL(p(x, y) || p(x) p(y)) for a joint probability table."""
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    return float(kl_divergence(joint.ravel(), np.outer(px, py).ravel()))


def mutual_information_from_entropies(joint):
    """The same quantity as H(X) + H(Y) - H(X, Y)."""
    marginal_x, marginal_y = joint.sum(axis=1), joint.sum(axis=0)
    return float(entropy(marginal_x) + entropy(marginal_y) - entropy(joint.ravel()))
# end::mutual-information[]
