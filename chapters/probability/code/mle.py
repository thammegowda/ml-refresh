"""Maximum likelihood, and the losses it produces (Chapter 6)."""
import numpy as np

from distributions import gaussian_log_pdf


# tag::gaussian-mle[]
def gaussian_mle(x):
    """Maximum-likelihood mean and standard deviation of 1-D samples."""
    mean = x.mean()
    return mean, np.sqrt(np.mean((x - mean) ** 2))    # divides by n, not n - 1


def gaussian_nll(x, mean, std):
    """Average negative log-likelihood of samples x under N(mean, std^2)."""
    return -np.mean(gaussian_log_pdf(x, mean, std))
# end::gaussian-mle[]


# tag::losses[]
def gaussian_regression_nll(y, prediction, std=1.0):
    """Targets y ~ N(prediction, std^2): mean squared error / (2 std^2) + constant."""
    return -np.mean(gaussian_log_pdf(y, prediction, std))


def bernoulli_nll(y, p):
    """Binary labels y ~ Bernoulli(p): the binary cross-entropy."""
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def categorical_nll(labels, probabilities):
    """Class labels ~ Categorical(probabilities[i]): the cross-entropy."""
    return -np.mean(np.log(probabilities[np.arange(len(labels)), labels]))
# end::losses[]
