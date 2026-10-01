"""Stable softmax and cross-entropy utilities (Chapter 12)."""
import numpy as np

__chapter__ = "softmax-cross-entropy"


# tag::softmax[]
def logsumexp(x, axis=-1, keepdims=False):
    """Stable log(sum(exp(x))) along an axis."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    result = np.max(x, axis=axis, keepdims=True) + np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )
    return result if keepdims else np.squeeze(result, axis=axis)


def log_softmax(logits, temperature=1.0, axis=-1):
    z = np.asarray(logits, dtype=np.float64) / temperature
    return z - logsumexp(z, axis=axis, keepdims=True)


def softmax(logits, temperature=1.0, axis=-1):
    return np.exp(log_softmax(logits, temperature, axis))
# end::softmax[]


# tag::jacobian[]
def softmax_jacobian(probabilities):
    """Jacobian of a single softmax vector p: diag(p) - p p^T."""
    p = np.asarray(probabilities, dtype=np.float64)
    return np.diag(p) - np.outer(p, p)
# end::jacobian[]


def one_hot(labels, num_classes):
    y = np.zeros((len(labels), num_classes), dtype=np.float64)
    y[np.arange(len(labels)), labels] = 1.0
    return y


# tag::fused[]
def smooth_one_hot(labels, num_classes, epsilon=0.0):
    y = one_hot(np.asarray(labels, dtype=np.int64), num_classes)
    return (1 - epsilon) * y + epsilon / num_classes


def softmax_cross_entropy(logits, labels, temperature=1.0,
                          label_smoothing=0.0, z_loss=0.0):
    """Return mean loss and gradient with respect to logits."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = smooth_one_hot(labels, logits.shape[-1], label_smoothing)
    log_p = log_softmax(logits, temperature)
    p = np.exp(log_p)
    batch = logits.shape[0]
    loss = -np.sum(targets * log_p) / batch
    grad = (p - targets) / (batch * temperature)
    if z_loss:
        log_z = logsumexp(logits, axis=-1)
        loss = loss + z_loss * np.mean(log_z ** 2)
        grad = grad + (2 * z_loss / batch) * log_z[:, None] * softmax(logits)
    return float(loss), grad
# end::fused[]


# tag::extras[]
def sigmoid_as_two_class_softmax(logit):
    """The positive-class probability of softmax([0, logit])."""
    pairs = np.stack([np.zeros_like(logit), logit], axis=-1)
    return softmax(pairs)[..., 1]
# end::extras[]
