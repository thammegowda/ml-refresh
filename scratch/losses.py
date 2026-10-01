"""Loss functions and gradients (Chapter 13)."""
import numpy as np

from scratch.softmax_cross_entropy import log_softmax, softmax

__chapter__ = "losses"


def _mean_loss_and_grad(per_example, grad_per_element):
    size = np.size(grad_per_element)
    return float(np.mean(per_example)), grad_per_element / size


# tag::regression[]
def mse_loss(prediction, target):
    residual = np.asarray(prediction, dtype=np.float64) - target
    return _mean_loss_and_grad(residual ** 2, 2 * residual)


def mae_loss(prediction, target):
    residual = np.asarray(prediction, dtype=np.float64) - target
    return _mean_loss_and_grad(np.abs(residual), np.sign(residual))


def huber_loss(prediction, target, delta=1.0):
    residual = np.asarray(prediction, dtype=np.float64) - target
    abs_r = np.abs(residual)
    quadratic = abs_r <= delta
    loss = np.where(quadratic, 0.5 * residual ** 2,
                    delta * (abs_r - 0.5 * delta))
    grad = np.where(quadratic, residual, delta * np.sign(residual))
    return _mean_loss_and_grad(loss, grad)
# end::regression[]


# tag::bce-focal[]
def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def bce_with_logits(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    loss = np.maximum(logits, 0) - logits * targets
    loss = loss + np.log1p(np.exp(-np.abs(logits)))
    return _mean_loss_and_grad(loss, sigmoid(logits) - targets)


def binary_focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    sign = 2 * targets - 1
    log_pt = -np.logaddexp(0, -sign * logits)
    pt = np.exp(log_pt)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * (1 - pt) ** gamma * log_pt
    dloss_dpt = alpha_t * gamma * (1 - pt) ** (gamma - 1) * log_pt
    dloss_dpt = dloss_dpt - alpha_t * (1 - pt) ** gamma / pt
    grad = dloss_dpt * sign * pt * (1 - pt)
    return _mean_loss_and_grad(loss, grad)
# end::bce-focal[]


# tag::divergences[]
def kl_forward_logits(target_probs, logits):
    p = np.asarray(target_probs, dtype=np.float64)
    log_q = log_softmax(logits)
    loss = np.sum(p * (np.log(p) - log_q), axis=-1)
    return float(np.mean(loss)), (np.exp(log_q) - p) / logits.shape[0]


def kl_reverse_logits(logits, target_probs):
    q = softmax(logits)
    log_q = log_softmax(logits)
    log_p = np.log(np.asarray(target_probs, dtype=np.float64))
    values = log_q - log_p + 1
    loss = np.sum(q * (log_q - log_p), axis=-1)
    centered = values - np.sum(q * values, axis=-1, keepdims=True)
    return float(np.mean(loss)), q * centered / logits.shape[0]


def jensen_shannon(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p) - np.log(m))) + 0.5 * np.sum(
        q * (np.log(q) - np.log(m))
    )
# end::divergences[]


# tag::distillation[]
def distillation_loss(student_logits, teacher_logits, temperature=2.0, scale=True):
    teacher = softmax(teacher_logits, temperature=temperature)
    student_log_p = log_softmax(student_logits, temperature=temperature)
    batch = student_logits.shape[0]
    factor = temperature ** 2 if scale else 1.0
    loss = -factor * np.sum(teacher * student_log_p) / batch
    student = np.exp(student_log_p)
    grad = factor * (student - teacher) / (batch * temperature)
    return float(loss), grad
# end::distillation[]
