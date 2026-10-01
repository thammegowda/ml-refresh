"""Logistic regression with binary cross-entropy gradients (Chapter 9)."""
import numpy as np


# tag::logistic-gradient[]
def sigmoid(z):
    positive = z >= 0
    out = np.empty_like(z, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def logistic_loss_and_grads(X, y, w, b):
    logits = X @ w + b
    p = sigmoid(logits)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    grad_logits = (p - y) / len(X)
    grad_w = X.T @ grad_logits
    grad_b = grad_logits.sum()
    return float(loss), grad_w, float(grad_b)
# end::logistic-gradient[]


# tag::minibatch-sgd[]
def fit_logistic_minibatch(X, y, w, b, steps, lr, batch_size, rng):
    w = np.array(w, dtype=np.float32, copy=True)
    b = np.array(b, dtype=np.float32)
    losses = []
    for _ in range(steps):
        idx = rng.integers(0, len(X), size=batch_size)
        _, grad_w, grad_b = logistic_loss_and_grads(X[idx], y[idx], w, b)
        w -= lr * grad_w.astype(np.float32)
        b -= np.float32(lr * grad_b)
        losses.append(logistic_loss_and_grads(X, y, w, b)[0])
    return w, float(b), np.array(losses, dtype=np.float32)
# end::minibatch-sgd[]
