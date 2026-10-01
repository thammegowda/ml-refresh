"""Small supervised-learning utilities for Chapter 9."""
import numpy as np


# tag::linear-regression[]
def predict_linear(X, w, b):
    return X @ w + b


def mse_loss(X, y, w, b):
    residual = predict_linear(X, w, b) - y
    return float(np.mean(residual ** 2))


def mse_gradients(X, y, w, b):
    residual = predict_linear(X, w, b) - y
    grad_w = 2.0 * X.T @ residual / len(X)
    grad_b = 2.0 * residual.mean()
    return grad_w, grad_b


def normal_equation(X, y):
    X1 = np.c_[X, np.ones(len(X), dtype=X.dtype)]
    theta = np.linalg.solve(X1.T @ X1, X1.T @ y)
    return theta[:-1], theta[-1]
# end::linear-regression[]


# tag::gradient-descent[]
def gradient_descent_linear(X, y, w, b, steps=200, lr=0.1):
    w = np.array(w, dtype=np.float32, copy=True)
    b = np.array(b, dtype=np.float32)
    losses = []
    for _ in range(steps):
        grad_w, grad_b = mse_gradients(X, y, w, b)
        w -= lr * grad_w.astype(np.float32)
        b -= np.float32(lr * grad_b)
        losses.append(mse_loss(X, y, w, b))
    return w, float(b), np.array(losses, dtype=np.float32)
# end::gradient-descent[]


# tag::polynomial[]
def polynomial_features(x, degree):
    x = np.asarray(x)
    powers = [x ** power for power in range(1, degree + 1)]
    return np.stack(powers, axis=1)
# end::polynomial[]
