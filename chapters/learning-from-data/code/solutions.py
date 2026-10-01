"""Tested snippets for Chapter 9 solutions."""
import numpy as np

from regression import normal_equation, polynomial_features, predict_linear


# tag::poly-demo[]
def polynomial_losses(seed=9):
    rng = np.random.default_rng(seed)
    x_train = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    y_clean = 0.5 + x_train - 1.5 * x_train ** 2 + 0.7 * x_train ** 3
    y_train = y_clean + rng.normal(0.0, 0.08, size=len(x_train)).astype(np.float32)
    x_val = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
    y_val = 0.5 + x_val - 1.5 * x_val ** 2 + 0.7 * x_val ** 3
    losses = {}
    for degree in (1, 3, 11):
        X_train = polynomial_features(x_train, degree)
        X_val = polynomial_features(x_val, degree)
        w, b = normal_equation(X_train, y_train)
        losses[degree] = float(np.mean((predict_linear(X_val, w, b) - y_val) ** 2))
    return losses
# end::poly-demo[]
