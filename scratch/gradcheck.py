"""Finite-difference gradient checking (Appendix B)."""
import numpy as np

__chapter__ = "numpy"


# tag::numerical-gradient[]
def numerical_gradient(f, x, h=1e-5):
    """Central-difference estimate of the gradient of a scalar function f at x."""
    x = np.array(x, dtype=np.float64)  # a float64 copy: never perturb the caller's x
    gradient = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + h
        plus = f(x)
        x[index] = original - h
        minus = f(x)
        x[index] = original
        gradient[index] = (plus - minus) / (2 * h)
    return gradient
# end::numerical-gradient[]


# tag::relative-error[]
def relative_error(a, b, floor=1e-12):
    """Largest elementwise |a - b| / (|a| + |b|), guarded against 0 / 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), floor)))
# end::relative-error[]


# tag::check-gradient[]
def check_gradient(f, x, analytic, h=1e-5, tolerance=1e-7):
    """Raise if an analytic gradient disagrees with central differences."""
    error = relative_error(analytic, numerical_gradient(f, x, h))
    if error > tolerance:
        raise AssertionError(f"gradient check failed: relative error {error:.2e}"
                             f" > {tolerance:.0e}")
    return error
# end::check-gradient[]
