"""Activation functions and gated feed-forward blocks (Chapter 11)."""
import math

import numpy as np

__chapter__ = "activations"

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_GELU_COEFF = 0.044715


def _erf(x):
    return np.vectorize(math.erf, otypes=[float])(x)


# tag::elementwise[]
def sigmoid(x):
    x = np.asarray(x)
    positive = x >= 0
    z = np.exp(-np.abs(x))
    return np.where(positive, 1 / (1 + z), z / (1 + z))


def sigmoid_grad(x):
    y = sigmoid(x)
    return y * (1 - y)


def tanh_grad(x):
    y = np.tanh(x)
    return 1 - y * y


def relu(x):
    return np.maximum(x, 0)


def relu_grad(x):
    return (np.asarray(x) > 0).astype(float)


def leaky_relu(x, negative_slope=0.01):
    return np.where(np.asarray(x) >= 0, x, negative_slope * np.asarray(x))


def leaky_relu_grad(x, negative_slope=0.01):
    return np.where(np.asarray(x) >= 0, 1.0, negative_slope)
# end::elementwise[]


# tag::gelu[]
def normal_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(x) ** 2)


def normal_cdf(x):
    return 0.5 * (1 + _erf(np.asarray(x) / _SQRT_2))


def gelu_exact(x):
    x = np.asarray(x)
    return x * normal_cdf(x)


def gelu_exact_grad(x):
    x = np.asarray(x)
    return normal_cdf(x) + x * normal_pdf(x)


def gelu_tanh(x):
    x = np.asarray(x)
    u = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x ** 3)
    return 0.5 * x * (1 + np.tanh(u))


def gelu_tanh_grad(x):
    x = np.asarray(x)
    u = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x ** 3)
    du = _SQRT_2_OVER_PI * (1 + 3 * _GELU_COEFF * x ** 2)
    sech2 = 1 - np.tanh(u) ** 2
    return 0.5 * (1 + np.tanh(u)) + 0.5 * x * sech2 * du
# end::gelu[]


# tag::smooth[]
def silu(x):
    x = np.asarray(x)
    return x * sigmoid(x)


def silu_grad(x):
    x = np.asarray(x)
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softplus(x):
    x = np.asarray(x)
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def softplus_grad(x):
    return sigmoid(x)
# end::smooth[]


_ACTIVATIONS = {
    "glu": (sigmoid, sigmoid_grad),
    "reglu": (relu, relu_grad),
    "geglu": (gelu_exact, gelu_exact_grad),
    "swiglu": (silu, silu_grad),
}


# tag::gated[]
def gated_ffn(x, W, V, W2, kind="swiglu"):
    """(act(x @ W) * (x @ V)) @ W2, with rows as examples."""
    act, _ = _ACTIVATIONS[kind]
    return (act(x @ W) * (x @ V)) @ W2


def gated_ffn_backward(x, W, V, W2, grad_y, kind="swiglu"):
    act, act_grad = _ACTIVATIONS[kind]
    z, u = x @ W, x @ V
    a, h = act(z), act(z) * u
    grad_W2 = h.T @ grad_y
    grad_h = grad_y @ W2.T
    grad_z = grad_h * u * act_grad(z)
    grad_u = grad_h * a
    grad_W = x.T @ grad_z
    grad_V = x.T @ grad_u
    grad_x = grad_z @ W.T + grad_u @ V.T
    return grad_x, grad_W, grad_V, grad_W2
# end::gated[]
