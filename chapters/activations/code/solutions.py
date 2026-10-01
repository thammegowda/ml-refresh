"""Worked solutions to the Chapter 11 exercises."""
import numpy as np

from scratch.activations import gelu_exact, gelu_tanh


# tag::gelu-error[]
def gelu_tanh_max_error(limit=8.0, points=200_001):
    x = np.linspace(-limit, limit, points)
    return float(np.max(np.abs(gelu_exact(x) - gelu_tanh(x))))
# end::gelu-error[]


# tag::gated-width[]
def gated_hidden_width(model_width, mlp_multiplier=4):
    """Hidden width h with 3 d h parameters matching a d -> 4d -> d MLP."""
    return mlp_multiplier * 2 * model_width / 3
# end::gated-width[]
