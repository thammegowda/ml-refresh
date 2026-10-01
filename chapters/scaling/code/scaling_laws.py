import numpy as np


# tag::compute[]
def transformer_training_compute(parameters, tokens):
    """Approximate training FLOPs for a dense Transformer."""
    forward = 2 * parameters * tokens
    backward = 4 * parameters * tokens
    return forward + backward, forward, backward
# end::compute[]


# tag::fit-power[]
def fit_power_law(x, y):
    """Fit y = coefficient * x ** (-exponent) in log space."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(np.log(x), np.log(y), deg=1)
    return float(np.exp(intercept)), float(-slope)
# end::fit-power[]


CHINCHILLA = {
    "E": 1.69,
    "A": 406.4,
    "B": 410.7,
    "alpha": 0.34,
    "beta": 0.28,
}


# tag::chinchilla[]
def chinchilla_loss(parameters, tokens, constants=CHINCHILLA):
    """Approach-3 Chinchilla loss fit."""
    return (constants["E"]
            + constants["A"] / parameters ** constants["alpha"]
            + constants["B"] / tokens ** constants["beta"])


def chinchilla_optimal_allocation(compute, constants=CHINCHILLA):
    """Minimize the parametric loss under compute ~= 6ND."""
    a, b = constants["alpha"], constants["beta"]
    A, B = constants["A"], constants["B"]
    product = compute / 6
    parameters = ((a * A) / (b * B) * product ** b) ** (1 / (a + b))
    tokens = product / parameters
    return parameters, tokens


def chinchilla_rule_tokens(parameters):
    """The common Chinchilla rule of thumb: about 20 tokens per parameter."""
    return 20 * parameters
# end::chinchilla[]
