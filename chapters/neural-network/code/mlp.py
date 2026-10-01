import numpy as np


# tag::initialize[]
def initialize(seed=0, input_dim=5, hidden_dim=10, output_dim=4, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return {
        "W1": (rng.normal(size=(input_dim, hidden_dim))
               * np.sqrt(2.0 / input_dim)).astype(dtype),
        "b1": np.zeros(hidden_dim, dtype=dtype),
        "W2": (rng.normal(size=(hidden_dim, output_dim))
               * np.sqrt(1.0 / hidden_dim)).astype(dtype),
        "b2": np.zeros(output_dim, dtype=dtype),
    }
# end::initialize[]


# tag::forward-backward[]
def softmax_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probs)
    loss = -log_probs[np.arange(len(labels)), labels].mean()
    return loss, probabilities


def forward(params, inputs, labels):
    z1 = inputs @ params["W1"] + params["b1"]
    hidden = np.maximum(z1, 0)
    logits = hidden @ params["W2"] + params["b2"]
    loss, probabilities = softmax_cross_entropy(logits, labels)
    return loss, {"X": inputs, "Z1": z1, "H": hidden, "P": probabilities}


def backward(params, cache, labels):
    grad_logits = cache["P"].copy()
    grad_logits[np.arange(len(labels)), labels] -= 1
    grad_logits /= len(labels)

    grad_W2 = cache["H"].T @ grad_logits
    grad_b2 = grad_logits.sum(axis=0)
    grad_hidden = grad_logits @ params["W2"].T
    grad_z1 = grad_hidden * (cache["Z1"] > 0)
    return {
        "W1": cache["X"].T @ grad_z1,
        "b1": grad_z1.sum(axis=0),
        "W2": grad_W2,
        "b2": grad_b2,
    }
# end::forward-backward[]


# tag::sgd[]
def synthetic_data(seed=1, examples_per_class=24):
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.5, -1.5, 1, 0, -1], [-1.5, 1.5, -1, 1, 0],
                        [1.5, -1.5, 0, -1, 1], [1.5, 1.5, 1, 1, 1]],
                       dtype=np.float32)
    labels = np.repeat(np.arange(4), examples_per_class)
    points = np.concatenate([
        center + rng.normal(0, 0.55, size=(examples_per_class, 5))
        for center in centers
    ]).astype(np.float32)
    return points, labels


def train_sgd(inputs, labels, epochs=80, learning_rate=0.08, seed=0):
    params = initialize(seed)
    history = []
    for _ in range(epochs):
        loss, cache = forward(params, inputs, labels)
        history.append(float(loss))
        gradients = backward(params, cache, labels)
        for name, value in params.items():
            value -= learning_rate * gradients[name]
    return params, np.array(history)
# end::sgd[]
