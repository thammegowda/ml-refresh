import numpy as np

from mlp import (backward, forward, initialize, softmax_cross_entropy,
                 synthetic_data, train_sgd)
from solutions import memorization_gap, parameter_count
from scratch.gradcheck import check_gradient


def packed_loss(theta, template, inputs, labels):
    params = {}
    offset = 0
    for name, shape in template:
        size = int(np.prod(shape))
        params[name] = theta[offset:offset + size].reshape(shape)
        offset += size
    loss, _ = forward(params, inputs, labels)
    return float(loss)


def pack(params):
    template = [(name, value.shape) for name, value in params.items()]
    theta = np.concatenate([params[name].ravel() for name, _ in template])
    return theta, template


def test_forward_shapes_and_stable_cross_entropy():
    params = initialize(seed=2)
    assert {name: value.shape for name, value in params.items()} == {
        "W1": (5, 10), "b1": (10,), "W2": (10, 4), "b2": (4,),
    }
    assert sum(value.size for value in params.values()) == 104
    logits = np.array([[1000.0, -1000.0, 1.0, 2.0]])
    loss, probabilities = softmax_cross_entropy(logits, np.array([1]))
    assert np.isfinite(loss) and loss > 1000
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_backpropagation_matches_finite_differences():
    rng = np.random.default_rng(3)
    params = initialize(seed=4, dtype=np.float64)
    inputs = rng.normal(size=(6, 5)) * 0.2 + 0.1
    labels = np.array([0, 1, 2, 3, 1, 0])
    loss, cache = forward(params, inputs, labels)
    assert np.isfinite(loss)
    gradients = backward(params, cache, labels)
    theta, template = pack(params)
    analytic = np.concatenate([gradients[name].ravel() for name, _ in template])
    check_gradient(lambda t: packed_loss(t, template, inputs, labels), theta,
                   analytic, tolerance=2e-7)


def test_softmax_cross_entropy_gradient_is_p_minus_y_over_batch():
    logits = np.array([[0.3, -0.1, 0.7], [-0.2, 0.6, 0.1]], dtype=np.float64)
    labels = np.array([2, 0])
    _, probabilities = softmax_cross_entropy(logits, labels)
    target = np.eye(3)[labels]
    analytic = (probabilities - target) / len(labels)
    check_gradient(lambda z: softmax_cross_entropy(z.reshape(logits.shape), labels)[0],
                   logits.ravel(), analytic.ravel())


def test_plain_sgd_learns_synthetic_clusters():
    inputs, labels = synthetic_data(seed=5)
    params, history = train_sgd(inputs, labels, epochs=120, learning_rate=0.08, seed=6)
    final_loss, cache = forward(params, inputs, labels)
    accuracy = np.mean(cache["P"].argmax(axis=1) == labels)
    assert history[0] > 1.0
    assert final_loss < 0.35 * history[0]
    assert accuracy > 0.9


def test_solution_helpers_cover_count_and_overfitting_experiment():
    assert parameter_count() == 104
    train_accuracy, clean_accuracy = memorization_gap()
    assert train_accuracy > clean_accuracy
