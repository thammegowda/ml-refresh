import numpy as np

from scratch.gradcheck import check_gradient
from scratch.softmax_cross_entropy import (
    log_softmax, logsumexp, sigmoid_as_two_class_softmax, smooth_one_hot,
    softmax, softmax_cross_entropy, softmax_jacobian,
)
from solutions import loss_only, probabilities_at_temperatures


def rng(seed=12):
    return np.random.default_rng(seed)


def test_softmax_is_shift_invariant_and_stable():
    logits = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1002.0, -999.0]])
    shifted = logits + np.array([[123.0], [-456.0]])
    np.testing.assert_allclose(softmax(logits), softmax(shifted))
    np.testing.assert_allclose(np.exp(log_softmax(logits)).sum(axis=-1), 1.0)
    assert np.all(np.isfinite(log_softmax(logits)))
    expected = np.max(logits, axis=-1) + np.log(
        np.sum(np.exp(logits - np.max(logits, axis=-1, keepdims=True)), axis=-1)
    )
    np.testing.assert_allclose(logsumexp(logits), expected)


def test_temperature_controls_peakiness():
    logits = np.array([2.0, 1.0, -1.0])
    cold, base, hot = probabilities_at_temperatures(logits, [0.5, 1.0, 2.0])
    assert cold[0] > base[0] > hot[0]
    uniform = np.full_like(logits, 1 / len(logits))
    assert np.linalg.norm(hot - uniform) < np.linalg.norm(base - uniform)


def test_softmax_jacobian_matches_finite_differences():
    z = rng().normal(size=5)
    upstream = rng(13).normal(size=5)
    p = softmax(z)
    analytic = softmax_jacobian(p).T @ upstream
    check_gradient(lambda value: float(np.sum(softmax(value) * upstream)), z, analytic)
    np.testing.assert_allclose(softmax_jacobian(p).sum(axis=1), 0.0, atol=1e-15)


def test_cross_entropy_gradient_is_p_minus_y():
    logits = rng().normal(size=(4, 6))
    labels = np.array([0, 2, 4, 5])
    loss, grad = softmax_cross_entropy(logits, labels)
    assert loss > 0
    expected = softmax(logits) - smooth_one_hot(labels, 6)
    np.testing.assert_allclose(grad * len(labels), expected)
    check_gradient(lambda value: loss_only(value, labels), logits, grad)


def test_label_smoothing_temperature_and_z_loss_are_gradient_checked():
    logits = rng().normal(size=(3, 5))
    labels = np.array([1, 3, 0])
    eps = 0.1
    targets = smooth_one_hot(labels, 5, eps)
    np.testing.assert_allclose(targets.sum(axis=1), 1.0)
    np.testing.assert_allclose(targets[0], [0.02, 0.92, 0.02, 0.02, 0.02])

    temp_loss, temp_grad = softmax_cross_entropy(logits, labels, temperature=1.7,
                                                 label_smoothing=eps)
    assert temp_loss > 0
    check_gradient(
        lambda value: softmax_cross_entropy(value, labels, temperature=1.7,
                                            label_smoothing=eps)[0],
        logits,
        temp_grad,
    )

    loss, grad = softmax_cross_entropy(logits, labels,
                                       label_smoothing=eps, z_loss=0.01)
    assert loss > 0
    check_gradient(
        lambda value: loss_only(value, labels, smoothing=eps, z_loss=0.01),
        logits,
        softmax_cross_entropy(logits, labels,
                              label_smoothing=eps,
                              z_loss=0.01)[1],
    )
    assert grad.shape == logits.shape


def test_sigmoid_is_two_class_softmax():
    logit = np.array([-3.0, 0.0, 2.0])
    expected = 1 / (1 + np.exp(-logit))
    np.testing.assert_allclose(sigmoid_as_two_class_softmax(logit), expected)
    logits = np.stack([np.zeros_like(logit), logit], axis=-1)
    np.testing.assert_allclose(softmax(logits)[:, 1], expected)
