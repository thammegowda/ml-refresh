import numpy as np

from scratch.gradcheck import check_gradient
from scratch.losses import (bce_with_logits, binary_focal_loss, distillation_loss,
                            huber_loss, jensen_shannon, kl_forward_logits,
                            kl_reverse_logits, mae_loss, mse_loss)
from scratch.softmax_cross_entropy import softmax
from solutions import scaled_and_unscaled_distillation, stable_bce_example


def rng(seed=13):
    return np.random.default_rng(seed)


def test_regression_losses_and_gradients_are_checked():
    prediction = np.array([-2.0, -0.3, 0.8, 3.0])
    target = np.array([0.5, 0.2, -0.1, 1.0])
    for loss_fn in (mse_loss, mae_loss, huber_loss):
        loss, grad = loss_fn(prediction, target)
        assert loss > 0
        check_gradient(lambda value, fn=loss_fn: fn(value, target)[0], prediction, grad)

    residual = np.array([10.0])
    _, mse_grad = mse_loss(residual, np.array([0.0]))
    _, mae_grad = mae_loss(residual, np.array([0.0]))
    _, huber_grad = huber_loss(residual, np.array([0.0]), delta=1.0)
    grads = [mse_grad[0], mae_grad[0], huber_grad[0]]
    np.testing.assert_allclose(grads, [20.0, 1.0, 1.0])


def test_bce_with_logits_is_stable_and_gradient_checked():
    assert stable_bce_example() == 0.0
    logits = rng().normal(size=(2, 3))
    targets = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss, grad = bce_with_logits(logits, targets)
    assert loss > 0
    check_gradient(lambda value: bce_with_logits(value, targets)[0], logits, grad)


def test_focal_loss_downweights_easy_examples_and_has_gradients():
    easy = binary_focal_loss(np.array([4.0]), np.array([1.0]))[0]
    hard = binary_focal_loss(np.array([-4.0]), np.array([1.0]))[0]
    assert hard / easy > 100_000
    logits = rng().normal(size=5)
    targets = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    loss, grad = binary_focal_loss(logits, targets)
    assert loss > 0
    check_gradient(lambda value: binary_focal_loss(value, targets)[0], logits, grad)


def test_forward_reverse_kl_and_js_divergence():
    r = rng()
    target = r.dirichlet(np.ones(4), size=3)
    logits = r.normal(size=(3, 4))
    for loss_fn in (kl_forward_logits,):
        loss, grad = loss_fn(target, logits)
        assert loss >= 0
        check_gradient(lambda value, fn=loss_fn: fn(target, value)[0], logits, grad)
    reverse_loss, reverse_grad = kl_reverse_logits(logits, target)
    assert reverse_loss >= 0
    check_gradient(lambda value: kl_reverse_logits(value, target)[0],
                   logits, reverse_grad)

    p, q = np.array([0.7, 0.3]), np.array([0.2, 0.8])
    np.testing.assert_allclose(jensen_shannon(p, q), jensen_shannon(q, p))
    assert 0 <= jensen_shannon(p, q) <= np.log(2)


def test_distillation_temperature_squared_gradient_scaling():
    student = rng().normal(size=(2, 5))
    teacher = rng(14).normal(size=(2, 5))
    temperature = 3.0
    loss, grad = distillation_loss(student, teacher, temperature)
    assert loss > 0
    check_gradient(lambda value: distillation_loss(value, teacher, temperature)[0],
                   student, grad)
    scaled, unscaled = scaled_and_unscaled_distillation(student, teacher, temperature)
    np.testing.assert_allclose(scaled, unscaled * temperature ** 2)
    expected = temperature * (softmax(student, temperature)
                              - softmax(teacher, temperature)) / student.shape[0]
    np.testing.assert_allclose(grad, expected)
