"""Worked solutions to the Chapter 13 exercises."""
import numpy as np

from scratch.losses import bce_with_logits, distillation_loss, huber_loss


# tag::outliers[]
def outlier_gradients(residual, delta=1.0):
    prediction = np.array([residual], dtype=np.float64)
    target = np.array([0.0])
    _, mse_grad = (residual ** 2, np.array([2 * residual]))
    _, huber_grad = huber_loss(prediction, target, delta)
    return mse_grad[0], huber_grad[0]
# end::outliers[]


# tag::bce-stable[]
def stable_bce_example():
    logits = np.array([1000.0, -1000.0])
    labels = np.array([1.0, 0.0])
    return bce_with_logits(logits, labels)[0]
# end::bce-stable[]


# tag::distill-scale[]
def scaled_and_unscaled_distillation(student, teacher, temperature):
    _, scaled = distillation_loss(student, teacher, temperature, scale=True)
    _, unscaled = distillation_loss(student, teacher, temperature, scale=False)
    return scaled, unscaled
# end::distill-scale[]
