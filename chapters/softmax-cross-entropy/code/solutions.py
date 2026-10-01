"""Worked solutions to the Chapter 12 exercises."""
import numpy as np

from scratch.softmax_cross_entropy import softmax, softmax_cross_entropy


# tag::temperature[]
def probabilities_at_temperatures(logits, temperatures):
    return [softmax(logits, temperature=T) for T in temperatures]
# end::temperature[]


# tag::fused-check[]
def loss_only(logits, labels, smoothing=0.0, z_loss=0.0):
    loss, _ = softmax_cross_entropy(logits, labels,
                                    label_smoothing=smoothing,
                                    z_loss=z_loss)
    return loss
# end::fused-check[]
