import numpy as np

from scratch.contrastive_learning import siglip_loss_and_grad, softmax_rows


# tag::temperature[]
def positive_probability(gap, temperature):
    logits = np.array([[gap, 0.0]]) / temperature
    return float(softmax_rows(logits)[0, 0])
# end::temperature[]


# tag::siglip-bias[]
def one_positive_three_negative_bias():
    similarity = np.zeros((3, 3))
    _, _, grad_bias = siglip_loss_and_grad(similarity, bias=0.0)
    return grad_bias
# end::siglip-bias[]
