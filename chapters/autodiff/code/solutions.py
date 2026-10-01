"""Tested snippets for Chapter 10 solutions."""
import numpy as np

from scratch.autodiff import Tensor


# tag::tiny-network[]
def tiny_network_loss_and_grads(X, W, b):
    x = Tensor(X)
    w = Tensor(W)
    bias = Tensor(b)
    loss = ((x @ w + bias).relu().exp().log()).sum()
    loss.backward()
    return float(loss.data), x.grad, w.grad, bias.grad
# end::tiny-network[]
