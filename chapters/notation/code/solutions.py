"""Worked solutions to the Appendix A exercises."""
import numpy as np


# tag::pytorch[]
def from_pytorch_linear(weight, bias):
    """nn.Linear stores weight as (d_out, d_in) and computes x @ weight.T + bias."""
    return weight.T, bias
# end::pytorch[]


# tag::elementwise[]
def elementwise_vjp(grad_y, x, derivative):
    return grad_y * derivative(x)              # O(n): the Jacobian is never built


def elementwise_vjp_dense(grad_y, x, derivative):
    jacobian = np.diag(derivative(x))          # (n, n), zero off the diagonal
    return jacobian.T @ grad_y                 # O(n^2) memory for the same answer
# end::elementwise[]
