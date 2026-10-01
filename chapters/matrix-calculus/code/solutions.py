"""Worked solutions to Appendix C exercises."""
from vjps import attention_vjp, matmul_vjp, softmax_forward, softmax_vjp


# tag::broadcast[]
def broadcast_add_gradient(grad, x, bias):
    del x
    return grad.sum(axis=0, keepdims=True).reshape(bias.shape)
# end::broadcast[]


# tag::matmul[]
def matmul_shapes(a, b, grad):
    grad_a, grad_b = matmul_vjp(grad, a, b)
    return grad_a.shape, grad_b.shape
# end::matmul[]


# tag::softmax[]
def softmax_jacobian_times_vector(logits, vector):
    y = softmax_forward(logits)
    return softmax_vjp(vector, y)
# end::softmax[]


# tag::attention[]
def attention_query_gradient(q, k, v, grad):
    grad_q, _, _ = attention_vjp(grad, q, k, v)
    return grad_q
# end::attention[]
