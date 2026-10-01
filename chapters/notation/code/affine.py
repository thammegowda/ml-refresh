"""The affine layer, used as the worked example of the book's gradient notation."""


# tag::affine[]
def affine_forward(X, W, b):
    """X (N, d_in), W (d_in, d_out), b (d_out,) -> Y (N, d_out)."""
    return X @ W + b


def affine_backward(grad_Y, X, W):
    """Given grad_Y = dL/dY (N, d_out), return dL/dX, dL/dW, dL/db."""
    grad_X = grad_Y @ W.T          # (N, d_out) @ (d_out, d_in) -> (N, d_in)
    grad_W = X.T @ grad_Y          # (d_in, N) @ (N, d_out)     -> (d_in, d_out)
    grad_b = grad_Y.sum(axis=0)    # b was broadcast over the N rows
    return grad_X, grad_W, grad_b
# end::affine[]
