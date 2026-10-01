import numpy as np

from scratch.transformer import (
    init_block_params,
    transformer_block_backward,
    transformer_block_forward,
)


# tag::block-loss[]
def block_scalar_loss(x, params, n_heads, upstream):
    y, _ = transformer_block_forward(x, params, n_heads)
    return float(np.sum(y * upstream))
# end::block-loss[]


# tag::count[]
def block_parameter_count(d_model, hidden_dim):
    attention = 4 * d_model * d_model
    swiglu = 3 * d_model * hidden_dim
    norms = 2 * d_model
    return attention + swiglu + norms
# end::count[]


# tag::one-block[]
def one_block_output(seed=22):
    rng = np.random.default_rng(seed)
    params = init_block_params(8, 2, 32, rng)
    x = rng.normal(size=(2, 4, 8)).astype(np.float32)
    y, cache = transformer_block_forward(x, params, n_heads=2)
    dx, grads = transformer_block_backward(np.ones_like(y), cache)
    return y.shape, dx.shape, sorted(grads)
# end::one-block[]
