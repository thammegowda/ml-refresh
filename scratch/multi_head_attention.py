"""Multi-head attention built from scaled dot-product attention (Chapter 20)."""
import numpy as np

from scratch.attention import attention_backward, attention_forward

__chapter__ = "multi-head-attention"


# tag::split[]
def split_heads(x, num_heads):
    """(B, T, d_model) -> (B, H, T, d_head)."""
    batch, length, width = x.shape
    if width % num_heads:
        raise ValueError("model width must be divisible by the number of heads")
    head_width = width // num_heads
    return x.reshape(batch, length, num_heads, head_width).transpose(0, 2, 1, 3)


def combine_heads(x):
    """(B, H, T, d_head) -> (B, T, d_model)."""
    batch, heads, length, head_width = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_width)
# end::split[]


def _project_grad(x, grad_y):
    x2 = x.reshape(-1, x.shape[-1])
    g2 = grad_y.reshape(-1, grad_y.shape[-1])
    return x2.T @ g2


def _head_mask(mask):
    if mask is None or mask.ndim == 2:
        return mask
    return mask[:, None, :, :]


# tag::forward[]
def multi_head_attention_forward(Xq, Xk, Xv, params, num_heads, mask=None,
                                 return_cache=False):
    """Project, split into heads, attend, concatenate, and project out."""
    W_Q, W_K, W_V, W_O = params
    Q_linear, K_linear, V_linear = Xq @ W_Q, Xk @ W_K, Xv @ W_V
    Q = split_heads(Q_linear, num_heads)
    K = split_heads(K_linear, num_heads)
    V = split_heads(V_linear, num_heads)
    head_output, attn_cache = attention_forward(Q, K, V, _head_mask(mask), True)
    joined = combine_heads(head_output)
    output = joined @ W_O
    if not return_cache:
        return output
    cache = (Xq, Xk, Xv, params, num_heads, attn_cache, head_output, joined)
    return output, cache
# end::forward[]


# tag::backward[]
def multi_head_attention_backward(grad_output, cache):
    """Backward pass for multi-head attention."""
    Xq, Xk, Xv, params, num_heads, attn_cache, _head_output, joined = cache
    W_Q, W_K, W_V, W_O = params
    grad_W_O = _project_grad(joined, grad_output)
    grad_joined = grad_output @ W_O.T
    grad_heads = split_heads(grad_joined, num_heads)
    grad_Q, grad_K, grad_V = attention_backward(grad_heads, attn_cache)
    grad_Q_linear = combine_heads(grad_Q)
    grad_K_linear = combine_heads(grad_K)
    grad_V_linear = combine_heads(grad_V)
    grad_Xq = grad_Q_linear @ W_Q.T
    grad_Xk = grad_K_linear @ W_K.T
    grad_Xv = grad_V_linear @ W_V.T
    grad_W_Q = _project_grad(Xq, grad_Q_linear)
    grad_W_K = _project_grad(Xk, grad_K_linear)
    grad_W_V = _project_grad(Xv, grad_V_linear)
    return (grad_Xq, grad_Xk, grad_Xv), (grad_W_Q, grad_W_K, grad_W_V, grad_W_O)
# end::backward[]


# tag::counts[]
def parameter_count(model_width):
    """Four dense d_model by d_model matrices."""
    return 4 * model_width * model_width


def self_attention_flops(batch, length, model_width):
    """Dominant multiply-add count: projections plus score/value products."""
    projections = 4 * batch * length * model_width * model_width
    attention = 2 * batch * length * length * model_width
    return projections + attention
# end::counts[]
