"""Worked solution helpers for Chapter 20."""
import numpy as np

from scratch.attention import causal_mask
from scratch.multi_head_attention import (
    multi_head_attention_forward,
    parameter_count,
    self_attention_flops,
)


# tag::budget[]
def tiny_budget(model_width=8, length=5, batch=2):
    """Parameters and dominant self-attention FLOPs for a tiny setting."""
    flops = self_attention_flops(batch, length, model_width)
    return parameter_count(model_width), flops
# end::budget[]


# tag::causal-heads[]
def first_head_causal_weights():
    """Return attention weights from the first head of a tiny masked MHA."""
    X = np.arange(12, dtype=np.float64).reshape(1, 3, 4) / 10
    eye = np.eye(4)
    params = (eye, eye, eye, eye)
    _out, cache = multi_head_attention_forward(
        X, X, X, params, 2, causal_mask(3), True
    )
    return cache[5][3][0, 0]
# end::causal-heads[]
