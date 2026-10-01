import numpy as np

from scratch.precision import round_to_bfloat16


# tag::precision[]
def fp16_overflows_but_bfloat16_keeps_range():
    large = np.array([1e5, 1e30], dtype=np.float32)
    fp16 = large.astype(np.float16).astype(np.float32)
    bf16 = round_to_bfloat16(large)
    return fp16, bf16


def loss_scaled_gradient(gradient, scale):
    scaled = (gradient * scale).astype(np.float16)
    unscaled = scaled.astype(np.float32) / scale
    return unscaled
# end::precision[]
