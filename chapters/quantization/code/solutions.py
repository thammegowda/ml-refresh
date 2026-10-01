"""Solutions for Chapter 40."""
import numpy as np

from scratch.quantization import mean_squared_error, quantize_absmax


# tag::channel-errors[]
def tensor_channel_group_errors(x):
    """Return MSEs for tensor, row-channel, and group quantization."""
    _, tensor, _ = quantize_absmax(x, bits=4)
    _, channel, _ = quantize_absmax(x, bits=4, axis=1)
    grouped = x.reshape(x.shape[0], 2, x.shape[1] // 2)
    _, group_dequant, _ = quantize_absmax(grouped, bits=4, axis=2)
    group_dequant = group_dequant.reshape(x.shape)
    return (mean_squared_error(x, tensor),
            mean_squared_error(x, channel),
            mean_squared_error(x, group_dequant))
# end::channel-errors[]
