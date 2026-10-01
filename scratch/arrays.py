"""Shape utilities for backpropagation through broadcasting (Appendix B)."""
import numpy as np

__chapter__ = "numpy"


# tag::unbroadcast[]
def unbroadcast(gradient, shape):
    """Sum a gradient over the axes that broadcasting stretched, returning `shape`."""
    extra = gradient.ndim - len(shape)
    gradient = gradient.sum(axis=tuple(range(extra))) if extra else gradient
    stretched = tuple(axis for axis, size in enumerate(shape)
                      if size == 1 and gradient.shape[axis] != 1)
    return gradient.sum(axis=stretched, keepdims=True) if stretched else gradient
# end::unbroadcast[]
