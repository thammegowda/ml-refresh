import numpy as np

from vit import patchify, sequence_length


# tag::tokens[]
def token_counts():
    grid = 224 // 16
    return grid, grid * grid, sequence_length(224, 224, 16, True)
# end::tokens[]


# tag::patch-order[]
def patch_order_example():
    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    return patchify(image, 2)[0].astype(int).tolist()
# end::patch-order[]
