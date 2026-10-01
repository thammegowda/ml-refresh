"""Emulating low-precision floating point in NumPy (Appendix B)."""
import numpy as np

__chapter__ = "numpy"


# tag::bfloat16[]
def round_to_bfloat16(x):
    """Round float32 values to the nearest bfloat16 (ties to even), returned as float32.

    bfloat16 keeps float32's sign bit and 8 exponent bits but only the top 7 of its
    23 fraction bits, so rounding happens on the low 16 bits of the float32 pattern.
    """
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1  # the last kept bit decides ties
    rounded = ((bits + 0x7FFF + lsb) >> 16) << 16
    result = rounded.astype(np.uint32).view(np.float32)
    return np.where(np.isnan(x), np.float32(np.nan), result)
# end::bfloat16[]
