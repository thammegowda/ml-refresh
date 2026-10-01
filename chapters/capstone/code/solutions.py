"""Worked solutions to the Chapter 44 exercises."""
import math

from budget import EXAMPLE, kv_cache_bytes, transformer_parameters


# tag::allocate[]
def allocate(flops, tokens_per_parameter=20):
    """Split a compute budget C = 6ND with D = 20N: N = sqrt(C / 120)."""
    parameters = math.sqrt(flops / (6 * tokens_per_parameter))
    return parameters, tokens_per_parameter * parameters
# end::allocate[]


# tag::serving[]
def concurrent_sequences(memory_bytes, context, kv_heads):
    """Sequences whose KV cache fits beside bf16 weights in a memory budget."""
    head_width = EXAMPLE["width"] // EXAMPLE["heads"]
    weights = 2 * transformer_parameters(**{**EXAMPLE, "kv_heads": kv_heads})
    per_sequence = kv_cache_bytes(EXAMPLE["layers"], kv_heads, head_width, context)
    return int((memory_bytes - weights) // per_sequence)
# end::serving[]
