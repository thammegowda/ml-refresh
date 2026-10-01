"""Training-at-scale calculators and simulations (Chapter 41)."""
import numpy as np

from scratch.quantization import fp8_e4m3

__chapter__ = "distributed-training"


# tag::memory[]
def adam_memory_bytes(parameters, weight=2, grad=2, master=4, moments=8):
    """Mixed-precision Adam bytes: bf16 weights/grads plus fp32 state."""
    return parameters * (weight + grad + master + moments)


def activation_memory_bytes(tokens, hidden, layers, bytes_per_value=2,
                            tensors_per_layer=1):
    return tokens * hidden * layers * bytes_per_value * tensors_per_layer


def checkpointed_activation_bytes(tokens, hidden, layers, segments,
                                  bytes_per_value=2):
    """Store segment boundaries and recompute interiors during backward."""
    segment_length = int(np.ceil(layers / segments))
    saved_layers = segments + segment_length
    return tokens * hidden * saved_layers * bytes_per_value
# end::memory[]


# tag::collectives-zero[]
def ring_all_reduce_bytes(size_bytes, ranks):
    """Bytes sent per rank by ring all-reduce."""
    return 2 * (ranks - 1) / ranks * size_bytes


def zero_memory_bytes(parameters, ranks):
    """Per-rank model-state bytes for data parallel Adam and ZeRO stages."""
    p = parameters
    return {
        "dp": 16 * p,
        "zero1": 4 * p + 12 * p / ranks,
        "zero2": 2 * p + 14 * p / ranks,
        "zero3": 16 * p / ranks,
    }
# end::collectives-zero[]


# tag::tensor-parallel[]
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) *
                                    (x + 0.044715 * x ** 3)))


def mlp(x, w1, b1, w2, b2):
    return gelu(x @ w1 + b1) @ w2 + b2


def tensor_parallel_mlp(x, w1, b1, w2, b2, ranks):
    """Column-parallel first layer, row-parallel second layer."""
    w1_parts = np.array_split(w1, ranks, axis=1)
    b1_parts = np.array_split(b1, ranks)
    w2_parts = np.array_split(w2, ranks, axis=0)
    partials = []
    for w1_i, b1_i, w2_i in zip(w1_parts, b1_parts, w2_parts):
        partials.append(gelu(x @ w1_i + b1_i) @ w2_i)
    return np.sum(partials, axis=0) + b2
# end::tensor-parallel[]


# tag::pipeline-expert[]
def pipeline_bubble_fraction(stages, microbatches):
    return (stages - 1) / (microbatches + stages - 1)


def expert_all_to_all_counts(assignments, expert_to_rank, ranks):
    """Count tokens sent from each source rank to each expert-owning rank."""
    assignments = np.asarray(assignments)
    counts = np.zeros((ranks, ranks), dtype=np.int64)
    for source in range(ranks):
        for expert in assignments[source]:
            counts[source, expert_to_rank[int(expert)]] += 1
    return counts
# end::pipeline-expert[]


# tag::fp8[]
def fine_grained_fp8(x, block_size=16, max_value=448.0):
    """Block-scale values into E4M3 range, quantize, then dequantize."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % block_size:
        raise ValueError("last dimension must be divisible by block_size")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scale = np.max(np.abs(blocks), axis=-1, keepdims=True) / max_value
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    dequant = fp8_e4m3(blocks / scale) * scale
    return dequant.reshape(x.shape), scale
# end::fp8[]
