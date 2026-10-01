"""Back-of-the-envelope sizing for a decoder-only language model (Chapter 44)."""


# tag::budget[]
def transformer_parameters(layers, width, heads, kv_heads, vocabulary, tied=True):
    """Weights of a pre-norm decoder stack; norms and biases are ignored."""
    head_width = width // heads
    attention = 2 * width * width + 2 * width * kv_heads * head_width  # Q, O; K, V
    feed_forward = 8 * width * width            # a 4d MLP, or SwiGLU at width 8d/3
    embeddings = vocabulary * width * (1 if tied else 2)
    return layers * (attention + feed_forward) + embeddings


def training_flops(parameters, tokens):
    return 6 * parameters * tokens              # 2ND forward + 4ND backward


def compute_optimal_tokens(parameters, tokens_per_parameter=20):
    return tokens_per_parameter * parameters    # the Chinchilla rule of thumb


def training_memory_bytes(parameters):
    # bf16 weights and gradients (2 + 2), fp32 master copy (4), two Adam moments (4 + 4)
    return 16 * parameters


def kv_cache_bytes(layers, kv_heads, head_width, tokens, bytes_per_value=2):
    return 2 * layers * kv_heads * head_width * tokens * bytes_per_value
# end::budget[]


EXAMPLE = dict(layers=16, width=2048, heads=16, kv_heads=4, vocabulary=32_768)


# tag::example[]
def size_example(sustained_flops=400e12, context=4096):
    """Size the example model and its training run."""
    n = transformer_parameters(**EXAMPLE)
    d = compute_optimal_tokens(n)
    head_width = EXAMPLE["width"] // EXAMPLE["heads"]
    return {
        "parameters": n,
        "tokens": d,
        "flops": training_flops(n, d),
        "days": training_flops(n, d) / sustained_flops / 86_400,
        "training_gb": training_memory_bytes(n) / 1e9,
        "kv_mib": kv_cache_bytes(EXAMPLE["layers"], EXAMPLE["kv_heads"], head_width,
                                 context) / 2 ** 20,
    }
# end::example[]
