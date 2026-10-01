"""Solution snippets for the KV-cache chapter."""
from scratch.kv_cache import kv_cache_bytes, sliding_window_mask


# tag::memory-example[]
def memory_example_mib():
    bytes_used = kv_cache_bytes(
        layers=32,
        num_kv_heads=8,
        head_dim=128,
        tokens=4096,
        bytes_per_value=2,
    )
    return bytes_used // (1024 ** 2)
# end::memory-example[]


# tag::window-example[]
def last_row_with_sink():
    return sliding_window_mask(tokens=6, window=3, sinks=1)[5].astype(int)
# end::window-example[]
