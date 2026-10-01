"""Solution snippets for multi-head latent attention."""
from scratch.latent_attention import cache_size_table


# tag::cache-table[]
def cache_table_mib():
    rows = cache_size_table(
        layers=32,
        tokens=4096,
        bytes_per_value=2,
        heads=32,
        kv_heads=8,
        head_dim=128,
        latent_dim=512,
    )
    return {name: bytes_used // (1024 ** 2) for name, bytes_used in rows.items()}
# end::cache-table[]
