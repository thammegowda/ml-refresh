import math

from budget import (EXAMPLE, compute_optimal_tokens, kv_cache_bytes, size_example,
                    training_flops, training_memory_bytes, transformer_parameters)
from solutions import allocate, concurrent_sequences


def test_parameter_count_matches_an_explicit_sum_of_weight_matrices():
    L, d, H, Hkv, V = 16, 2048, 16, 4, 32_768
    dh = d // H
    shapes = [(d, H * dh), (d, Hkv * dh), (d, Hkv * dh), (H * dh, d),        # attention
              (d, 8 * d // 3), (d, 8 * d // 3), (8 * d // 3, d)]               # SwiGLU
    per_layer = sum(a * b for a, b in shapes)
    swiglu_exact = L * per_layer + V * d
    assert abs(transformer_parameters(L, d, H, Hkv, V) - swiglu_exact) / swiglu_exact < 1e-3
    assert transformer_parameters(1, 8, 2, 2, 0) == 12 * 8 * 8   # multi-head, 4x MLP: 12 d^2
    assert transformer_parameters(1, 8, 2, 2, 10, tied=False) == 12 * 64 + 160


def test_example_budget_numbers_quoted_in_the_text():
    example = size_example()
    assert example["parameters"] == 771_751_936
    assert example["tokens"] == 15_435_038_720
    assert math.isclose(example["flops"], 7.147e19, rel_tol=1e-3)
    assert round(example["days"], 2) == 2.07
    assert round(example["training_gb"], 1) == 12.3
    assert example["kv_mib"] == 128
    assert kv_cache_bytes(16, 16, 128, 4096) / 2 ** 20 == 512
    assert training_flops(1, 1) == 6 and compute_optimal_tokens(5) == 100
    assert training_memory_bytes(10) == 160


def test_compute_allocation_exercise():
    n, d = allocate(1e21)
    assert math.isclose(training_flops(n, d), 1e21)
    assert round(n / 1e9, 2) == 2.89 and round(d / 1e9, 1) == 57.7


def test_serving_exercise():
    gqa = concurrent_sequences(80e9, 8192, kv_heads=4)
    mha = concurrent_sequences(80e9, 8192, kv_heads=16)
    assert (gqa, mha) == (292, 72)
