import numpy as np

from scratch.distributed_training import (activation_memory_bytes,
                                          adam_memory_bytes,
                                          checkpointed_activation_bytes,
                                          expert_all_to_all_counts,
                                          fine_grained_fp8, mlp,
                                          pipeline_bubble_fraction,
                                          ring_all_reduce_bytes,
                                          tensor_parallel_mlp,
                                          zero_memory_bytes)
from scratch.quantization import mean_squared_error


def rng(seed=41):
    return np.random.default_rng(seed)


def test_memory_accounting_and_checkpointing():
    assert adam_memory_bytes(1_000_000) == 16_000_000
    full = activation_memory_bytes(tokens=8, hidden=16, layers=12)
    checked = checkpointed_activation_bytes(tokens=8, hidden=16, layers=12,
                                            segments=3)
    assert full == 3072
    assert checked == 1792
    assert checked < full


def test_ring_all_reduce_and_zero_formulas():
    np.testing.assert_allclose(ring_all_reduce_bytes(1024, 4), 1536)
    mem = zero_memory_bytes(1_000, ranks=4)
    assert mem["dp"] == 16_000
    assert mem["zero1"] == 7_000
    assert mem["zero2"] == 5_500
    assert mem["zero3"] == 4_000


def test_tensor_parallel_mlp_matches_unsharded_layer():
    r = rng()
    x = r.normal(size=(5, 4))
    w1 = r.normal(size=(4, 8))
    b1 = r.normal(size=8)
    w2 = r.normal(size=(8, 3))
    b2 = r.normal(size=3)
    np.testing.assert_allclose(tensor_parallel_mlp(x, w1, b1, w2, b2, 2),
                               mlp(x, w1, b1, w2, b2), atol=1e-12)


def test_pipeline_bubble_and_expert_all_to_all_counts():
    np.testing.assert_allclose(pipeline_bubble_fraction(4, 12), 3 / 15)
    assignments = np.array([[0, 1, 3], [2, 3, 2]])
    counts = expert_all_to_all_counts(assignments, {0: 0, 1: 0, 2: 1, 3: 1}, 2)
    np.testing.assert_array_equal(counts, [[2, 1], [0, 3]])


def test_fine_grained_fp8_scaling_beats_one_global_block():
    x = np.array([[0.07, -0.14, 0.21, -0.28, 20.0, -20.0, 10.0, -10.0]],
                 dtype=np.float32)
    local, scales = fine_grained_fp8(x, block_size=4)
    global_block, _ = fine_grained_fp8(x, block_size=8)
    assert mean_squared_error(x, local) < mean_squared_error(x, global_block)
    assert scales.shape == (1, 2, 1)
