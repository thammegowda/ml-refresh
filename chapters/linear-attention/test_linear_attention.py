import numpy as np

from linear_ops import (
    delta_step,
    feature_map,
    gated_delta_step,
    parallel_linear_attention,
    recurrent_linear_attention,
    selective_state_space,
)
from solutions import decode_one, decoding_state_size, last_token_from_prefix


def rng(seed=28):
    return np.random.default_rng(seed)


def test_parallel_and_recurrent_linear_attention_agree():
    r = rng()
    q = r.normal(size=(6, 4)).astype(np.float64)
    k = r.normal(size=(6, 4)).astype(np.float64)
    v = r.normal(size=(6, 3)).astype(np.float64)
    parallel = parallel_linear_attention(q, k, v)
    recurrent = recurrent_linear_attention(q, k, v)
    np.testing.assert_allclose(parallel, recurrent, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(last_token_from_prefix(q, k, v), parallel[-1])


def test_decoding_updates_only_constant_size_state():
    r = rng()
    q = r.normal(size=(5, 3)).astype(np.float64)
    k = r.normal(size=(5, 3)).astype(np.float64)
    v = r.normal(size=(5, 2)).astype(np.float64)
    state = np.zeros((3, 2))
    normalizer = np.zeros(3)
    outputs = []
    for qt, kt, vt in zip(q, k, v):
        out, state, normalizer = decode_one(qt, kt, vt, state, normalizer, feature_map)
        outputs.append(out)
    np.testing.assert_allclose(outputs, recurrent_linear_attention(q, k, v))
    assert decoding_state_size(3, 2) == 9


def test_delta_rule_reduces_repeated_key_prediction_error():
    key = np.array([1.0, 0.0])
    value = np.array([2.0, -1.0])
    state = np.zeros((2, 2))
    errors = []
    for _ in range(5):
        state, error = delta_step(state, key, value, beta=0.5)
        errors.append(np.linalg.norm(error))
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    np.testing.assert_allclose(key @ state, value * (1 - 0.5 ** 5))


def test_gated_delta_forgets_old_values_before_writing_new_error():
    key = np.array([1.0, 0.0])
    old = np.array([1.0, 0.0])
    new = np.array([0.0, 1.0])
    state = np.zeros((2, 2))
    state, _ = delta_step(state, key, old, beta=1.0)
    gated, error = gated_delta_step(state, key, new, beta=1.0, gate=0.25)
    np.testing.assert_allclose(error, [-0.25, 1.0])
    np.testing.assert_allclose(key @ gated, new)


def test_selective_state_space_matches_manual_recurrence():
    x = np.array([1.0, 2.0, -1.0])
    a = np.array([-0.5, -1.0])
    b = np.array([[1.0, 0.5], [0.5, 1.0], [1.5, -0.5]])
    c = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    delta = np.array([0.2, 0.1, 0.3])
    y = selective_state_space(x, a, b, c, delta)
    state = np.zeros(2)
    manual = []
    for xt, bt, ct, dt in zip(x, b, c, delta):
        state = np.exp(dt * a) * state + dt * bt * xt
        manual.append(ct @ state)
    np.testing.assert_allclose(y, manual)
