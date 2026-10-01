import numpy as np

from scratch.fine_tuning import (
    lora_mse_loss_and_grads,
    init_lora,
    lora_output,
    lora_parameter_savings,
    masked_cross_entropy,
    merge_lora,
    next_token_training_arrays,
    pack_documents,
    packed_lm_arrays,
    render_chat,
)
from scratch.gradcheck import check_gradient


def rng(seed=33):
    return np.random.default_rng(seed)


def test_chat_template_keeps_role_markers_and_masks_assistant_targets():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    text = render_chat(messages, add_generation_prompt=True)
    assert text == (
        "<|system|> be brief <|end|> <|user|> 2+2? <|end|> "
        "<|assistant|> 4 <|end|> <|assistant|>"
    )
    tokens = np.array([10, 11, 20, 21, 22, 30])
    roles = np.array(["user", "user", "assistant", "assistant", "assistant", "user"])
    inputs, targets, mask = next_token_training_arrays(tokens, roles)
    np.testing.assert_array_equal(inputs, [10, 11, 20, 21, 22])
    np.testing.assert_array_equal(targets, [11, 20, 21, 22, 30])
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_masked_loss_and_gradient_ignore_prompt_tokens():
    r = rng()
    logits = r.normal(size=(6, 5)).astype(np.float64)
    targets = np.array([0, 4, 1, 3, 2, 1])
    mask = np.array([False, True, True, False, True, False])
    loss, grad = masked_cross_entropy(logits, targets, mask)

    def objective(z):
        return masked_cross_entropy(z.reshape(logits.shape), targets, mask)[0]

    assert loss > 0
    assert check_gradient(objective, logits, grad, tolerance=1e-8) < 1e-8
    np.testing.assert_array_equal(grad[~mask], np.zeros_like(grad[~mask]))
    shifted = logits.copy()
    shifted[~mask] += r.normal(size=shifted[~mask].shape)
    np.testing.assert_allclose(masked_cross_entropy(shifted, targets, mask)[0], loss)


def test_packing_keeps_documents_from_predicting_across_boundaries():
    documents = [[1, 2], [3, 4, 5], [6]]
    packed, doc_ids = pack_documents(documents, max_length=5, boundary_id=99)
    np.testing.assert_array_equal(
        packed,
        [[1, 2, 99, 0, 0], [3, 4, 5, 99, 0], [6, 99, 0, 0, 0]],
    )
    inputs, targets, mask = packed_lm_arrays(packed, doc_ids)
    np.testing.assert_array_equal(inputs[0], [1, 2, 99, 0])
    np.testing.assert_array_equal(targets[0], [2, 99, 0, 0])
    np.testing.assert_array_equal(mask[0], [True, False, False, False])
    assert not np.any(mask & (doc_ids[:, 1:] < 0))
    assert not np.any(mask & (doc_ids[:, :-1] != doc_ids[:, 1:]))


def test_lora_initializes_identically_and_merges_exactly():
    r = rng()
    x = r.normal(size=(4, 3)).astype(np.float32)
    w = r.normal(size=(3, 5)).astype(np.float32)
    a, b = init_lora(3, 5, rank=2, rng=r)
    np.testing.assert_allclose(lora_output(x, w, a, b, alpha=4.0), x @ w)

    b[1, 0] = 0.25
    b[2, 1] = -0.5
    merged = merge_lora(w, a, b, alpha=4.0)
    np.testing.assert_allclose(lora_output(x, w, a, b, alpha=4.0), x @ merged)


def test_lora_gradients_for_a_and_b_are_checked():
    r = rng()
    x = r.normal(size=(5, 4)).astype(np.float64)
    w = r.normal(size=(4, 3)).astype(np.float64)
    a = r.normal(size=(2, 3)).astype(np.float64)
    b = r.normal(size=(4, 2)).astype(np.float64)
    target = r.normal(size=(5, 3)).astype(np.float64)
    alpha = 3.0
    loss, grad_a, grad_b = lora_mse_loss_and_grads(x, w, a, b, alpha, target)
    assert loss > 0

    def objective_a(a_value):
        return lora_mse_loss_and_grads(x, w, a_value, b, alpha, target)[0]

    def objective_b(b_value):
        return lora_mse_loss_and_grads(x, w, a, b_value, alpha, target)[0]

    assert check_gradient(objective_a, a, grad_a, tolerance=1e-8) < 1e-8
    assert check_gradient(objective_b, b, grad_b, tolerance=1e-8) < 1e-8


def test_lora_parameter_savings_are_computed_by_code():
    base, trainable, ratio = lora_parameter_savings(4096, 4096, rank=8)
    assert base == 16_777_216
    assert trainable == 65_536
    assert ratio == 256
