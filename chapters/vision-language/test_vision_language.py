import numpy as np

from solutions import dynamic_counts_example, small_mrope_ids, toy_zero_shot_label
from vlm import (clip_zero_shot, dynamic_token_counts,
                 flamingo_gated_cross_attention, interleave_image_tokens,
                 linear_projector, merge_2x2_tokens, mlp_projector,
                 mrope_position_ids, perceiver_resampler)


def test_clip_zero_shot_prompt_similarity():
    images = np.array([[1.0, 0.0], [0.0, 1.0]])
    prompts = np.array([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]])
    probs = clip_zero_shot(images, prompts, logit_scale=10.0)
    assert np.argmax(probs, axis=1).tolist() == [0, 1]
    assert toy_zero_shot_label() == [0, 1]
    assert probs[0, 0] > 0.99 and probs[1, 1] > 0.99


def test_projectors_change_visual_width_to_llm_width():
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((2, 5, 4)).astype(np.float32)
    w = rng.standard_normal((4, 6)).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)
    projected = linear_projector(tokens, w, b)
    np.testing.assert_allclose(projected, tokens @ w + b)
    w1 = rng.standard_normal((4, 7)).astype(np.float32)
    b1 = rng.standard_normal(7).astype(np.float32)
    w2 = rng.standard_normal((7, 6)).astype(np.float32)
    b2 = rng.standard_normal(6).astype(np.float32)
    assert mlp_projector(tokens, w1, b1, w2, b2).shape == (2, 5, 6)


def test_resampler_returns_fixed_query_count():
    rng = np.random.default_rng(1)
    queries = rng.standard_normal((4, 8)).astype(np.float32)
    short = rng.standard_normal((2, 6, 8)).astype(np.float32)
    long = rng.standard_normal((2, 20, 8)).astype(np.float32)
    assert perceiver_resampler(short, queries).shape == (2, 4, 8)
    assert perceiver_resampler(long, queries).shape == (2, 4, 8)


def test_flamingo_gate_is_identity_at_initialization_and_interleaves_tokens():
    rng = np.random.default_rng(2)
    text = rng.standard_normal((1, 3, 4)).astype(np.float32)
    image = rng.standard_normal((1, 2, 4)).astype(np.float32)
    np.testing.assert_allclose(flamingo_gated_cross_attention(text, image, gate=0.0), text)
    changed = flamingo_gated_cross_attention(text, image, gate=1.0)
    assert not np.allclose(changed, text)
    joined = interleave_image_tokens(text, image, image_at=1)
    assert joined.shape == (1, 5, 4)
    np.testing.assert_allclose(joined[:, :1], text[:, :1])
    np.testing.assert_allclose(joined[:, 1:3], image)
    np.testing.assert_allclose(joined[:, 3:], text[:, 1:])


def test_mrope_ids_and_dynamic_resolution_token_merging():
    ids = mrope_position_ids(frames=2, grid_h=2, grid_w=3)
    assert ids.shape == (12, 3)
    np.testing.assert_array_equal(ids[0], [0, 0, 0])
    np.testing.assert_array_equal(ids[-1], [1, 1, 2])
    np.testing.assert_array_equal(small_mrope_ids(), ids)
    assert dynamic_counts_example() == ((24, 48), 1152, 288)
    assert dynamic_token_counts(336, 672, patch_size=14, merge=2) == ((24, 48), 1152, 288)
    tokens = np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 16, 1)
    merged = merge_2x2_tokens(tokens, (4, 4))
    np.testing.assert_allclose(merged.reshape(4), [2.5, 4.5, 10.5, 12.5])
