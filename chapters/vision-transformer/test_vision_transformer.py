import numpy as np

from solutions import patch_order_example, token_counts
from vit import (add_2d_position, conv2d_strided, linear_patch_embedding,
                 make_tiny_vit_params, multihead_self_attention, patchify,
                 pool_sequence, prepend_class_token, sequence_length,
                 tiny_vit_forward)


def test_patchify_order_and_sequence_length_arithmetic():
    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    expected = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    assert patchify(image, 2)[0].astype(int).tolist() == expected
    assert patch_order_example() == expected
    assert token_counts() == (14, 196, 197)
    assert sequence_length(224, 224, 16) == 196
    assert sequence_length(448, 448, 16) == 784


def test_linear_patch_embedding_is_a_strided_convolution():
    rng = np.random.default_rng(0)
    images = rng.standard_normal((2, 6, 4, 3)).astype(np.float32)
    patch_size, out_dim = 2, 5
    weight = rng.standard_normal((patch_size * patch_size * 3, out_dim)).astype(np.float32)
    bias = rng.standard_normal(out_dim).astype(np.float32)
    patches = patchify(images, patch_size)
    embedded = linear_patch_embedding(patches, weight, bias)
    embedded = embedded.reshape(2, 3, 2, out_dim)
    kernel = weight.reshape(patch_size, patch_size, 3, out_dim)
    convolved = conv2d_strided(images, kernel, bias, stride=patch_size)
    np.testing.assert_allclose(embedded, convolved, rtol=1e-6, atol=1e-6)


def test_2d_positions_class_token_and_pooling():
    tokens = np.zeros((2, 6, 4), dtype=np.float32)
    row = np.arange(12, dtype=np.float32).reshape(3, 4)
    col = 100 + np.arange(8, dtype=np.float32).reshape(2, 4)
    positioned = add_2d_position(tokens, (3, 2), row, col)
    np.testing.assert_allclose(positioned[0, 3], row[1] + col[1])
    cls = np.full((1, 4), -1.0, dtype=np.float32)
    sequence = prepend_class_token(positioned, cls)
    assert sequence.shape == (2, 7, 4)
    np.testing.assert_allclose(pool_sequence(sequence, True), -1.0)
    np.testing.assert_allclose(pool_sequence(positioned, False), positioned.mean(axis=1))


def test_attention_is_self_contained_and_normalized():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    weights = [rng.standard_normal((8, 8)).astype(np.float32) * 0.1 for _ in range(4)]
    out, attention = multihead_self_attention(x, *weights, num_heads=2)
    assert out.shape == x.shape
    assert attention.shape == (2, 2, 5, 5)
    np.testing.assert_allclose(attention.sum(axis=-1), 1.0, rtol=1e-6)


def test_tiny_vit_forward_on_synthetic_images():
    rng = np.random.default_rng(2)
    images = rng.standard_normal((3, 8, 8, 1)).astype(np.float32)
    params = make_tiny_vit_params(images.shape, patch_size=2, seed=3)
    logits_cls = tiny_vit_forward(images, params, use_class_token=True)
    logits_mean = tiny_vit_forward(images, params, use_class_token=False)
    assert logits_cls.shape == (3, 3)
    assert logits_mean.shape == (3, 3)
    assert np.isfinite(logits_cls).all() and np.isfinite(logits_mean).all()
    assert not np.allclose(logits_cls, logits_mean)
