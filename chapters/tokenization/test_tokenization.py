import numpy as np

from scratch.tokenization import (
    TINY_CORPUS,
    decode,
    embedding_backward,
    embedding_lookup,
    encode,
    one_hot_matmul,
    train_byte_bpe,
    utf8_bytes,
)
from solutions import corpus_token_lengths, repeated_index_gradient


def test_utf8_bytes_cover_one_to_four_byte_characters():
    assert [len(utf8_bytes(text)) for text in ["a", "é", "世", "🙂"]] == [1, 2, 3, 4]
    assert bytes(utf8_bytes("café")).decode("utf-8") == "café"


def test_byte_bpe_trains_deterministically_and_round_trips():
    vocab, merges = train_byte_bpe(TINY_CORPUS, vocab_size=266)
    assert len(vocab) == 266
    assert len(merges) == 10
    assert merges[0][3] >= merges[-1][3]
    for text in TINY_CORPUS + ("lower newer 🙂",):
        encoded = encode(text, merges)
        assert decode(encoded, vocab) == text
    before, after, size = corpus_token_lengths()
    assert (before, after, size) == (27, 10, 266)


def test_embedding_lookup_equals_one_hot_matmul():
    rng = np.random.default_rng(17)
    weight = rng.standard_normal((7, 4), dtype=np.float32)
    indices = np.array([[0, 2, 2], [6, 1, 0]])
    np.testing.assert_allclose(
        embedding_lookup(indices, weight),
        one_hot_matmul(indices, weight),
    )


def test_embedding_backward_scatter_adds_repeated_rows():
    indices = np.array([[0, 2, 2], [3, 2, 0]])
    grad_output = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    grad = embedding_backward(indices, grad_output, vocab_size=5)
    manual = np.zeros((5, 4), dtype=np.float64)
    for index, row_grad in zip(indices.ravel(), grad_output.reshape(-1, 4)):
        manual[index] += row_grad
    np.testing.assert_allclose(grad, manual)
    np.testing.assert_allclose(repeated_index_gradient()[1], [4.0, 4.0])
