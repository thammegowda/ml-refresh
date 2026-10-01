"""Worked solution helpers for Chapter 17."""
import numpy as np

from scratch.tokenization import (
    TINY_CORPUS,
    embedding_backward,
    encode,
    train_byte_bpe,
    utf8_bytes,
)


# tag::budget[]
def corpus_token_lengths(texts=TINY_CORPUS, vocab_size=266):
    """Byte count before BPE and token count after BPE on a tiny corpus."""
    _vocab, merges = train_byte_bpe(texts, vocab_size)
    before = sum(len(utf8_bytes(text)) for text in texts)
    after = sum(len(encode(text, merges)) for text in texts)
    return before, after, 256 + len(merges)
# end::budget[]


# tag::scatter[]
def repeated_index_gradient():
    """A repeated token id receives the sum of both upstream gradients."""
    indices = np.array([1, 3, 1])
    grad_output = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    return embedding_backward(indices, grad_output, vocab_size=5)
# end::scatter[]
