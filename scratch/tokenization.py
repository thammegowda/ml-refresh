"""Byte-level BPE and embedding lookup utilities (Chapter 17)."""
import numpy as np

__chapter__ = "tokenization"

TINY_CORPUS = ("low lower lowest", "newer wider")


def utf8_bytes(text):
    """Return the integer UTF-8 bytes that represent text."""
    return list(text.encode("utf-8"))


def _merge_pair(tokens, pair, new_id):
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


# tag::train[]
def train_byte_bpe(texts, vocab_size):
    """Train byte-level BPE by repeatedly merging the most common pair."""
    if vocab_size < 256:
        raise ValueError("byte-level BPE needs room for all 256 bytes")
    sequences = [tuple(utf8_bytes(text)) for text in texts]
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    while len(vocab) < vocab_size:
        counts = {}
        for tokens in sequences:
            for pair in zip(tokens, tokens[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        new_id = len(vocab)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((pair[0], pair[1], new_id, count))
        sequences = [tuple(_merge_pair(tokens, pair, new_id)) for tokens in sequences]
    return vocab, merges
# end::train[]


# tag::encode[]
def encode(text, merges):
    """Encode text by applying learned merges in order."""
    tokens = utf8_bytes(text)
    for left, right, new_id, _count in merges:
        tokens = _merge_pair(tokens, (left, right), new_id)
    return tokens


def decode(tokens, vocab):
    """Decode token ids by concatenating their byte strings, then UTF-8 decoding."""
    return b"".join(vocab[int(token)] for token in tokens).decode("utf-8")
# end::encode[]


# tag::embedding[]
def embedding_lookup(indices, weight):
    """Gather embedding rows: weight[indices]."""
    return weight[np.asarray(indices)]


def one_hot_matmul(indices, weight):
    """The same lookup, written as a one-hot matrix multiply."""
    one_hot = np.eye(weight.shape[0], dtype=weight.dtype)[np.asarray(indices)]
    return one_hot @ weight


def embedding_backward(indices, grad_output, vocab_size):
    """Scatter-add output gradients into the rows that were gathered."""
    grad_weight = np.zeros((vocab_size, grad_output.shape[-1]), grad_output.dtype)
    np.add.at(grad_weight, np.asarray(indices), grad_output)
    return grad_weight
# end::embedding[]
