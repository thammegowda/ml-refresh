"""Tiny retrieval and pass@k utilities for Chapter 43."""
import hashlib
import math
import re
from itertools import product

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN_RE.findall(text.lower())


# tag::retrieval[]
def chunk_words(text, size=40, overlap=8):
    """Split text into overlapping word chunks."""
    words = text.split()
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def hashed_embedding(text, dims=64):
    """A deterministic bag-of-words embedding with signed hash buckets."""
    vector = np.zeros(dims, dtype=np.float64)
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def cosine_top_k(query, documents, k=2, dims=64):
    q = hashed_embedding(query, dims)
    matrix = np.vstack([hashed_embedding(doc, dims) for doc in documents])
    scores = matrix @ q
    order = np.argsort(-scores)[:k]
    return [(int(index), float(scores[index])) for index in order]


def retrieval_prompt(question, documents, k=2):
    hits = cosine_top_k(question, documents, k)
    context = "\n".join(f"[{i}] {documents[i]}" for i, _ in hits)
    return f"Use only this context:\n{context}\n\nQuestion: {question}"
# end::retrieval[]


# tag::memory[]
def add_vector_memory(memory, text, dims=64):
    memory.append({"text": text, "embedding": hashed_embedding(text, dims)})


def recall_vector_memory(memory, query, k=2, dims=64):
    if not memory:
        return []
    q = hashed_embedding(query, dims)
    scores = [float(item["embedding"] @ q) for item in memory]
    order = np.argsort(-np.array(scores))[:k]
    return [(memory[int(i)]["text"], scores[int(i)]) for i in order]
# end::memory[]


# tag::pass-at-k[]
def pass_at_k(n, c, k):
    """Unbiased estimator: probability a k-subset contains a correct sample."""
    if not 0 <= c <= n:
        raise ValueError("c must be between 0 and n")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")
    if n - c < k:
        return 1.0
    failed = 1.0
    for i in range(k):
        failed *= (n - c - i) / (n - i)
    return 1.0 - failed


def expected_pass_at_k(n, k, p):
    total = 0.0
    for bits in product((0, 1), repeat=n):
        c = sum(bits)
        probability = (p ** c) * ((1 - p) ** (n - c))
        total += probability * pass_at_k(n, c, k)
    return total
# end::pass-at-k[]
