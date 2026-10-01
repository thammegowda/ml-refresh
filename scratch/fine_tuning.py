"""Supervised fine-tuning and LoRA utilities (Chapter 33)."""
import numpy as np

__chapter__ = "fine-tuning"

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"


# tag::templates[]
def render_chat(messages, add_generation_prompt=False):
    """Render role-marked messages into a tiny chat template string."""
    parts = []
    for message in messages:
        role = message["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {role}")
        parts.extend([ROLE_TOKENS[role], message["content"], END_TOKEN])
    if add_generation_prompt:
        parts.append(ROLE_TOKENS["assistant"])
    return " ".join(parts)


def next_token_training_arrays(tokens, token_roles):
    """Inputs, next-token targets, and a mask for assistant targets only."""
    tokens = np.asarray(tokens)
    roles = np.asarray(token_roles)
    if tokens.shape != roles.shape:
        raise ValueError("tokens and token_roles must have the same shape")
    inputs = tokens[:-1]
    targets = tokens[1:]
    train_mask = roles[1:] == "assistant"
    return inputs, targets, train_mask
# end::templates[]


# tag::masked-loss[]
def softmax(logits, axis=-1):
    """Stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def masked_cross_entropy(logits, targets, train_mask):
    """Mean next-token cross-entropy over masked positions, with gradient."""
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.int64)
    mask = np.asarray(train_mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("at least one position must be trainable")
    probabilities = softmax(logits, axis=-1)
    rows = np.arange(targets.shape[0])
    losses = -np.log(probabilities[rows, targets])
    normalizer = np.sum(mask)
    loss = np.sum(np.where(mask, losses, 0.0)) / normalizer
    grad_logits = probabilities.copy()
    grad_logits[rows, targets] -= 1.0
    grad_logits *= mask[:, None] / normalizer
    return loss, grad_logits
# end::masked-loss[]


def pack_documents(documents, max_length, boundary_id, pad_id=0):
    """Greedily pack tokenized documents, inserting untrained boundaries."""
    chunks, chunk_ids = [], []
    current, current_ids = [], []

    def flush():
        if not current:
            return
        pad = max_length - len(current)
        chunks.append(current + [pad_id] * pad)
        chunk_ids.append(current_ids + [-1] * pad)
        current.clear()
        current_ids.clear()

    for doc_id, document in enumerate(documents):
        seq = list(document) + [boundary_id]
        ids = [doc_id] * len(document) + [-1]
        if len(seq) <= max_length and len(current) + len(seq) > max_length:
            flush()
        for token, owner in zip(seq, ids):
            if len(current) == max_length:
                flush()
            current.append(token)
            current_ids.append(owner)
    flush()
    return np.array(chunks), np.array(chunk_ids)


# tag::packing[]
def packed_lm_arrays(packed_tokens, packed_doc_ids):
    """Next-token arrays whose mask forbids document-boundary targets."""
    inputs = packed_tokens[:, :-1]
    targets = packed_tokens[:, 1:]
    same_document = packed_doc_ids[:, :-1] == packed_doc_ids[:, 1:]
    train_mask = same_document & (packed_doc_ids[:, 1:] >= 0)
    return inputs, targets, train_mask
# end::packing[]


# tag::lora-forward[]
def init_lora(d_in, d_out, rank, rng, scale=0.01):
    """Return A and zero B so the LoRA layer matches the frozen base at init."""
    a = rng.normal(0.0, scale, size=(rank, d_out)).astype(np.float32)
    b = np.zeros((d_in, rank), dtype=np.float32)
    return a, b


def lora_output(x, w, a, b, alpha):
    """Compute X @ (W + (alpha / rank) * B @ A)."""
    rank = a.shape[0]
    adapted = w + (alpha / rank) * (b @ a)
    return x @ adapted


def merge_lora(w, a, b, alpha):
    """Fold the trained low-rank update into W for inference."""
    return w + (alpha / a.shape[0]) * (b @ a)
# end::lora-forward[]


# tag::lora-gradients[]
def lora_gradients(x, a, b, alpha, grad_y):
    """Backpropagate through the LoRA update for a loss on Y."""
    scale = alpha / a.shape[0]
    grad_update = x.T @ grad_y
    grad_a = scale * b.T @ grad_update
    grad_b = scale * grad_update @ a.T
    return grad_a, grad_b


def lora_mse_loss_and_grads(x, w, a, b, alpha, target):
    """Tiny objective used by the chapter tests."""
    y = lora_output(x, w, a, b, alpha)
    diff = y - target
    loss = 0.5 * np.mean(diff * diff)
    grad_y = diff / diff.size
    grad_a, grad_b = lora_gradients(x, a, b, alpha, grad_y)
    return loss, grad_a, grad_b


def lora_parameter_savings(d_in, d_out, rank):
    base = d_in * d_out
    trainable = rank * (d_in + d_out)
    return base, trainable, base / trainable
# end::lora-gradients[]
