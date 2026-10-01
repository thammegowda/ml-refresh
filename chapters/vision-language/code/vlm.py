import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def normalize_rows(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


# tag::clip-zero-shot[]
def clip_zero_shot(image_embeddings, prompt_embeddings, logit_scale=10.0):
    """Return prompt probabilities from cosine similarities."""
    image = normalize_rows(image_embeddings)
    prompts = normalize_rows(prompt_embeddings)
    logits = logit_scale * (image @ prompts.T)
    return softmax(logits, axis=1)
# end::clip-zero-shot[]


# tag::projectors[]
def linear_projector(image_tokens, weight, bias):
    """Map visual token width to the LLM embedding width."""
    return image_tokens @ weight + bias


def mlp_projector(image_tokens, w1, b1, w2, b2):
    hidden = np.tanh(image_tokens @ w1 + b1)
    return hidden @ w2 + b2
# end::projectors[]


# tag::resampler[]
def cross_attention(queries, context):
    """Single-head cross-attention: queries attend to context tokens."""
    scores = queries @ context.transpose(0, 2, 1) / np.sqrt(queries.shape[-1])
    weights = softmax(scores, axis=-1)
    return weights @ context, weights


def perceiver_resampler(image_tokens, learned_queries):
    """Return a fixed number of visual tokens, regardless of image token count."""
    batch = image_tokens.shape[0]
    queries = np.broadcast_to(learned_queries, (batch,) + learned_queries.shape)
    return cross_attention(queries, image_tokens)[0]
# end::resampler[]


# tag::gated-interleave[]
def flamingo_gated_cross_attention(text_tokens, image_tokens, gate=0.0):
    attended, _ = cross_attention(text_tokens, image_tokens)
    return text_tokens + np.tanh(gate) * attended


def interleave_image_tokens(text_tokens, image_tokens, image_at):
    before = text_tokens[:, :image_at]
    after = text_tokens[:, image_at:]
    return np.concatenate([before, image_tokens, after], axis=1)
# end::gated-interleave[]


# tag::mrope-merge[]
def mrope_position_ids(frames, grid_h, grid_w):
    ids = []
    for t in range(frames):
        for h in range(grid_h):
            for w in range(grid_w):
                ids.append((t, h, w))
    return np.array(ids, dtype=np.int32)


def dynamic_token_counts(height, width, patch_size, merge=2):
    grid_h, grid_w = height // patch_size, width // patch_size
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    if grid_h % merge or grid_w % merge:
        raise ValueError("patch grid must be divisible by merge")
    before = grid_h * grid_w
    after = (grid_h // merge) * (grid_w // merge)
    return (grid_h, grid_w), before, after


def merge_2x2_tokens(tokens, grid_hw):
    batch, _, dim = tokens.shape
    grid_h, grid_w = grid_hw
    x = tokens.reshape(batch, grid_h, grid_w, dim)
    x = x.reshape(batch, grid_h // 2, 2, grid_w // 2, 2, dim)
    return x.mean(axis=(2, 4)).reshape(batch, (grid_h // 2) * (grid_w // 2), dim)
# end::mrope-merge[]
