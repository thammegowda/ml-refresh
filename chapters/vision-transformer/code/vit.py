import numpy as np


# tag::patchify[]
def patchify(images, patch_size):
    """Images (B, H, W, C) -> flattened patches (B, GH*GW, P*P*C)."""
    b, height, width, channels = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    gh, gw = height // patch_size, width // patch_size
    x = images.reshape(b, gh, patch_size, gw, patch_size, channels)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, gh * gw, patch_size * patch_size * channels)


def sequence_length(height, width, patch_size, class_token=False):
    tokens = (height // patch_size) * (width // patch_size)
    return tokens + int(class_token)
# end::patchify[]


# tag::embedding[]
def linear_patch_embedding(patches, weight, bias):
    return patches @ weight + bias


def conv2d_strided(images, kernel, bias, stride):
    """Small NHWC convolution with kernel (P, P, C, D) and stride P."""
    b, height, width, _ = images.shape
    patch = kernel.shape[0]
    gh, gw = height // stride, width // stride
    out = np.empty((b, gh, gw, kernel.shape[-1]), dtype=images.dtype)
    for r in range(gh):
        for c in range(gw):
            window = images[:, r * stride:r * stride + patch,
                            c * stride:c * stride + patch, :]
            out[:, r, c, :] = np.tensordot(window, kernel,
                                           axes=([1, 2, 3], [0, 1, 2])) + bias
    return out
# end::embedding[]


# tag::positions[]
def add_2d_position(tokens, grid_hw, row_embed, col_embed):
    """Add learned row+column embeddings to patch tokens."""
    gh, gw = grid_hw
    pos = row_embed[:gh, None, :] + col_embed[None, :gw, :]
    return tokens + pos.reshape(gh * gw, -1)[None, :, :]


def prepend_class_token(tokens, class_token):
    batch = tokens.shape[0]
    cls = np.broadcast_to(class_token, (batch, 1, tokens.shape[-1]))
    return np.concatenate([cls, tokens], axis=1)


def pool_sequence(sequence, use_class_token=True):
    return sequence[:, 0] if use_class_token else sequence.mean(axis=1)
# end::positions[]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                    * (x + 0.044715 * x**3)))


# tag::attention[]
def split_heads(x, num_heads):
    b, tokens, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(b, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)


def combine_heads(x):
    b, heads, tokens, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, tokens, heads * head_dim)


def multihead_self_attention(x, wq, wk, wv, wo, num_heads):
    q = split_heads(x @ wq, num_heads)
    k = split_heads(x @ wk, num_heads)
    v = split_heads(x @ wv, num_heads)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
    weights = softmax(scores, axis=-1)
    return combine_heads(weights @ v) @ wo, weights
# end::attention[]


# tag::forward[]
def transformer_block(x, params):
    attn, _ = multihead_self_attention(layer_norm(x), params["wq"], params["wk"],
                                       params["wv"], params["wo"],
                                       params["num_heads"])
    x = x + attn
    hidden = gelu(layer_norm(x) @ params["mlp_w1"] + params["mlp_b1"])
    return x + hidden @ params["mlp_w2"] + params["mlp_b2"]


def tiny_vit_forward(images, params, use_class_token=True):
    patches = patchify(images, params["patch_size"])
    tokens = linear_patch_embedding(patches, params["patch_w"], params["patch_b"])
    gh = images.shape[1] // params["patch_size"]
    gw = images.shape[2] // params["patch_size"]
    tokens = add_2d_position(tokens, (gh, gw), params["row_pos"],
                             params["col_pos"])
    if use_class_token:
        tokens = prepend_class_token(tokens, params["class_token"])
    encoded = transformer_block(tokens, params)
    pooled = pool_sequence(encoded, use_class_token)
    return pooled @ params["head_w"] + params["head_b"]
# end::forward[]


def make_tiny_vit_params(image_shape, patch_size=2, dim=8, heads=2, classes=3,
                         seed=0):
    rng = np.random.default_rng(seed)
    _, height, width, channels = image_shape
    patch_dim = patch_size * patch_size * channels
    hidden = dim * 2

    def normal(shape, scale=0.1):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    return {
        "patch_size": patch_size,
        "patch_w": normal((patch_dim, dim)),
        "patch_b": np.zeros(dim, dtype=np.float32),
        "row_pos": normal((height // patch_size, dim), 0.02),
        "col_pos": normal((width // patch_size, dim), 0.02),
        "class_token": normal((1, dim), 0.02),
        "wq": normal((dim, dim)),
        "wk": normal((dim, dim)),
        "wv": normal((dim, dim)),
        "wo": normal((dim, dim)),
        "num_heads": heads,
        "mlp_w1": normal((dim, hidden)),
        "mlp_b1": np.zeros(hidden, dtype=np.float32),
        "mlp_w2": normal((hidden, dim)),
        "mlp_b2": np.zeros(dim, dtype=np.float32),
        "head_w": normal((dim, classes)),
        "head_b": np.zeros(classes, dtype=np.float32),
    }
