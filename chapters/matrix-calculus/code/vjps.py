"""Vector-Jacobian products for Appendix C."""
import numpy as np

from scratch.arrays import unbroadcast


# tag::core[]
def add_forward(x, y):
    return x + y


def add_vjp(grad, x, y):
    return unbroadcast(grad, x.shape), unbroadcast(grad, y.shape)


def multiply_forward(x, y):
    return x * y


def multiply_vjp(grad, x, y):
    return unbroadcast(grad * y, x.shape), unbroadcast(grad * x, y.shape)


def matmul_forward(a, b):
    return a @ b


def matmul_vjp(grad, a, b):
    return grad @ b.T, a.T @ grad
# end::core[]


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, tuple):
        axes = axis
    else:
        axes = (axis,)
    return tuple(a if a >= 0 else ndim + a for a in axes)


def sum_forward(x, axis=None, keepdims=False):
    return np.sum(x, axis=axis, keepdims=keepdims)


def sum_vjp(grad, x_shape, axis=None, keepdims=False):
    if axis is None:
        return np.ones(x_shape, dtype=np.asarray(grad).dtype) * grad
    expanded = grad
    if not keepdims:
        for ax in sorted(_axes(axis, len(x_shape))):
            expanded = np.expand_dims(expanded, ax)
    return np.broadcast_to(expanded, x_shape)


def mean_forward(x, axis=None, keepdims=False):
    return np.mean(x, axis=axis, keepdims=keepdims)


def mean_vjp(grad, x_shape, axis=None, keepdims=False):
    axes = _axes(axis, len(x_shape))
    count = int(np.prod([x_shape[ax] for ax in axes]))
    return sum_vjp(grad / count, x_shape, axis, keepdims)


# tag::elementwise[]
def exp_forward(x):
    return np.exp(x)


def exp_vjp(grad, y):
    return grad * y


def log_forward(x):
    return np.log(x)


def log_vjp(grad, x):
    return grad / x


def relu_forward(x):
    return np.maximum(x, 0)


def relu_vjp(grad, x):
    return grad * (x > 0)


def sigmoid_forward(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_vjp(grad, y):
    return grad * y * (1 - y)


def tanh_forward(x):
    return np.tanh(x)


def tanh_vjp(grad, y):
    return grad * (1 - y * y)
# end::elementwise[]


# tag::softmax[]
# tag::softmax-vjp[]
def softmax_forward(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_vjp(grad, y, axis=-1):
    dot = np.sum(grad * y, axis=axis, keepdims=True)
    return y * (grad - dot)
# end::softmax-vjp[]


def log_softmax_forward(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_z


def log_softmax_vjp(grad, log_y, axis=-1):
    y = np.exp(log_y)
    return grad - y * np.sum(grad, axis=axis, keepdims=True)


def cross_entropy_forward(logits, labels):
    log_probs = log_softmax_forward(logits, axis=-1)
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def cross_entropy_vjp(logits, labels):
    grad = softmax_forward(logits, axis=-1)
    grad[np.arange(len(labels)), labels] -= 1
    return grad / len(labels)
# end::softmax[]


# tag::normalization[]
def layer_norm_forward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    inv = 1 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
    normalized = centered * inv
    return normalized * gamma + beta


def layer_norm_vjp(grad, x, gamma, beta, eps=1e-5):
    del beta
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    inv = 1 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
    normalized = centered * inv
    grad_norm = grad * gamma
    width = x.shape[-1]
    grad_x = inv / width * (
        width * grad_norm
        - np.sum(grad_norm, axis=-1, keepdims=True)
        - normalized * np.sum(grad_norm * normalized, axis=-1, keepdims=True)
    )
    axes = tuple(range(grad.ndim - 1))
    return grad_x, np.sum(grad * normalized, axis=axes), np.sum(grad, axis=axes)


def rms_norm_forward(x, gamma, eps=1e-5):
    inv = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * inv * gamma


def rms_norm_vjp(grad, x, gamma, eps=1e-5):
    inv = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    normalized = x * inv
    grad_norm = grad * gamma
    width = x.shape[-1]
    inner = np.sum(grad_norm * x, axis=-1, keepdims=True)
    grad_x = grad_norm * inv - x * (inv ** 3) * inner / width
    axes = tuple(range(grad.ndim - 1))
    return grad_x, np.sum(grad * normalized, axis=axes)
# end::normalization[]


# tag::shape-and-embedding[]
def embedding_forward(weights, ids):
    return weights[ids]


def embedding_vjp(grad, ids, vocab_size):
    grad_weights = np.zeros((vocab_size, grad.shape[-1]), dtype=grad.dtype)
    np.add.at(grad_weights, ids, grad)
    return grad_weights


def reshape_forward(x, shape):
    return np.reshape(x, shape)


def reshape_vjp(grad, original_shape):
    return np.reshape(grad, original_shape)


def transpose_forward(x, axes):
    return np.transpose(x, axes)


def transpose_vjp(grad, axes):
    inverse = np.argsort(axes)
    return np.transpose(grad, inverse)
# end::shape-and-embedding[]


# tag::attention[]
def scaled_dot_product_attention(q, k, v):
    scale = 1 / np.sqrt(q.shape[-1])
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    probabilities = softmax_forward(scores, axis=-1)
    return probabilities @ v


def attention_vjp(grad, q, k, v):
    scale = 1 / np.sqrt(q.shape[-1])
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    probabilities = softmax_forward(scores, axis=-1)
    grad_v = np.swapaxes(probabilities, -1, -2) @ grad
    grad_prob = grad @ np.swapaxes(v, -1, -2)
    grad_scores = softmax_vjp(grad_prob, probabilities, axis=-1)
    grad_q = (grad_scores @ k) * scale
    grad_k = (np.swapaxes(grad_scores, -1, -2) @ q) * scale
    return grad_q, grad_k, grad_v
# end::attention[]
