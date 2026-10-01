"""Contrastive and metric-learning objectives (Chapter 30)."""
import numpy as np

__chapter__ = "contrastive-learning"


# tag::similarity[]
def normalize_rows(x, eps=1e-12):
    """Return rows scaled to unit length."""
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity(a, b):
    """All pairwise cosine similarities between row embeddings."""
    return normalize_rows(a) @ normalize_rows(b).T
# end::similarity[]


def cosine_distance(a, b):
    return 1.0 - np.sum(normalize_rows(a) * normalize_rows(b), axis=1)


# tag::metric-losses[]
def pairwise_contrastive_loss(a, b, same, margin=0.5):
    """Mean y d^2 + (1-y) max(0, margin-d)^2 for cosine distance d."""
    same = np.asarray(same, dtype=np.float64)
    d = cosine_distance(a, b)
    pull = same * d**2
    push = (1.0 - same) * np.maximum(0.0, margin - d) ** 2
    return float(np.mean(pull + push))


def hard_negative_indices(anchor, candidates):
    """For paired rows, choose the nonmatching candidate with largest cosine."""
    scores = cosine_similarity(anchor, candidates)
    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)
    return np.argmax(scores, axis=1)


def triplet_loss_hard_negative(anchor, positive, margin=0.2):
    """Mean max(0, d(a,p)-d(a,n)+margin), mining n from the batch."""
    negatives = positive[hard_negative_indices(anchor, positive)]
    d_pos = cosine_distance(anchor, positive)
    d_neg = cosine_distance(anchor, negatives)
    return float(np.mean(np.maximum(0.0, d_pos - d_neg + margin)))
# end::metric-losses[]


def softmax_rows(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# tag::info-nce[]
def info_nce_loss_and_grad(similarity, temperature=0.1):
    """Cross-entropy over rows, with correct pair on the diagonal."""
    n = similarity.shape[0]
    probabilities = softmax_rows(similarity / temperature)
    loss = -np.log(np.diag(probabilities)).mean()
    grad = probabilities.copy()
    grad[np.arange(n), np.arange(n)] -= 1.0
    grad /= n * temperature
    return float(loss), grad
# end::info-nce[]


# tag::clip-siglip[]
def clip_loss_and_grad(similarity, temperature=0.1):
    """Symmetric CLIP loss: row retrieval plus column retrieval."""
    row_loss, row_grad = info_nce_loss_and_grad(similarity, temperature)
    col_loss, col_grad_t = info_nce_loss_and_grad(similarity.T, temperature)
    return 0.5 * (row_loss + col_loss), 0.5 * (row_grad + col_grad_t.T)


def siglip_loss_and_grad(similarity, bias=0.0):
    """Pairwise sigmoid loss with +1 labels on the diagonal and -1 elsewhere."""
    n = similarity.shape[0]
    labels = -np.ones_like(similarity)
    labels[np.arange(n), np.arange(n)] = 1.0
    logits = labels * (similarity + bias)
    loss = np.logaddexp(0.0, -logits).mean()
    grad_logits = -labels / (1.0 + np.exp(logits)) / similarity.size
    grad_bias = float(np.sum(grad_logits))
    return float(loss), grad_logits, grad_bias
# end::clip-siglip[]


def _normalize_backward(upstream, raw, normalized):
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    dot = np.sum(upstream * normalized, axis=1, keepdims=True)
    return (upstream - normalized * dot) / np.maximum(norm, 1e-12)


# tag::retrieval[]
def make_synthetic_pairs(n=48, latent_dim=4, image_dim=7, text_dim=6, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, latent_dim)).astype(np.float32)
    image_map = rng.standard_normal((latent_dim, image_dim)).astype(np.float32)
    text_map = rng.standard_normal((latent_dim, text_dim)).astype(np.float32)
    images = z @ image_map + 0.05 * rng.standard_normal((n, image_dim))
    texts = z @ text_map + 0.05 * rng.standard_normal((n, text_dim))
    return images.astype(np.float32), texts.astype(np.float32)


def retrieval_recall_at_k(image_embeddings, text_embeddings, k=1):
    scores = cosine_similarity(image_embeddings, text_embeddings)
    topk = np.argsort(-scores, axis=1)[:, :k]
    target = np.arange(scores.shape[0])[:, None]
    return float(np.mean(np.any(topk == target, axis=1)))
# end::retrieval[]


def linear_pair_loss_and_grad(images, texts, w_image, w_text, temperature=0.2):
    raw_image, raw_text = images @ w_image, texts @ w_text
    image_embeddings = normalize_rows(raw_image)
    text_embeddings = normalize_rows(raw_text)
    loss, grad_scores = clip_loss_and_grad(image_embeddings @ text_embeddings.T,
                                           temperature)
    grad_image = _normalize_backward(grad_scores @ text_embeddings,
                                     raw_image, image_embeddings)
    grad_text = _normalize_backward(grad_scores.T @ image_embeddings,
                                    raw_text, text_embeddings)
    return loss, images.T @ grad_image, texts.T @ grad_text


# tag::train[]
def train_linear_pair(images, texts, embed_dim=4, steps=250, lr=0.8, seed=1):
    rng = np.random.default_rng(seed)
    w_image = (0.1 * rng.standard_normal((images.shape[1], embed_dim)))
    w_text = (0.1 * rng.standard_normal((texts.shape[1], embed_dim)))
    w_image, w_text = w_image.astype(np.float32), w_text.astype(np.float32)
    for _ in range(steps):
        _, grad_image, grad_text = linear_pair_loss_and_grad(images, texts,
                                                             w_image, w_text)
        w_image -= lr * grad_image.astype(np.float32)
        w_text -= lr * grad_text.astype(np.float32)
    return images @ w_image, texts @ w_text
# end::train[]
