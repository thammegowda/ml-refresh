import numpy as np

from scratch.positional_encoding import softmax
from scratch.sampling import sample_categorical
from scratch.transformer import (
    init_block_params,
    rmsnorm_backward,
    rmsnorm_forward,
    transformer_block_backward,
    transformer_block_forward,
)

TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, "and what
is the use of a book," thought Alice, "without pictures or conversations?"

So she was considering in her own mind, as well as she could, for the hot day
made her feel very sleepy and stupid, whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I
shall be late!" When the Rabbit actually took a watch out of its waistcoat
pocket, and looked at it, and then hurried on, Alice started to her feet.

In another moment down went Alice after it, never once considering how in the
world she was to get out again. The rabbit-hole went straight on like a tunnel
for some way, and then dipped suddenly down, so suddenly that Alice had not a
moment to think about stopping herself before she found herself falling down a
very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of
time as she went down to look about her and to wonder what was going to happen
next. Down, down, down. Would the fall never come to an end? "I wonder how
many miles I have fallen by this time?" she said aloud.

Presently she began again. "Dinah will miss me very much to-night, I should
think!" Dinah was the cat. Alice was not a bit hurt, and she jumped up on to
her feet in a moment: she looked up, but it was all dark overhead; before her
was another long passage, and the White Rabbit was still in sight, hurrying
down it.

Away went Alice like the wind, and was just in time to hear it say, as it
turned a corner, "Oh my ears and whiskers, how late it is getting!" She was
close behind it when she turned the corner, but the Rabbit was no longer to be
seen. She found herself in a long, low hall, which was lit up by a row of
lamps hanging from the roof.
""".strip()


def build_vocab(text=TEXT):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = np.array([stoi[ch] for ch in text], dtype=np.int64)
    return data, stoi, itos


def split_data(data, fraction=0.9):
    cut = int(len(data) * fraction)
    return data[:cut], data[cut:]


def get_batch(data, batch_size, seq_len, rng):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i + seq_len] for i in starts])
    y = np.stack([data[i + 1:i + seq_len + 1] for i in starts])
    return x, y


def init_gpt_params(vocab_size, d_model, n_heads, n_layers, hidden_dim, rng):
    params = {
        "tok_emb": rng.normal(0.0, 0.02, size=(vocab_size, d_model)).astype(np.float32),
        "blocks": [],
        "final_norm": np.ones(d_model, dtype=np.float32),
    }
    for _ in range(n_layers):
        params["blocks"].append(init_block_params(d_model, n_heads, hidden_dim, rng))
    return params


def cross_entropy_loss(logits, targets):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    n = flat_targets.size
    loss = -np.mean(np.log(probs[np.arange(n), flat_targets] + 1e-12))
    dlogits = probs
    dlogits[np.arange(n), flat_targets] -= 1.0
    dlogits = (dlogits / n).reshape(logits.shape)
    return float(loss), dlogits


def gpt_logits(params, tokens, n_heads):
    x = params["tok_emb"][tokens]
    for block in params["blocks"]:
        x, _cache = transformer_block_forward(x, block, n_heads)
    x, _norm_cache = rmsnorm_forward(x, params["final_norm"])
    return x @ params["tok_emb"].T


def gpt_loss_and_grads(params, tokens, targets, n_heads):
    x = params["tok_emb"][tokens]
    block_caches = []
    for block in params["blocks"]:
        x, cache = transformer_block_forward(x, block, n_heads)
        block_caches.append(cache)
    normalized, norm_cache = rmsnorm_forward(x, params["final_norm"])
    logits = normalized @ params["tok_emb"].T
    loss, dlogits = cross_entropy_loss(logits, targets)
    flat_logits = dlogits.reshape(-1, dlogits.shape[-1])
    flat_norm = normalized.reshape(-1, normalized.shape[-1])
    grads = {
        "tok_emb": flat_logits.T @ flat_norm,
        "blocks": [None] * len(params["blocks"]),
    }
    dnormalized = dlogits @ params["tok_emb"]
    dx, grads["final_norm"] = rmsnorm_backward(dnormalized, norm_cache)
    for index in range(len(params["blocks"]) - 1, -1, -1):
        dx, grads["blocks"][index] = transformer_block_backward(dx, block_caches[index])
    np.add.at(grads["tok_emb"], tokens, dx)
    return loss, grads


def tree_items(params, grads, path=()):
    if isinstance(params, dict):
        for key in params:
            yield from tree_items(params[key], grads[key], path + (key,))
    elif isinstance(params, list):
        for index, value in enumerate(params):
            yield from tree_items(value, grads[index], path + (index,))
    else:
        yield path, params, grads


# tag::adamw[]
def adamw_step(params, grads, state, lr, weight_decay=0.01,
               beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] = state.get("t", 0) + 1
    t = state["t"]
    for path, param, grad in tree_items(params, grads):
        slot = state.setdefault(path, {
            "m": np.zeros_like(param),
            "v": np.zeros_like(param),
        })
        slot["m"] = beta1 * slot["m"] + (1.0 - beta1) * grad
        slot["v"] = beta2 * slot["v"] + (1.0 - beta2) * (grad * grad)
        m_hat = slot["m"] / (1.0 - beta1 ** t)
        v_hat = slot["v"] / (1.0 - beta2 ** t)
        param *= 1.0 - lr * weight_decay
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
# end::adamw[]


def warmup_lr(step, base_lr, warmup_steps):
    return base_lr * min(1.0, step / max(1, warmup_steps))


def estimate_loss(params, data, n_heads, rng, batch_size=8, seq_len=16, batches=2):
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, seq_len, rng)
        loss, _grads = gpt_loss_and_grads(params, x, y, n_heads)
        losses.append(loss)
    return float(np.mean(losses))


# tag::train[]
def train_tiny_gpt(steps=30, seed=23, d_model=24, n_heads=2, n_layers=1,
                   hidden_dim=48, batch_size=8, seq_len=16, base_lr=3e-3):
    rng = np.random.default_rng(seed)
    data, stoi, itos = build_vocab()
    train_data, val_data = split_data(data)
    params = init_gpt_params(len(stoi), d_model, n_heads, n_layers, hidden_dim, rng)
    opt_state = {}
    train_losses, val_points = [], []
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, batch_size, seq_len, rng)
        loss, grads = gpt_loss_and_grads(params, x, y, n_heads)
        lr = warmup_lr(step, base_lr, warmup_steps=5)
        adamw_step(params, grads, opt_state, lr)
        train_losses.append(loss)
        if step == 1 or step == steps or step % 5 == 0:
            val_rng = np.random.default_rng(seed + 10_000 + step)
            val = estimate_loss(params, val_data, n_heads, val_rng, batch_size, seq_len)
            val_points.append((step, val))
    history = {"train": np.array(train_losses), "val": np.array(val_points)}
    return params, history, stoi, itos
# end::train[]


# tag::sample[]
def sample_text(params, prompt, stoi, itos, n_heads, steps, seed=0,
                temperature=1.0, top_k=None, max_context=64):
    rng = np.random.default_rng(seed)
    ids = [stoi[ch] for ch in prompt]
    for _ in range(steps):
        context = np.array([ids[-max_context:]], dtype=np.int64)
        logits = gpt_logits(params, context, n_heads)[0, -1] / temperature
        if top_k is not None and top_k < logits.size:
            keep = np.argpartition(logits, -top_k)[-top_k:]
            masked = np.full_like(logits, -np.inf)
            masked[keep] = logits[keep]
            logits = masked
        probs = softmax(logits)
        ids.append(int(sample_categorical(probs[None, :], rng)[0]))
    return "".join(itos[i] for i in ids)
# end::sample[]
