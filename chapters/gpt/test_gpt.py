import numpy as np

from gpt_train import (
    build_vocab,
    get_batch,
    gpt_loss_and_grads,
    init_gpt_params,
    sample_text,
    train_tiny_gpt,
    adamw_step,
)
from scratch.gradcheck import check_gradient


def rng(seed=23):
    return np.random.default_rng(seed)


def test_tied_embedding_and_final_norm_gradients_match_finite_differences():
    r = rng()
    params = init_gpt_params(6, 4, 1, 0, 8, r)
    params["tok_emb"] = params["tok_emb"].astype(np.float64)
    params["final_norm"] = params["final_norm"].astype(np.float64)
    tokens = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    targets = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
    _loss, grads = gpt_loss_and_grads(params, tokens, targets, n_heads=1)

    def emb_loss(embedding):
        trial = {
            "tok_emb": embedding,
            "blocks": [],
            "final_norm": params["final_norm"],
        }
        return gpt_loss_and_grads(trial, tokens, targets, n_heads=1)[0]

    def norm_loss(weight):
        trial = {"tok_emb": params["tok_emb"], "blocks": [], "final_norm": weight}
        return gpt_loss_and_grads(trial, tokens, targets, n_heads=1)[0]

    check_gradient(emb_loss, params["tok_emb"], grads["tok_emb"], tolerance=2e-6)
    check_gradient(norm_loss, params["final_norm"], grads["final_norm"], tolerance=2e-6)


def test_a_few_adamw_steps_reduce_loss_on_one_batch():
    r = rng()
    data, stoi, _itos = build_vocab()
    x, y = get_batch(data[:1800], batch_size=4, seq_len=12, rng=r)
    params = init_gpt_params(len(stoi), 16, 2, 1, 32, r)
    opt_state = {}
    losses = []
    for _ in range(10):
        loss, grads = gpt_loss_and_grads(params, x, y, n_heads=2)
        losses.append(loss)
        adamw_step(params, grads, opt_state, lr=0.01, weight_decay=0.0)
    assert losses[-1] < losses[0]


def test_sampling_is_deterministic_given_seed():
    r = rng()
    _data, stoi, itos = build_vocab()
    params = init_gpt_params(len(stoi), 12, 2, 1, 24, r)
    first = sample_text(params, "Alice", stoi, itos, 2, steps=12, seed=99, top_k=5)
    second = sample_text(params, "Alice", stoi, itos, 2, steps=12, seed=99, top_k=5)
    assert first == second
    assert len(first) == len("Alice") + 12


def test_training_loop_records_train_and_validation_losses():
    params, history, stoi, itos = train_tiny_gpt(
        steps=4,
        seed=24,
        d_model=12,
        n_heads=2,
        n_layers=1,
        hidden_dim=24,
        batch_size=4,
        seq_len=10,
        base_lr=5e-3,
    )
    assert history["train"].shape == (4,)
    assert history["val"].shape[1] == 2
    assert np.all(np.isfinite(history["train"]))
    assert params["tok_emb"].shape[0] == len(stoi) == len(itos)
