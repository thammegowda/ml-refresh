"""Small NumPy optimizers and schedules (Chapter 15)."""
import numpy as np

__chapter__ = "optimization"


# tag::clip[]
def global_norm(grads):
    return float(np.sqrt(sum(np.sum(np.asarray(g, dtype=np.float64) ** 2)
                             for g in grads.values())))


def clip_by_global_norm(grads, max_norm, eps=1e-12):
    norm = global_norm(grads)
    scale = min(1.0, max_norm / (norm + eps))
    return {name: grad * scale for name, grad in grads.items()}, norm
# end::clip[]


# tag::momentum-rmsprop[]
def momentum_state(params):
    return {"velocity": {name: np.zeros_like(value) for name, value in params.items()}}


def heavy_ball_step(params, grads, state, lr=1e-2, beta=0.9):
    for name, value in params.items():
        velocity = state["velocity"][name]
        velocity *= beta
        velocity -= lr * grads[name]
        value += velocity


def rmsprop_state(params):
    return {"square": {name: np.zeros_like(value) for name, value in params.items()}}


def rmsprop_step(params, grads, state, lr=1e-3, decay=0.99, eps=1e-8):
    for name, value in params.items():
        square = state["square"][name]
        square *= decay
        square += (1 - decay) * grads[name] * grads[name]
        value -= lr * grads[name] / (np.sqrt(square) + eps)
# end::momentum-rmsprop[]


# tag::adamw[]
def adam_state(params):
    return {
        "t": 0,
        "m": {name: np.zeros_like(value) for name, value in params.items()},
        "v": {name: np.zeros_like(value) for name, value in params.items()},
    }


def adamw_step(params, grads, state, lr=1e-3, beta1=0.9, beta2=0.999,
               eps=1e-8, weight_decay=0.0, decoupled=True):
    state["t"] += 1
    for name, value in params.items():
        grad = grads[name]
        if weight_decay and not decoupled:
            grad = grad + weight_decay * value
        state["m"][name] = beta1 * state["m"][name] + (1 - beta1) * grad
        state["v"][name] = beta2 * state["v"][name] + (1 - beta2) * grad * grad
        m_hat = state["m"][name] / (1 - beta1 ** state["t"])
        v_hat = state["v"][name] / (1 - beta2 ** state["t"])
        if weight_decay and decoupled:
            value *= 1 - lr * weight_decay
        value -= lr * m_hat / (np.sqrt(v_hat) + eps)
# end::adamw[]


# tag::schedules[]
def linear_warmup(step, warmup_steps, peak_lr):
    if warmup_steps <= 0:
        return peak_lr
    return peak_lr * min(1.0, (step + 1) / warmup_steps)


def cosine_decay(step, total_steps, peak_lr, final_lr=0.0):
    if total_steps <= 1:
        return final_lr
    progress = min(1.0, max(0.0, step / (total_steps - 1)))
    weight = 0.5 * (1 + np.cos(np.pi * progress))
    return final_lr + (peak_lr - final_lr) * weight


def wsd_schedule(step, total_steps, warmup_steps, stable_steps, peak_lr,
                 final_lr=0.0):
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps, peak_lr)
    decay_steps = max(1, total_steps - warmup_steps - stable_steps)
    if step < warmup_steps + stable_steps:
        return peak_lr
    return cosine_decay(step - warmup_steps - stable_steps, decay_steps,
                        peak_lr, final_lr)
# end::schedules[]


# tag::muon[]
def orthogonalize_newton_schulz(matrix, steps=20, eps=1e-7):
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2-D matrix")
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T.copy() if transposed else matrix.copy()
    x = x.astype(np.float64, copy=False)
    x /= np.linalg.norm(x) + eps
    a, b, c = 15 / 8, -10 / 8, 3 / 8
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


def muon_update(gradient, momentum, beta=0.95, steps=20):
    if gradient.ndim != 2:
        raise ValueError("Muon applies to 2-D hidden weight matrices")
    momentum *= beta
    momentum += (1 - beta) * gradient
    return orthogonalize_newton_schulz(momentum, steps=steps)
# end::muon[]
