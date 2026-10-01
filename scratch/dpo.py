"""Direct Preference Optimization helpers (Chapter 36)."""
import numpy as np

__chapter__ = "dpo"


def log_softmax(logits):
    logits = np.asarray(logits)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# tag::closed-form[]
def kl_regularized_optimum(reference, reward, beta):
    """The policy pi*(y) proportional to pi_ref(y) exp(r(y) / beta)."""
    reference = np.asarray(reference, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    unnormalized = reference * np.exp(reward / beta)
    return unnormalized / np.sum(unnormalized)
# end::closed-form[]


# tag::dpo-loss[]
def dpo_loss_and_grad(logits, reference, pairs, beta):
    """Mean DPO loss and gradient for (winner, loser) categorical pairs."""
    logits = np.asarray(logits, dtype=np.float64)
    logp = log_softmax(logits)
    log_ref = np.log(np.asarray(reference, dtype=np.float64))
    grad = np.zeros_like(logits)
    losses = []
    for winner, loser in pairs:
        delta = beta * ((logp[winner] - log_ref[winner])
                        - (logp[loser] - log_ref[loser]))
        weight = sigmoid(-delta)
        losses.append(np.logaddexp(0.0, -delta))
        grad[winner] -= beta * weight
        grad[loser] += beta * weight
    return float(np.mean(losses)), grad / len(pairs)
# end::dpo-loss[]


def soft_dpo_loss_and_grad(logits, reference, rewards, beta):
    """Expected DPO loss for all unordered pairs under Bradley-Terry labels."""
    logits = np.asarray(logits, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    logp = log_softmax(logits)
    log_ref = np.log(np.asarray(reference, dtype=np.float64))
    grad = np.zeros_like(logits)
    total = 0.0
    count = 0
    for i in range(len(logits)):
        for j in range(i + 1, len(logits)):
            target = sigmoid(rewards[i] - rewards[j])
            delta = beta * ((logp[i] - log_ref[i]) - (logp[j] - log_ref[j]))
            pred = sigmoid(delta)
            total += -(target * np.log(pred) + (1.0 - target) * np.log1p(-pred))
            grad[i] += beta * (pred - target)
            grad[j] -= beta * (pred - target)
            count += 1
    return float(total / count), grad / count


# tag::toy-run[]
def fit_toy_dpo(reference, rewards, beta=0.7, steps=800, lr=0.8):
    """Optimize a tiny categorical policy and return (policy, optimum)."""
    logits = np.log(np.asarray(reference, dtype=np.float64))
    for _ in range(steps):
        _, grad = soft_dpo_loss_and_grad(logits, reference, rewards, beta)
        logits -= lr * grad
    policy = np.exp(log_softmax(logits))
    optimum = kl_regularized_optimum(reference, rewards, beta)
    return policy, optimum
# end::toy-run[]
