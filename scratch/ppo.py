"""Reward models and PPO utilities (Chapter 35)."""
import numpy as np

from scratch.reinforcement_learning import softmax

__chapter__ = "ppo"


def make_preference_pairs(rng, n_pairs=80, n_features=4):
    """Synthetic pairs whose winner has the larger hidden linear reward."""
    true_w = np.linspace(-0.6, 0.9, n_features)
    left = rng.normal(size=(n_pairs, n_features))
    right = rng.normal(size=(n_pairs, n_features))
    left_score = left @ true_w
    right_score = right @ true_w
    left_wins = left_score >= right_score
    winners = np.where(left_wins[:, None], left, right)
    losers = np.where(left_wins[:, None], right, left)
    return winners, losers, true_w


# tag::reward-model[]
def bradley_terry_loss_and_grad(weights, winners, losers):
    """Loss -log sigmoid(r_w - r_l) for a linear reward model."""
    weights = np.asarray(weights, dtype=np.float64)
    features = np.asarray(winners) - np.asarray(losers)
    margins = features @ weights
    loss = np.mean(np.logaddexp(0.0, -margins))
    sigmoid = 1.0 / (1.0 + np.exp(-margins))
    grad = ((sigmoid - 1.0)[:, None] * features).mean(axis=0)
    return float(loss), grad


def train_reward_model(winners, losers, steps=300, lr=0.5):
    weights = np.zeros(winners.shape[1], dtype=np.float64)
    for _ in range(steps):
        _, grad = bradley_terry_loss_and_grad(weights, winners, losers)
        weights -= lr * grad
    return weights
# end::reward-model[]


# tag::rlhf-objective[]
def categorical_kl(policy_probs, reference_probs):
    """KL(pi || pi_ref) for categorical policies."""
    policy_probs = np.asarray(policy_probs, dtype=np.float64)
    reference_probs = np.asarray(reference_probs, dtype=np.float64)
    log_ratio = np.log(policy_probs) - np.log(reference_probs)
    return float(np.sum(policy_probs * log_ratio))


def rlhf_objective(policy_probs, rewards, reference_probs, beta):
    """E_pi[r] - beta KL(pi || pi_ref)."""
    reward_term = float(np.asarray(policy_probs) @ np.asarray(rewards))
    return reward_term - beta * categorical_kl(policy_probs, reference_probs)
# end::rlhf-objective[]


# tag::ppo-objective[]
def ppo_clipped_objective_and_grad(
    logits,
    old_logits,
    actions,
    advantages,
    clip_eps=0.2,
):
    """Mean PPO clipped surrogate and its gradient for one categorical state."""
    logits = np.asarray(logits, dtype=np.float64)
    old_logits = np.asarray(old_logits, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    advantages = np.asarray(advantages, dtype=np.float64)
    probs = softmax(logits)
    old_probs = softmax(old_logits)
    ratios = probs[actions] / old_probs[actions]
    clipped = np.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
    objective_terms = np.minimum(ratios * advantages, clipped * advantages)

    active = np.where(
        advantages >= 0.0,
        ratios <= 1.0 + clip_eps,
        ratios >= 1.0 - clip_eps,
    )
    grad = np.zeros_like(logits)
    for action, ratio, advantage, is_active in zip(actions, ratios, advantages, active):
        if not is_active:
            continue
        grad_logp = -probs.copy()
        grad_logp[action] += 1.0
        grad += advantage * ratio * grad_logp
    return float(np.mean(objective_terms)), grad / len(actions)
# end::ppo-objective[]


# tag::ppo-run[]
def categorical_entropy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return float(-np.sum(probs * np.log(probs)))


def value_loss_and_grad(values, returns):
    values = np.asarray(values, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    diff = values - returns
    return float(0.5 * np.mean(diff * diff)), diff / diff.size


def toy_ppo_run(action_rewards, steps=80, batch_size=96, lr=0.35, seed=0):
    """PPO on a one-state categorical policy with known action rewards."""
    rng = np.random.default_rng(seed)
    rewards = np.asarray(action_rewards, dtype=np.float64)
    logits = np.zeros_like(rewards)
    history = []
    for _ in range(steps):
        old_logits = logits.copy()
        old_probs = softmax(old_logits)
        actions = rng.choice(len(rewards), size=batch_size, p=old_probs)
        batch_rewards = rewards[actions]
        advantages = batch_rewards - np.mean(batch_rewards)
        for _ in range(4):
            _, grad = ppo_clipped_objective_and_grad(
                logits,
                old_logits,
                actions,
                advantages,
            )
            logits += lr * grad
        history.append(softmax(logits))
    return logits, np.array(history)
# end::ppo-run[]
