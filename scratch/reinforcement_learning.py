"""Reinforcement-learning foundations in NumPy (Chapter 34)."""
import numpy as np

__chapter__ = "reinforcement-learning"


# tag::returns-bellman[]
def discounted_returns(rewards, gamma):
    """Return G_t = r_t + gamma r_{t+1} + ... for one trajectory."""
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def evaluate_policy(transition, reward, gamma):
    """Solve v = r + gamma P v for a fixed policy's transition matrix."""
    transition = np.asarray(transition, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    system = np.eye(transition.shape[0]) - gamma * transition
    return np.linalg.solve(system, reward)


def tiny_chain(gamma):
    """Three-state chain: 0 -> 1 -> terminal, reward 1 on state 1."""
    transition = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    reward = np.array([0, 1, 0], dtype=np.float64)
    return evaluate_policy(transition, reward, gamma)
# end::returns-bellman[]


def softmax(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def bandit_objective(logits, action_values):
    """Expected one-step reward under a categorical policy."""
    probabilities = softmax(logits)
    return float(probabilities @ action_values)


def bandit_policy_gradient(logits, action_values):
    """Exact policy gradient for a categorical bandit."""
    probabilities = softmax(logits)
    value = probabilities @ action_values
    return probabilities * (action_values - value)


# tag::bandit[]
def reinforce_bandit(action_values, steps=200, batch_size=64, lr=0.2, seed=0):
    """Sampled REINFORCE updates for a one-state bandit."""
    rng = np.random.default_rng(seed)
    logits = np.zeros(len(action_values), dtype=np.float64)
    for _ in range(steps):
        probabilities = softmax(logits)
        actions = rng.choice(len(action_values), size=batch_size, p=probabilities)
        rewards = action_values[actions]
        baseline = np.mean(rewards)
        grad = np.zeros_like(logits)
        for action, reward in zip(actions, rewards):
            grad_logp = -probabilities.copy()
            grad_logp[action] += 1.0
            grad += (reward - baseline) * grad_logp
        logits += lr * grad / batch_size
    return logits, softmax(logits)
# end::bandit[]


# tag::off-policy[]
def importance_ratios(actions, target_probs, behavior_probs):
    """rho_t = pi(a_t) / b(a_t) for logged bandit actions."""
    actions = np.asarray(actions, dtype=np.int64)
    target_probs = np.asarray(target_probs, dtype=np.float64)
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)
    return target_probs[actions] / behavior_probs[actions]


def off_policy_value(actions, rewards, target_probs, behavior_probs):
    """Ordinary importance-sampling estimate of a target policy value."""
    ratios = importance_ratios(actions, target_probs, behavior_probs)
    return float(np.mean(ratios * rewards))
# end::off-policy[]


def td_errors(rewards, values, gamma):
    """delta_t = r_t + gamma V(s_{t+1}) - V(s_t)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return rewards + gamma * values[1:] - values[:-1]


# tag::gae[]
def gae_recursive(rewards, values, gamma, lam):
    """Generalized advantage estimates by the backward recursion."""
    deltas = td_errors(rewards, values, gamma)
    advantages = np.zeros_like(deltas)
    running = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        running = deltas[t] + gamma * lam * running
        advantages[t] = running
    return advantages
# end::gae[]


def gae_explicit(rewards, values, gamma, lam):
    """The explicit lambda-weighted sum of future TD errors."""
    deltas = td_errors(rewards, values, gamma)
    advantages = np.zeros_like(deltas)
    for t in range(len(deltas)):
        discount = 1.0
        for k in range(t, len(deltas)):
            advantages[t] += discount * deltas[k]
            discount *= gamma * lam
    return advantages
