"""GRPO helpers for verifiable rewards (Chapter 37)."""
import numpy as np

__chapter__ = "grpo"


def log_softmax(logits):
    logits = np.asarray(logits)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


# tag::groups[]
def arithmetic_reward(prompt, answer):
    """Return 1 when answer is the exact integer sum in a prompt (a, b)."""
    try:
        return float(int(str(answer).strip()) == int(prompt[0] + prompt[1]))
    except ValueError:
        return 0.0


def group_advantages(rewards, eps=1e-8):
    """A_i = (r_i - group_mean) / group_std for rewards shaped (B, G)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    centered = rewards - np.mean(rewards, axis=1, keepdims=True)
    std = np.std(rewards, axis=1, keepdims=True)
    return np.where(std > eps, centered / (std + eps), 0.0)
# end::groups[]


# tag::rloo[]
def rloo_advantages(rewards):
    """Leave-one-out baseline: compare each reward with its peers' mean."""
    rewards = np.asarray(rewards, dtype=np.float64)
    group_size = rewards.shape[1]
    if group_size < 2:
        raise ValueError("RLOO needs at least two samples per prompt")
    peer_sum = np.sum(rewards, axis=1, keepdims=True) - rewards
    return rewards - peer_sum / (group_size - 1)
# end::rloo[]


def normalization_weights(lengths, mode):
    lengths = np.asarray(lengths, dtype=np.float64)
    if mode == "sequence":
        return np.full_like(lengths, 1.0 / lengths.size)
    if mode == "token":
        return lengths / np.sum(lengths)
    raise ValueError("mode must be 'sequence' or 'token'")


# tag::grpo-loss[]
def k3_kl_to_reference(logp, log_ref):
    """Schulman's k3 estimate of KL(policy || reference) for policy samples."""
    log_ratio = log_ref - logp
    return np.expm1(log_ratio) - log_ratio


def grpo_loss_and_grad(logits, old_logits, ref_logits, actions, advantages,
                       lengths, beta_kl=0.02, clip_eps=0.2,
                       normalize="sequence"):
    """PPO-clipped GRPO loss with a k3 KL penalty and no critic."""
    logits = np.asarray(logits, dtype=np.float64)
    logp = log_softmax(logits)
    old_logp = log_softmax(old_logits)
    ref_logp = log_softmax(ref_logits)
    weights = normalization_weights(lengths, normalize)
    grad_logp = np.zeros_like(logits)
    loss = 0.0
    for prompt in range(actions.shape[0]):
        for sample in range(actions.shape[1]):
            action = actions[prompt, sample]
            advantage = advantages[prompt, sample]
            weight = weights[prompt, sample]
            ratio = np.exp(logp[prompt, action] - old_logp[prompt, action])
            clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            use_ratio = ((advantage >= 0.0 and ratio <= 1.0 + clip_eps)
                         or (advantage < 0.0 and ratio >= 1.0 - clip_eps))
            objective = (ratio if use_ratio else clipped) * advantage
            kl = k3_kl_to_reference(logp[prompt, action], ref_logp[prompt, action])
            loss += weight * (-objective + beta_kl * kl)
            if use_ratio:
                grad_logp[prompt, action] -= weight * advantage * ratio
            grad_logp[prompt, action] += weight * beta_kl * (
                1.0 - np.exp(ref_logp[prompt, action] - logp[prompt, action]))
    probs = np.exp(logp)
    grad = grad_logp - probs * np.sum(grad_logp, axis=1, keepdims=True)
    return float(loss), grad
# end::grpo-loss[]


# tag::toy-run[]
def toy_grpo_run(steps=80, lr=0.4):
    """Train two arithmetic prompts over three candidate answers each."""
    prompts = [(1, 2), (2, 3)]
    answers = np.array([["3", "4", "2"], ["5", "4", "6"]])
    actions = np.tile(np.arange(3), (2, 1))
    rewards = np.array([[arithmetic_reward(p, a) for a in row]
                        for p, row in zip(prompts, answers)])
    advantages = group_advantages(rewards)
    lengths = np.vectorize(len)(answers)
    logits = np.zeros((2, 3), dtype=np.float64)
    ref_logits = np.zeros_like(logits)
    for _ in range(steps):
        old_logits = logits.copy()
        _, grad = grpo_loss_and_grad(logits, old_logits, ref_logits, actions,
                                     advantages, lengths, beta_kl=0.03)
        logits -= lr * grad
    return np.exp(log_softmax(logits)), rewards
# end::toy-run[]
