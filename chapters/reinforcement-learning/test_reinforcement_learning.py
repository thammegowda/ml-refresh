import numpy as np

from scratch.gradcheck import check_gradient
from scratch.reinforcement_learning import (
    bandit_objective,
    bandit_policy_gradient,
    discounted_returns,
    evaluate_policy,
    gae_explicit,
    gae_recursive,
    importance_ratios,
    off_policy_value,
    reinforce_bandit,
    tiny_chain,
)


def rng(seed=34):
    return np.random.default_rng(seed)


def test_returns_and_bellman_policy_evaluation_on_tiny_chain():
    np.testing.assert_allclose(discounted_returns([0, 0, 1], gamma=0.9), [0.81, 0.9, 1])
    values = tiny_chain(gamma=0.9)
    np.testing.assert_allclose(values, [0.9, 1.0, 0.0])

    transition = np.array([[0.5, 0.5], [0.0, 1.0]])
    reward = np.array([1.0, 2.0])
    values = evaluate_policy(transition, reward, gamma=0.5)
    residual = reward + 0.5 * transition @ values - values
    np.testing.assert_allclose(residual, np.zeros(2), atol=1e-12)


def test_bandit_policy_gradient_is_gradient_checked():
    r = rng()
    logits = r.normal(size=4)
    action_values = np.array([-0.2, 0.5, 1.3, 0.1])
    analytic = bandit_policy_gradient(logits, action_values)
    objective = lambda z: bandit_objective(z, action_values)
    assert check_gradient(objective, logits, analytic, tolerance=1e-8) < 1e-8
    np.testing.assert_allclose(np.sum(analytic), 0.0, atol=1e-12)


def test_reinforce_bandit_moves_probability_to_best_action():
    action_values = np.array([0.0, 1.0, -0.4])
    logits, probabilities = reinforce_bandit(
        action_values,
        steps=240,
        batch_size=96,
        lr=0.35,
        seed=9,
    )
    assert np.argmax(logits) == 1
    assert probabilities[1] > 0.94
    assert bandit_objective(logits, action_values) > 0.92


def test_importance_sampling_estimates_target_policy_from_behavior_data():
    behavior = np.array([0.8, 0.2])
    target = np.array([0.25, 0.75])
    actions = np.array([0] * 80 + [1] * 20)
    rewards = np.where(actions == 0, 1.0, 3.0)
    ratios = importance_ratios(actions, target, behavior)
    np.testing.assert_allclose(ratios[:80], 0.25 / 0.8)
    np.testing.assert_allclose(ratios[80:], 0.75 / 0.2)
    estimate = off_policy_value(actions, rewards, target, behavior)
    np.testing.assert_allclose(estimate, 2.5)


def test_gae_recursive_matches_the_explicit_lambda_weighted_sum():
    r = rng()
    rewards = r.normal(size=6)
    values = r.normal(size=7)
    for gamma in (0.0, 0.91):
        for lam in (0.0, 0.4, 1.0):
            recursive = gae_recursive(rewards, values, gamma, lam)
            explicit = gae_explicit(rewards, values, gamma, lam)
            np.testing.assert_allclose(recursive, explicit, atol=1e-12)
