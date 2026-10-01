import numpy as np

from scratch.gradcheck import check_gradient
from scratch.ppo import (
    bradley_terry_loss_and_grad,
    categorical_entropy,
    categorical_kl,
    make_preference_pairs,
    ppo_clipped_objective_and_grad,
    rlhf_objective,
    toy_ppo_run,
    train_reward_model,
    value_loss_and_grad,
)
from scratch.reinforcement_learning import softmax


def rng(seed=35):
    return np.random.default_rng(seed)


def test_bradley_terry_loss_gradient_is_checked_and_model_trains():
    r = rng()
    winners, losers, true_w = make_preference_pairs(r, n_pairs=120, n_features=4)
    weights = r.normal(size=4) * 0.1
    loss, grad = bradley_terry_loss_and_grad(weights, winners, losers)
    objective = lambda w: bradley_terry_loss_and_grad(w, winners, losers)[0]
    assert loss > 0
    assert check_gradient(objective, weights, grad, tolerance=1e-8) < 1e-8

    learned = train_reward_model(winners, losers, steps=400, lr=0.4)
    margins = (winners - losers) @ learned
    assert np.mean(margins > 0) > 0.95
    assert np.dot(learned, true_w) > 0


def test_bradley_terry_solution_numbers_are_computed():
    winner = np.array([[2.0]])
    loser = np.array([[0.5]])
    weights = np.array([1.0])
    loss, _ = bradley_terry_loss_and_grad(weights, winner, loser)
    probability = 1.0 / (1.0 + np.exp(-1.5))
    np.testing.assert_allclose(probability, 0.8175744761936437)
    np.testing.assert_allclose(loss, 0.2014132779827524)


def test_rlhf_objective_reduces_to_reward_when_policy_matches_reference():
    policy = np.array([0.2, 0.5, 0.3])
    rewards = np.array([0.0, 1.0, 0.4])
    assert categorical_kl(policy, policy) == 0.0
    np.testing.assert_allclose(rlhf_objective(policy, rewards, policy, beta=0.7), 0.62)

    shifted = np.array([0.1, 0.8, 0.1])
    assert categorical_kl(shifted, policy) > 0
    assert rlhf_objective(shifted, rewards, policy, beta=0.7) < shifted @ rewards


def test_ppo_clipped_gradient_is_checked_inside_unclipped_region():
    logits = np.array([0.1, -0.2, 0.05])
    old_logits = np.array([0.0, -0.1, 0.0])
    actions = np.array([0, 1, 2, 0])
    advantages = np.array([1.2, -0.7, 0.4, -0.2])
    objective, grad = ppo_clipped_objective_and_grad(
        logits,
        old_logits,
        actions,
        advantages,
        clip_eps=0.4,
    )
    assert np.isfinite(objective)

    def f(z):
        return ppo_clipped_objective_and_grad(
            z,
            old_logits,
            actions,
            advantages,
            clip_eps=0.4,
        )[0]

    assert check_gradient(f, logits, grad, tolerance=1e-8) < 1e-8


def test_ppo_case_analysis_clips_the_blocked_direction():
    old_logits = np.array([0.0, 0.0])
    high_ratio_logits = np.array([2.0, -2.0])
    low_ratio_logits = np.array([-2.0, 2.0])

    _, grad_pos = ppo_clipped_objective_and_grad(
        high_ratio_logits,
        old_logits,
        actions=np.array([0]),
        advantages=np.array([1.0]),
        clip_eps=0.2,
    )
    _, grad_neg = ppo_clipped_objective_and_grad(
        low_ratio_logits,
        old_logits,
        actions=np.array([0]),
        advantages=np.array([-1.0]),
        clip_eps=0.2,
    )
    np.testing.assert_allclose(grad_pos, [0.0, 0.0])
    np.testing.assert_allclose(grad_neg, [0.0, 0.0])


def test_value_loss_entropy_and_toy_ppo_find_the_best_action():
    loss, grad = value_loss_and_grad(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    assert loss == 1.25
    np.testing.assert_allclose(grad, [0.5, -1.0])
    np.testing.assert_allclose(categorical_entropy([0.5, 0.5]), np.log(2))

    logits, history = toy_ppo_run(np.array([0.0, 1.0, -0.2]), seed=4)
    probabilities = softmax(logits)
    assert np.argmax(probabilities) == 1
    assert probabilities[1] > 0.9
    assert history[-1, 1] > history[0, 1]
