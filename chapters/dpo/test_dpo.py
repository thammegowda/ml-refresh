import numpy as np

from scratch.dpo import (dpo_loss_and_grad, fit_toy_dpo, kl_regularized_optimum,
                         log_softmax, sigmoid, soft_dpo_loss_and_grad)
from scratch.gradcheck import check_gradient
from scratch.information import kl_divergence
from solutions import dpo_delta, preference_weight, toy_distance


def test_gibbs_form_rewrites_the_objective_and_finds_the_optimum():
    reference = np.array([0.5, 0.3, 0.2])
    reward = np.array([0.0, 0.7, -0.4])
    beta = 0.6
    optimum = kl_regularized_optimum(reference, reward, beta)
    z = np.sum(reference * np.exp(reward / beta))
    for policy in [reference, np.array([0.2, 0.7, 0.1]), optimum]:
        objective = np.sum(policy * reward) - beta * kl_divergence(policy, reference)
        gap_form = beta * np.log(z) - beta * kl_divergence(policy, optimum)
        np.testing.assert_allclose(objective, gap_form)
    values = [np.sum(p * reward) - beta * kl_divergence(p, reference)
              for p in [reference, np.array([0.2, 0.7, 0.1]), optimum]]
    assert values[-1] == max(values)


def test_implicit_reward_and_bradley_terry_cancel_the_partition_function():
    reference = np.array([0.5, 0.3, 0.2])
    reward = np.array([0.0, 0.7, -0.4])
    beta = 0.6
    policy = kl_regularized_optimum(reference, reward, beta)
    z = np.sum(reference * np.exp(reward / beta))
    implicit = beta * (np.log(policy) - np.log(reference)) + beta * np.log(z)
    np.testing.assert_allclose(implicit, reward, atol=1e-12)
    logp, logref = np.log(policy), np.log(reference)
    np.testing.assert_allclose(dpo_delta(logp[1], logp[2], logref[1], logref[2], beta),
                               reward[1] - reward[2])


def test_dpo_gradient_matches_finite_differences_and_weight_is_loser_probability():
    logits = np.array([0.2, -0.4, 0.7], dtype=np.float64)
    reference = np.array([0.45, 0.35, 0.20], dtype=np.float64)
    pairs = np.array([[2, 1], [0, 1], [2, 0]])
    loss, grad = dpo_loss_and_grad(logits, reference, pairs, beta=0.8)
    assert loss > 0
    check_gradient(lambda z: dpo_loss_and_grad(z, reference, pairs, 0.8)[0],
                   logits, grad)
    logp, logref = log_softmax(logits), np.log(reference)
    delta = 0.8 * ((logp[2] - logref[2]) - (logp[1] - logref[1]))
    np.testing.assert_allclose(preference_weight(delta), 1.0 - sigmoid(delta))
    assert preference_weight(-3.0) > preference_weight(3.0)


def test_soft_dpo_gradient_and_toy_run_match_the_closed_form_optimum():
    reference = np.array([0.50, 0.30, 0.20])
    rewards = np.array([0.0, 0.7, -0.4])
    beta = 0.6
    logits = np.array([-0.2, 0.1, 0.3], dtype=np.float64)
    loss, grad = soft_dpo_loss_and_grad(logits, reference, rewards, beta)
    assert loss > 0
    check_gradient(lambda z: soft_dpo_loss_and_grad(z, reference, rewards, beta)[0],
                   logits, grad)
    policy, optimum = fit_toy_dpo(reference, rewards, beta=beta)
    np.testing.assert_allclose(policy, optimum, atol=1e-3)
    assert toy_distance() < 1e-3
