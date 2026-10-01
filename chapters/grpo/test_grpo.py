import numpy as np

from scratch.gradcheck import check_gradient
from scratch.grpo import (arithmetic_reward, grpo_loss_and_grad, group_advantages,
                          k3_kl_to_reference, log_softmax, normalization_weights,
                          rloo_advantages, toy_grpo_run)
from scratch.information import kl_divergence
from solutions import (centered_group_example, exact_k3_kl, example_weights,
                       toy_correct_probabilities)


def test_arithmetic_checker_and_group_relative_advantages():
    assert arithmetic_reward((2, 5), "7") == 1.0
    assert arithmetic_reward((2, 5), "8") == 0.0
    assert arithmetic_reward((2, 5), "seven") == 0.0
    rewards = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    advantages = group_advantages(rewards)
    np.testing.assert_allclose(advantages[0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(advantages[0].std(), 1.0, atol=3e-8)
    np.testing.assert_allclose(advantages[1], 0.0)
    np.testing.assert_allclose(centered_group_example(), advantages[:1][0])


def test_rloo_is_a_leave_one_out_baseline():
    rewards = np.array([[1.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    expected = np.array([[1.0, -0.5, -0.5], [1.0, 0.25, -1.25]])
    np.testing.assert_allclose(rloo_advantages(rewards), expected)


def test_k3_matches_exact_reverse_kl_in_expectation():
    policy = np.array([0.2, 0.5, 0.3])
    reference = np.array([0.4, 0.4, 0.2])
    logp = np.log(policy)
    logref = np.log(reference)
    estimate = np.sum(policy * k3_kl_to_reference(logp, logref))
    np.testing.assert_allclose(estimate, kl_divergence(policy, reference))
    np.testing.assert_allclose(exact_k3_kl(policy, reference), estimate)
    assert np.all(k3_kl_to_reference(logp, logref) >= 0.0)


def test_sequence_and_token_normalization_weights_differ():
    lengths = np.array([[2.0, 6.0]])
    sequence, token = example_weights()
    np.testing.assert_allclose(sequence, [[0.5, 0.5]])
    np.testing.assert_allclose(token, [[0.25, 0.75]])
    np.testing.assert_allclose(normalization_weights(lengths, "sequence"), sequence)
    np.testing.assert_allclose(normalization_weights(lengths, "token"), token)


def test_grpo_loss_gradient_matches_finite_differences():
    logits = np.array([[0.2, -0.1, 0.0], [-0.3, 0.2, 0.1]], dtype=np.float64)
    old_logits = logits + np.array([[0.01, -0.02, 0.0], [0.0, 0.02, -0.01]])
    ref_logits = np.zeros_like(logits)
    actions = np.array([[0, 1, 2], [0, 1, 2]])
    rewards = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    advantages = group_advantages(rewards)
    lengths = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    loss, grad = grpo_loss_and_grad(logits, old_logits, ref_logits, actions,
                                    advantages, lengths, beta_kl=0.03)
    assert loss < 0.1
    check_gradient(lambda z: grpo_loss_and_grad(z.reshape(logits.shape), old_logits,
                                                ref_logits, actions, advantages,
                                                lengths, 0.03)[0],
                   logits.ravel(), grad.ravel(), tolerance=3e-7)


def test_toy_grpo_improves_verifiable_answers():
    policy, rewards = toy_grpo_run()
    correct = rewards.astype(bool)
    assert np.all(policy[correct] > 0.7)
    assert np.all(toy_correct_probabilities() > 0.7)
    assert np.all(policy[~correct].reshape(2, 2) < 0.16)
