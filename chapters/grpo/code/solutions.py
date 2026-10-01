"""Solutions for Chapter 37 exercises."""
import numpy as np

from scratch.grpo import (group_advantages, k3_kl_to_reference, normalization_weights,
                          rloo_advantages, toy_grpo_run)


# tag::advantages[]
def centered_group_example():
    rewards = np.array([[1.0, 0.0, 0.0]])
    return group_advantages(rewards)[0]
# end::advantages[]


# tag::k3[]
def exact_k3_kl(policy, reference):
    logp = np.log(policy)
    logref = np.log(reference)
    return float(np.sum(policy * k3_kl_to_reference(logp, logref)))
# end::k3[]


# tag::normalization[]
def example_weights():
    lengths = np.array([[2.0, 6.0]])
    sequence = normalization_weights(lengths, "sequence")
    token = normalization_weights(lengths, "token")
    return sequence, token
# end::normalization[]


# tag::toy[]
def toy_correct_probabilities():
    policy, rewards = toy_grpo_run()
    correct = rewards.astype(bool)
    return policy[correct]
# end::toy[]
