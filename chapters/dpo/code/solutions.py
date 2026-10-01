"""Solutions for Chapter 36 exercises."""
import numpy as np

from scratch.dpo import dpo_loss_and_grad, fit_toy_dpo, kl_regularized_optimum, sigmoid


# tag::z-cancels[]
def dpo_delta(logp_w, logp_l, logref_w, logref_l, beta):
    return beta * ((logp_w - logref_w) - (logp_l - logref_l))
# end::z-cancels[]


# tag::gradient-weight[]
def preference_weight(delta):
    return sigmoid(-delta)
# end::gradient-weight[]


# tag::toy-check[]
def toy_distance():
    reference = np.array([0.50, 0.30, 0.20])
    rewards = np.array([0.0, 0.7, -0.4])
    policy, optimum = fit_toy_dpo(reference, rewards, beta=0.6)
    return float(np.max(np.abs(policy - optimum)))
# end::toy-check[]
