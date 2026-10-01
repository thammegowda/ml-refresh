"""Joint, marginal, and conditional distributions, and Bayes' rule (Chapter 6)."""


# tag::joint[]
def marginals(joint):
    """joint[i, j] = P(X = i, Y = j) -> (P(X = i) for each i, P(Y = j) for each j)."""
    return joint.sum(axis=1), joint.sum(axis=0)


def conditional_y_given_x(joint):
    """Row i holds P(Y = j | X = i): each row of the joint, renormalized."""
    return joint / joint.sum(axis=1, keepdims=True)
# end::joint[]


# tag::bayes[]
def posterior(prior, likelihood):
    """P(H = h | evidence) from priors P(H = h) and likelihoods P(evidence | H = h)."""
    unnormalized = prior * likelihood       # P(H = h, evidence)
    return unnormalized / unnormalized.sum()  # divide by P(evidence)
# end::bayes[]
