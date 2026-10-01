"""Deterministic train/validation/test splits for Chapter 9."""
import numpy as np


# tag::split[]
def train_validation_test_split(n, train=0.6, validation=0.2, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    order = rng.permutation(n)
    n_train = int(round(train * n))
    n_validation = int(round(validation * n))
    train_idx = order[:n_train]
    validation_idx = order[n_train:n_train + n_validation]
    test_idx = order[n_train + n_validation:]
    return train_idx, validation_idx, test_idx
# end::split[]
