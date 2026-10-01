import numpy as np

from mlp import backward, forward, initialize, synthetic_data, train_sgd


# tag::parameter-count[]
def parameter_count():
    params = initialize(seed=0)
    return sum(value.size for value in params.values())
# end::parameter-count[]


# tag::overfit[]
def memorization_gap(seed=7):
    inputs, labels = synthetic_data(seed=seed, examples_per_class=4)
    rng = np.random.default_rng(seed)
    noisy = rng.permutation(labels)
    params, _ = train_sgd(inputs, noisy, epochs=400, learning_rate=0.12, seed=seed)
    _, cache = forward(params, inputs, noisy)
    train_accuracy = np.mean(cache["P"].argmax(axis=1) == noisy)
    _, clean_cache = forward(params, inputs, labels)
    clean_accuracy = np.mean(clean_cache["P"].argmax(axis=1) == labels)
    return float(train_accuracy), float(clean_accuracy)
# end::overfit[]
