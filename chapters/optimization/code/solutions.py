import numpy as np

from scratch.optimization import adam_state, adamw_step, clip_by_global_norm


# tag::condition-rate[]
def optimal_gd_rate(condition_number):
    return (condition_number - 1) / (condition_number + 1)
# end::condition-rate[]


# tag::adam-one-step[]
def adam_first_step():
    params = {"w": np.array([2.0, -3.0])}
    grads = {"w": np.array([0.5, -0.25])}
    state = adam_state(params)
    before = params["w"].copy()
    adamw_step(params, grads, state, lr=0.01)
    return before - params["w"]
# end::adam-one-step[]


# tag::clip-demo[]
def clipped_demo_norm():
    grads = {"a": np.array([3.0, 4.0]), "b": np.array([12.0])}
    clipped, before = clip_by_global_norm(grads, max_norm=5.0)
    after = np.sqrt(sum(np.sum(g * g) for g in clipped.values()))
    return before, float(after)
# end::clip-demo[]
