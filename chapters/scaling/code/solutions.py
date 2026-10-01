import numpy as np

from scaling_laws import chinchilla_loss


# tag::grid-frontier[]
def best_on_grid(compute, parameter_grid, token_grid):
    """Search a small grid for the lowest Chinchilla loss under 6ND <= compute."""
    best = None
    for parameters in parameter_grid:
        for tokens in token_grid:
            if 6 * parameters * tokens > compute:
                continue
            loss = chinchilla_loss(parameters, tokens)
            if best is None or loss < best[0]:
                best = (loss, parameters, tokens)
    return best
# end::grid-frontier[]


# tag::wsd[]
def warmup_stable_decay(step, warmup, stable, total):
    """A simple WSD learning-rate multiplier."""
    if step < warmup:
        return step / warmup
    if step < stable:
        return 1.0
    progress = (step - stable) / max(total - stable, 1)
    return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
# end::wsd[]
