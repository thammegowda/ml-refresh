"""Fitting one Gaussian to two modes by forward and reverse KL (Chapter 7)."""
import numpy as np

GRID = np.linspace(-8, 8, 801)
STEP = GRID[1] - GRID[0]


def gaussian_density(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def two_modes(x):
    """The target p: an equal mixture of N(-2, 0.6^2) and N(2, 0.6^2)."""
    return 0.5 * gaussian_density(x, -2.0, 0.6) + 0.5 * gaussian_density(x, 2.0, 0.6)


# tag::fit[]
def kl_on_grid(p, q):
    """KL(p || q) for densities sampled on GRID: a Riemann sum of p log(p / q)."""
    inside = p > 1e-300
    with np.errstate(divide="ignore"):              # q = 0 where p > 0: KL is infinite
        log_ratio = np.log(p[inside]) - np.log(q[inside])
    return np.sum(p[inside] * log_ratio) * STEP


def fit_gaussian(target, direction, means, stds):
    """Find the Gaussian q minimizing KL(p||q) or KL(q||p)."""
    p = target(GRID)
    best = (np.inf, None, None)
    for mean in means:
        for std in stds:
            q = gaussian_density(GRID, mean, std)
            forward = direction == "forward"
            divergence = kl_on_grid(p, q) if forward else kl_on_grid(q, p)
            best = min(best, (divergence, mean, std), key=lambda item: item[0])
    return best[1], best[2]
# end::fit[]
