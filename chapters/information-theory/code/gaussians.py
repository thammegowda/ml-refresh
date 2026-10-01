"""KL divergence between Gaussians (Chapter 7)."""
import numpy as np


# tag::gaussian-kl[]
def gaussian_kl(mean_p, std_p, mean_q, std_q):
    """KL(N(mean_p, std_p^2) || N(mean_q, std_q^2)) in closed form."""
    return (np.log(std_q / std_p)
            + (std_p ** 2 + (mean_p - mean_q) ** 2) / (2 * std_q ** 2) - 0.5)
# end::gaussian-kl[]
