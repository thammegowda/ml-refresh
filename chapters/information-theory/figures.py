"""Build-time figures for Chapter 7 (deterministic SVG)."""
import sys

import numpy as np
import matplotlib.pyplot as plt

import figstyle

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/code")
from fitting import GRID, fit_gaussian, gaussian_density, two_modes  # noqa: E402

figstyle.setup()
output = sys.argv[1]

# Surprisal of one outcome, and the entropy of a coin.
p = np.linspace(0.001, 0.999, 500)
fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained")
left.plot(p, -np.log2(p), color=figstyle.COLORS[0])
left.set_xlabel("probability of the outcome")
left.set_ylabel("surprisal (bits)")
left.set_ylim(0, 8)
right.plot(p, -(p * np.log2(p) + (1 - p) * np.log2(1 - p)), color=figstyle.COLORS[1])
right.set_xlabel("probability of heads")
right.set_ylabel("entropy of the coin (bits)")
right.set_ylim(0, 1.05)
figstyle.save(fig, output, "surprisal-entropy")

# One Gaussian fitted to two modes, by forward and by reverse KL.
means, stds = np.arange(-4, 4.001, 0.05), np.arange(0.2, 4.001, 0.02)
forward = fit_gaussian(two_modes, "forward", means, stds)
reverse = fit_gaussian(two_modes, "reverse", means, stds)
fig, ax = plt.subplots()
ax.fill_between(GRID, two_modes(GRID), color="#dce3df", label="target p (two modes)")
ax.plot(GRID, gaussian_density(GRID, *forward), color=figstyle.COLORS[0], label=f"argmin KL(p || q): mean {round(forward[0], 1) + 0.0:.1f}, std {forward[1]:.2f}")
ax.plot(GRID, gaussian_density(GRID, *reverse), "--", color=figstyle.COLORS[1], label=f"argmin KL(q || p): mean {round(reverse[0], 1) + 0.0:.1f}, std {reverse[1]:.2f}")
ax.set_xlim(-6, 6)
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend(loc="upper left")
figstyle.save(fig, output, "forward-reverse-kl")
