"""Build-time figures for Chapter 6 (deterministic SVG)."""
import sys

import numpy as np
import matplotlib.pyplot as plt

import figstyle

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/code")
from distributions import gaussian_pdf  # noqa: E402

figstyle.setup()
output = sys.argv[1]
rng = np.random.default_rng(6)

# Samples against the density; the shaded area is P(-1 < X < 1).
x = rng.standard_normal(10_000)
grid = np.linspace(-4, 4, 400)
fig, ax = plt.subplots()
ax.hist(x, bins=60, range=(-4, 4), density=True, color="#dce3df", edgecolor="#c8d4cd", label="10,000 samples (histogram)")
ax.plot(grid, gaussian_pdf(grid, 0, 1), color=figstyle.COLORS[0], label="density of N(0, 1)")
inside = (grid > -1) & (grid < 1)
ax.fill_between(grid[inside], gaussian_pdf(grid[inside], 0, 1), color=figstyle.COLORS[0], alpha=0.2, label="area 0.683 = P(-1 < X < 1)")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend(loc="upper left")
figstyle.save(fig, output, "gaussian-samples")

# A Monte Carlo estimate converging, with a band of two standard errors.
values = rng.standard_normal(10_000) ** 2
n = np.arange(1, len(values) + 1)
running = np.cumsum(values) / n
spread = np.sqrt(np.maximum(np.cumsum(values ** 2) / n - running ** 2, 0) / n)
fig, ax = plt.subplots()
ax.axhline(1.0, color="#64716d", linestyle=":", linewidth=1, label="true value E[X²] = 1")
ax.fill_between(n[9:], (running - 2 * spread)[9:], (running + 2 * spread)[9:], color=figstyle.COLORS[1], alpha=0.18, label="estimate ± 2 standard errors")
ax.plot(n[9:], running[9:], color=figstyle.COLORS[1], label="running average")
ax.set_xscale("log")
ax.set_ylim(0, 2.5)
ax.set_xlabel("number of samples n")
ax.set_ylabel("estimate of E[X²], X ~ N(0, 1)")
ax.legend(loc="upper right")
figstyle.save(fig, output, "monte-carlo")

# The central limit theorem: means of n uniforms look more Gaussian as n grows.
fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.1), sharey=True, layout="constrained")
for ax, count in zip(axes, [1, 2, 4, 16]):
    means = rng.random((50_000, count)).mean(axis=1)
    z = (means - 0.5) / np.sqrt(1 / (12 * count))
    ax.hist(z, bins=50, range=(-3.5, 3.5), density=True, color="#dce3df", edgecolor="#c8d4cd")
    ax.plot(grid, gaussian_pdf(grid, 0, 1), color=figstyle.COLORS[0], linewidth=1.4)
    ax.set_title(f"n = {count}", fontsize=9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_xticks([-3, 0, 3])
axes[0].set_ylabel("density")
fig.supxlabel("standardized mean of n Uniform(0, 1) draws", fontsize=9)
figstyle.save(fig, output, "central-limit")
