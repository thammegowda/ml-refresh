"""Build-time figures for Chapter 11."""
import sys

import matplotlib.pyplot as plt
import numpy as np

import figstyle

ROOT = __file__.rsplit("/chapters/", 1)[0]
sys.path.insert(0, ROOT)
from scratch.activations import (gelu_exact, gelu_exact_grad, relu, relu_grad,
                                 sigmoid, sigmoid_grad, silu, silu_grad,
                                 softplus, softplus_grad)

figstyle.setup()
output = sys.argv[1]
x = np.linspace(-5, 5, 600)
curves = [
    ("sigmoid", sigmoid, sigmoid_grad),
    ("ReLU", relu, relu_grad),
    ("GELU", gelu_exact, gelu_exact_grad),
    ("SiLU", silu, silu_grad),
    ("softplus", softplus, softplus_grad),
]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharex=True,
                         layout="constrained")
for index, (name, forward, grad) in enumerate(curves):
    color = figstyle.COLORS[index % len(figstyle.COLORS)]
    axes[0].plot(x, forward(x), label=name, color=color)
    axes[1].plot(x, grad(x), label=name, color=color)
for ax, title in zip(axes, ["activation", "derivative"]):
    ax.axhline(0, color="#64716d", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="#64716d", linewidth=0.7, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("x")
axes[0].set_ylim(-0.4, 5.1)
axes[1].set_ylim(-0.2, 1.3)
axes[0].set_ylabel("value")
axes[1].legend(loc="upper left", fontsize=8)
figstyle.save(fig, output, "activations-derivatives")
