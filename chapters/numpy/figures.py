"""Build-time figures for Appendix B (deterministic SVG)."""
import sys

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from scratch.precision import round_to_bfloat16

figstyle.setup()
output = sys.argv[1]


def bfloat16_spacing(x):
    """Distance from x (a bfloat16 value) to the next larger bfloat16 value."""
    x = round_to_bfloat16(np.asarray(x, dtype=np.float32))
    bits = x.view(np.uint32) + np.uint32(1 << 16)
    return bits.view(np.float32) - x


x = np.logspace(-4, 4.8, 400)
fig, ax = plt.subplots()
ax.loglog(x, np.spacing(x.astype(np.float16)).astype(np.float64), label="float16 (10 fraction bits)")
ax.loglog(x, bfloat16_spacing(x).astype(np.float64), label="bfloat16 (7 fraction bits)")
ax.loglog(x, np.spacing(x.astype(np.float32)).astype(np.float64), label="float32 (23 fraction bits)")
ax.axvline(65504, color="#64716d", linestyle=":", linewidth=1)
ax.annotate("float16 max 65504", (65504, 1e-9), xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8, color="#64716d")
ax.set_xlabel("magnitude of x")
ax.set_ylabel("gap to the next representable number")
ax.legend(loc="upper left")
figstyle.save(fig, output, "float-spacing")

# Finite-difference error for d/dx sin(x) at x = 1: truncation falls with h, round-off grows as 1/h.
h = np.logspace(-15, -1, 300)
exact = np.cos(1.0)
fig, ax = plt.subplots()
for dtype, label, style in [(np.float64, "central, float64", "-"), (np.float64, "forward, float64", "--"), (np.float32, "central, float32", ":")]:
    x0, step = dtype(1.0), h.astype(dtype)
    if label.startswith("central"):
        estimate = (np.sin(x0 + step) - np.sin(x0 - step)) / (2 * step)
    else:
        estimate = (np.sin(x0 + step) - np.sin(x0)) / step
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(estimate.astype(np.float64) - exact)
    ax.loglog(h, np.maximum(error, 1e-17), style, label=label)
ax.set_ylim(1e-12, 1e1)
ax.set_xlabel("step size h")
ax.set_ylabel("|estimate - exact derivative|")
ax.legend(loc="lower left")
figstyle.save(fig, output, "finite-difference-error")
