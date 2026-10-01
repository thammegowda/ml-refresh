"""Shared matplotlib style for build-time book figures (deterministic SVG output)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

COLORS = ["#087f72", "#b04e75", "#a76613", "#4169a1", "#6b5792", "#8b4b32"]


def setup():
    plt.rcParams.update({
        "svg.hashsalt": "refresh",
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=COLORS),
        "axes.grid": True,
        "grid.color": "#dce3df",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "figure.figsize": (6.4, 3.2),
    })


def save(fig, directory, name):
    path = Path(directory) / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None, "Creator": "ML Refresher"})
    plt.close(fig)
    return path
