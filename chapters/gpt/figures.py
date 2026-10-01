"""Build-time figure for Chapter 23 (deterministic SVG)."""
import sys

import matplotlib.pyplot as plt
import numpy as np

import figstyle

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/code")
from gpt_train import train_tiny_gpt  # noqa: E402

figstyle.setup()
output = sys.argv[1]

_params, history, _stoi, _itos = train_tiny_gpt(
    steps=35,
    seed=23,
    d_model=24,
    n_heads=2,
    n_layers=1,
    hidden_dim=48,
    batch_size=8,
    seq_len=16,
    base_lr=3e-3,
)

fig, ax = plt.subplots()
steps = np.arange(1, len(history["train"]) + 1)
ax.plot(steps, history["train"], label="train batch loss", color=figstyle.COLORS[0])
ax.plot(history["val"][:, 0], history["val"][:, 1], marker="o",
        label="validation loss", color=figstyle.COLORS[1])
ax.set_xlabel("optimization step")
ax.set_ylabel("cross-entropy (nats/char)")
ax.set_title("A tiny character-level GPT: training and validation loss")
ax.legend(loc="upper right")
figstyle.save(fig, output, "tiny-gpt-loss")
