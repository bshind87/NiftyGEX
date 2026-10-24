"""Shared figure style for report scripts: validated light palette, recessive grid, thin marks, PNG output."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots in fixed order (validated adjacent-pair CVD and normal-vision floors on the light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 1.6, "figure.dpi": 110,
})


def save(fig, directory: Path, name: str) -> Path:
    """Write ``name``.png under ``directory`` and close the figure."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def label_line_end(ax, x, y, text: str) -> None:
    """Direct label just right of a line's last point, in secondary text ink."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
