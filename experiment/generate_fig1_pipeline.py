"""Minimal publication-quality pipeline figure (Springer-style)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "manuscript" / "figures" / "fig1_pipeline.png"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
    }
)

STEPS = ["Raw Text", "Cleaning", "TF-IDF", "Classifier", "Metrics"]

fig, ax = plt.subplots(figsize=(9.5, 1.85))
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# Single accent: navy outline, white fill
EDGE = "#1a365d"
TEXT = "#1a202c"
ARROW = "#4a5568"

box_w, box_h, y = 1.42, 0.42, 0.42
gap = 0.55
total_w = len(STEPS) * box_w + (len(STEPS) - 1) * gap
start_x = (10 - total_w) / 2
centers = []

for i, step in enumerate(STEPS):
    x = start_x + i * (box_w + gap)
    patch = FancyBboxPatch(
        (x, y - box_h / 2),
        box_w,
        box_h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=1.25,
        edgecolor=EDGE,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(patch)
    cx = x + box_w / 2
    centers.append(cx)
    ax.text(
        cx,
        y,
        step,
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT,
        zorder=3,
    )

for i in range(len(centers) - 1):
    ax.add_patch(
        FancyArrowPatch(
            (centers[i] + box_w / 2 + 0.05, y),
            (centers[i + 1] - box_w / 2 - 0.05, y),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=ARROW,
            zorder=1,
        )
    )

ax.text(
    5.0,
    0.82,
    "Experimental Pipeline",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=TEXT,
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
print(f"Saved {OUT}")
