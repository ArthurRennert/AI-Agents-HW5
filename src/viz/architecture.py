"""Pipeline architecture diagram (Task 3.13)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_BOXES = [
    # (center_x, center_y, label, bg_color)
    (0.9,  2.0, ".env /\nSettings",           "#AED6F1"),
    (2.4,  2.0, "HF\nTokenizer",              "#A9DFBF"),
    (4.0,  2.0, "AirLLM\nAutoModel\n(Qwen)", "#F9E79F"),
    (5.6,  2.0, "Measurement\nHarness",       "#D7BDE2"),
    (7.2,  2.0, "Results\nJSON",              "#AEB6BF"),
    (8.8,  2.0, "Figures\n(plots.py)",        "#F1948A"),
    (4.0,  0.7, "NVMe Shards\n(SafeTensors)", "#FAD7A0"),
]

_ARROWS = [
    (0.9,  2.4),
    (2.4,  4.0),
    (4.0,  5.6),
    (5.6,  7.2),
    (7.2,  8.8),
]


def plot_architecture(out_dir: str = "figures") -> None:
    """Generate a pipeline architecture overview diagram."""
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    for cx, cy, label, color in _BOXES:
        rect = mpatches.FancyBboxPatch(
            (cx - 0.65, cy - 0.55), 1.3, 1.1,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#555555", linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.5,
                multialignment="center")

    # horizontal arrows between main pipeline boxes
    for src_x, dst_x in _ARROWS:
        ax.annotate(
            "", xy=(dst_x - 0.65, 2.0), xytext=(src_x + 0.65, 2.0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
        )

    # vertical arrow: AirLLM <-> NVMe Shards
    ax.annotate(
        "", xy=(4.0, 0.7 + 0.55), xytext=(4.0, 2.0 - 0.55),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.1),
    )
    ax.text(4.35, 1.35, "load\nlayer", fontsize=6.5, color="#333333",
            ha="left", va="center")

    ax.set_title(
        "AirLLM Inference Pipeline — Layer-by-Layer Paging Architecture",
        fontsize=11, fontweight="bold", pad=6,
    )

    out = Path(out_dir) / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {out}")
