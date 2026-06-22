"""Figure for the size sweep: the RAM wall vs AirLLM's disk-bound latency cost."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scaling.size_sweep import SizeResult

_FITS_COLOR = "#55A868"
_OOM_COLOR = "#C44E52"


def plot_size_comparison(
    results: list[SizeResult], ram_gb: float, out_dir: str = "figures"
) -> Path:
    """Plot baseline RAM requirement vs the RAM ceiling, with AirLLM TPOT overlaid."""
    if not results:
        raise ValueError("no size-sweep results to plot")

    labels = [r.name.replace("Qwen2.5-", "").replace("-Instruct", "") for r in results]
    required = [r.baseline_required_gb for r in results]
    airllm_ram = [r.airllm_peak_ram_gb for r in results]
    tpot = [r.est_tpot_s for r in results]
    colors = [_FITS_COLOR if r.fits_baseline else _OOM_COLOR for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(results))

    ax.bar(
        x, required, color=colors, edgecolor="black", linewidth=0.6,
        label="Baseline RAM required (FP16)",
    )
    ax.plot(
        x, airllm_ram, "o-", color="#4C72B0", linewidth=2, markersize=7,
        label="AirLLM peak RAM (bounded)",
    )
    ax.axhline(
        ram_gb, color="black", linestyle="--", linewidth=1.6,
        label=f"Physical RAM ceiling ({ram_gb:.1f} GB)",
    )
    top = max(required) * 1.08
    ax.fill_between([-0.5, len(results) - 0.5], ram_gb, top, color="red", alpha=0.06)
    ax.text(
        len(results) - 0.55, ram_gb + (top - ram_gb) * 0.5, "OOM region",
        color=_OOM_COLOR, ha="right", va="center", fontsize=9, fontweight="bold",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Memory (GB)")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.5, len(results) - 0.5)

    ax2 = ax.twinx()
    ax2.plot(
        x, tpot, "s--", color="#8172B3", linewidth=2, markersize=6,
        label="AirLLM est. TPOT (s/token)",
    )
    ax2.set_ylabel("AirLLM latency  —  TPOT (s/token)", color="#8172B3")
    ax2.tick_params(axis="y", labelcolor="#8172B3")
    ax2.set_ylim(0, max(tpot) * 1.25)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    ax.set_title(
        "Extension A — Bottleneck shifts with model size\n"
        "Direct execution hits the RAM wall; AirLLM trades it for disk-bound latency",
        fontsize=11, fontweight="bold",
    )

    out = Path(out_dir) / "extension_size_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {out}")
    return out
