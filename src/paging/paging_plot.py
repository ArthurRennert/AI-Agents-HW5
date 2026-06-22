"""Figure for the paging demo: the RSS sawtooth and fault/latency correlation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_paging(trace: dict, out_dir: str = "figures") -> Path:
    """Two stacked panels: RSS-over-time sawtooth and per-layer faults vs load time."""
    samples = trace.get("samples", [])
    if not samples:
        raise ValueError("no paging samples to plot")

    t = [s["t"] for s in samples]
    rss = [s["rss_gb"] for s in samples]
    faults = trace["per_layer_faults"]
    load_sec = trace["per_layer_load_sec"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7))

    # --- Panel 1: RSS sawtooth ------------------------------------------------
    ax_top.plot(t, rss, color="#4C72B0", linewidth=1.3)
    ax_top.fill_between(t, rss, color="#4C72B0", alpha=0.12)
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Resident memory — RSS (GB)")
    ax_top.set_title(
        "Extension B — AirLLM is OS paging\n"
        "Each layer pages in (RSS rises), computes, then pages out (RSS falls)",
        fontsize=11, fontweight="bold",
    )
    ax_top.grid(True, alpha=0.3)

    # --- Panel 2: per-layer faults (bars) vs load time (line) -----------------
    layers = list(range(len(faults)))
    ax_bot.bar(
        layers, faults, color="#C44E52", edgecolor="black", linewidth=0.4,
        label="Page faults per layer",
    )
    ax_bot.set_xlabel("Layer index (paged in order)")
    ax_bot.set_ylabel("Page faults", color="#C44E52")
    ax_bot.tick_params(axis="y", labelcolor="#C44E52")

    ax2 = ax_bot.twinx()
    ax2.plot(
        layers, load_sec, "o-", color="#8172B3", linewidth=1.8, markersize=4,
        label="Per-layer load+compute time (s)",
    )
    ax2.set_ylabel("Per-layer time (s)", color="#8172B3")
    ax2.tick_params(axis="y", labelcolor="#8172B3")

    total = trace.get("total_page_faults", 0)
    ax_bot.set_title(
        f"Paging cost per layer  (total page faults this run: {total:,})",
        fontsize=10,
    )
    lines1, lab1 = ax_bot.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")

    out = Path(out_dir) / "extension_paging.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {out}")
    return out
