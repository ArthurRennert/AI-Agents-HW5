"""Model Roofline diagram for RTX 3090 (Task 3.14 - P2 extension)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# RTX 3090 hardware specs
_PEAK_TFLOPS_FP16 = 35.6        # teraFLOPS (FP16 Tensor Core)
_MEM_BW_GB_S = 936.0            # GB/s HBM bandwidth
_RIDGE_POINT = _PEAK_TFLOPS_FP16 * 1e12 / (_MEM_BW_GB_S * 1e9)  # ~38 FLOPs/byte

# Operating points on the roofline
_OPERATING_POINTS = [
    # (arithmetic_intensity, label, color, marker)
    (1.0,   "Prefill\n(GEMM, seq>1)",   "#4C72B0", "o"),   # compute moves toward ridge
    (0.08,  "Decode\n(GEMV, seq=1)",    "#C44E52", "s"),   # deeply memory-bound
]


def plot_roofline(out_dir: str = "figures") -> None:
    """Roofline model: shows Prefill (compute-bound) vs Decode (memory-bound)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ai = np.logspace(-2, 3, 500)
    memory_roof = _MEM_BW_GB_S * ai          # GB/s * FLOPs/byte = GFLOPs/s, then /1e3 -> TFLOPS
    compute_roof = _PEAK_TFLOPS_FP16 * 1e3 * np.ones_like(ai)  # GFLOPs/s
    attainable = np.minimum(memory_roof * 1e9 / 1e9, compute_roof)  # keep in GFLOPs/s
    # Recompute cleanly in GFLOPs/s
    mem_roof_gflops = _MEM_BW_GB_S * ai       # GFLOPs/s (GB/s * FLOPs/byte)
    peak_gflops = _PEAK_TFLOPS_FP16 * 1000    # 35600 GFLOPs/s
    attainable_gflops = np.minimum(mem_roof_gflops, peak_gflops)

    ax.loglog(ai, attainable_gflops, "k-", linewidth=2, label="Attainable performance")
    ax.axvline(_RIDGE_POINT, color="gray", linestyle=":", linewidth=1, label=f"Ridge point (~{_RIDGE_POINT:.0f} FLOPs/byte)")
    ax.axhline(peak_gflops, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Shade memory-bound and compute-bound regions
    ax.fill_betweenx([0.01, peak_gflops], 0.01, _RIDGE_POINT, alpha=0.06, color="red", label="Memory-bound region")
    ax.fill_betweenx([0.01, peak_gflops], _RIDGE_POINT, 1000, alpha=0.06, color="blue", label="Compute-bound region")

    for ai_pt, label, color, marker in _OPERATING_POINTS:
        perf = min(_MEM_BW_GB_S * ai_pt, peak_gflops)
        ax.scatter([ai_pt], [perf], color=color, marker=marker, s=120, zorder=5, edgecolors="black")
        ax.annotate(label, (ai_pt, perf), textcoords="offset points",
                    xytext=(10, 5), fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=10)
    ax.set_ylabel("Attainable Performance (GFLOPs/s)", fontsize=10)
    ax.set_title("Roofline Model — RTX 3090 (FP16)\nPrefill vs Decode Operating Points",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(10, peak_gflops * 3)
    ax.grid(True, which="both", alpha=0.3)

    ax.text(0.013, 1200, "Memory-bound\n(GEMV / Decode)", fontsize=8, color="red", alpha=0.7)
    ax.text(150, 1200, "Compute-bound\n(GEMM / Prefill)", fontsize=8, color="blue", alpha=0.7)

    out = Path(out_dir) / "roofline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {out}")
