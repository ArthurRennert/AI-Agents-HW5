"""Run the baseline experiment — direct model loading without AirLLM.

Expected outcome: OOM / crash. Evidence is saved to results/baseline_failure/.

Run from the project root:
    python experiments/run_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_settings
from src.runners.baseline_runner import run_baseline

RESULTS_DIR = "results"


def main() -> None:
    settings = load_settings()

    print(f"[baseline] Model      : {settings.model_name}")
    print(f"[baseline] Quant level: {settings.quant_level}")
    print(f"[baseline] Attempting direct load — expect OOM ...")

    record = run_baseline(settings, results_dir=RESULTS_DIR)

    print("\n[baseline] Run complete.")
    print(f"  Engine    : {record.scenario.engine}")
    print(f"  Output    : {record.output.generated_text[:120]!r}")
    print(f"  TTFT      : {record.metrics.ttft_ms:.1f} ms")
    print(f"  Peak RAM  : {record.metrics.peak_ram_gb:.2f} GB")
    print(f"  Peak VRAM : {record.metrics.peak_vram_gb:.2f} GB")
    print(f"  Wall clock: {record.metrics.wall_clock_sec:.2f} s")

    failed = record.output.generated_text.startswith("ERROR:")
    phys_ram_gb, vram_gb = _physical_ram_gb(), _vram_gb()
    peak_ram = record.metrics.peak_ram_gb
    peak_vram = record.metrics.peak_vram_gb
    ttft_s = record.metrics.ttft_ms / 1000.0

    if failed:
        failure_dir = Path(RESULTS_DIR) / "baseline_failure"
        print(f"\n[baseline] Direct load FAILED (CUDA/OOM) — bottleneck hit.")
        print(f"[baseline] Failure evidence: {failure_dir}/error.txt")
    elif vram_gb and peak_vram > vram_gb:
        print(f"\n[baseline] VRAM BOTTLENECK: peak VRAM {peak_vram:.1f} GB exceeds the "
              f"{vram_gb:.1f} GB on the GPU — overflow spilled to shared system memory.")
        print(f"[baseline] TTFT {ttft_s:.0f}s is non-viable: the model only 'runs' via "
              f"driver memory fallback. AirLLM (bounded VRAM) is required.")
    elif peak_ram > phys_ram_gb:
        print(f"\n[baseline] MEMORY BOTTLENECK (swap thrash): peak {peak_ram:.1f} GB "
              f"exceeds physical {phys_ram_gb:.1f} GB — overflow paged to disk.")
        print(f"[baseline] TTFT {ttft_s:.0f}s is non-viable. AirLLM is required.")
    else:
        print(f"\n[baseline] Direct load fit (peak VRAM {peak_vram:.1f} GB / "
              f"RAM {peak_ram:.1f} GB). Pick a larger model to show the bottleneck.")


def _physical_ram_gb() -> float:
    return _hw_field("ram_gb", 34.0)


def _vram_gb() -> float:
    return _hw_field("vram_gb", 24.0)


def _hw_field(field: str, default: float) -> float:
    try:
        import json

        return float(json.loads(
            (Path(RESULTS_DIR) / "hardware.json").read_text())[field])
    except Exception:
        return default


if __name__ == "__main__":
    main()
