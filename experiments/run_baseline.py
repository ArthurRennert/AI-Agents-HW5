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

    failure_dir = Path(RESULTS_DIR) / "baseline_failure"
    if failure_dir.exists():
        print(f"\n[baseline] Failure evidence: {failure_dir}/error.txt")
        print("[baseline] BOTTLENECK IDENTIFIED — see error.txt for details.")
    else:
        print("[baseline] Model loaded successfully (unexpected for 32B on 24 GB VRAM).")


if __name__ == "__main__":
    main()
