"""Quantization sweep experiment: Q8, Q4, Q2 via AirLLM (Tasks 3.4–3.7).

Runs Qwen2.5-32B-Instruct at each quantization level and records metrics.
Each level writes results/airllm_<level>.json.

Requires: HF_TOKEN and SHARD_PATH in .env; CUDA GPU; AirLLM + bitsandbytes.

Run from the project root:
    python experiments/run_quantization_sweep.py [--levels q8 q4 q2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_settings
from src.runners.airllm_runner import run_experiment

LEVELS = ("q8", "q4", "q2")
RESULTS_DIR = "results"

# Quality thresholds: levels at or below this note are the "red line"
_RED_LINE_NOTE = "incoherent"


def _parse_args() -> list[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    return parser.parse_args().levels


def main() -> None:
    levels = _parse_args()
    settings = load_settings()

    if not settings.shard_path:
        print("[sweep] ERROR: SHARD_PATH not set in .env")
        sys.exit(1)
    if not settings.hf_token:
        print("[sweep] ERROR: HF_TOKEN not set in .env")
        sys.exit(1)

    print(f"[sweep] Quantization levels to run: {levels}")
    red_line: str | None = None

    for level in levels:
        print(f"\n[sweep] === {level.upper()} ===")
        s = load_settings(quant_level=level)
        try:
            record = run_experiment(s, RESULTS_DIR)
            m = record.metrics
            print(f"  TTFT={m.ttft_ms:.0f} ms  TPOT={m.tpot_ms:.0f} ms  "
                  f"RAM={m.peak_ram_gb:.1f} GB  VRAM={m.peak_vram_gb:.2f} GB  "
                  f"Quality={record.output.quality_note}")
            if record.output.quality_note == _RED_LINE_NOTE and red_line is None:
                red_line = level
        except Exception as exc:
            print(f"  ERROR at {level}: {exc}")

    print("\n[sweep] Sweep complete.")
    if red_line:
        print(f"[sweep] Accuracy RED LINE: {red_line} — output quality incoherent.")
        print("[sweep] Recommendation: use Q4 or above for coherent output.")
    else:
        print("[sweep] All tested levels produced coherent output.")


if __name__ == "__main__":
    main()
