"""Run the AirLLM FP16 baseline experiment (Task 3.1–3.3).

Requires: HF_TOKEN and SHARD_PATH in .env; Qwen2.5-32B-Instruct accepted on HF;
          NVIDIA GPU with CUDA; AirLLM installed; NVMe with >= 70 GB free.

Run from the project root:
    python experiments/run_airllm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_settings
from src.runners.airllm_runner import run_experiment

RESULTS_DIR = "results"


def main() -> None:
    settings = load_settings(quant_level="fp16")

    if not settings.shard_path:
        print("[airllm] ERROR: SHARD_PATH not set in .env — cannot run.")
        sys.exit(1)
    if not settings.hf_token:
        print("[airllm] ERROR: HF_TOKEN not set in .env — cannot run.")
        sys.exit(1)

    print(f"[airllm] Model     : {settings.model_name}")
    print(f"[airllm] Precision : fp16")
    print(f"[airllm] Shards    : {settings.shard_path}")
    print(f"[airllm] Max tokens: {settings.max_new_tokens}")
    print("[airllm] Starting experiment — this may take 1–2 hours at FP16 ...")

    record = run_experiment(settings, RESULTS_DIR)
    m = record.metrics
    print("\n[airllm] Results:")
    print(f"  TTFT           : {m.ttft_ms:.1f} ms")
    print(f"  TPOT           : {m.tpot_ms:.1f} ms/token")
    print(f"  Throughput     : {m.throughput_tokens_per_sec:.4f} t/s")
    print(f"  Peak RAM       : {m.peak_ram_gb:.2f} GB")
    print(f"  Peak VRAM      : {m.peak_vram_gb:.2f} GB")
    print(f"  Wall clock     : {m.wall_clock_sec:.1f} s")
    print(f"  Energy         : {m.energy_wh:.2f} Wh")
    print(f"  Quality        : {record.output.quality_note}")
    print(f"\n[airllm] Saved -> results/airllm_fp16.json  (Tasks 3.1-3.3 DONE)")


if __name__ == "__main__":
    main()
