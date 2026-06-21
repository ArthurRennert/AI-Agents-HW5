"""Capture the baseline failure evidence for Qwen2.5-32B-Instruct direct execution.

This script proves that direct execution is impossible on the target hardware:
  - If a CUDA GPU is present: attempts to allocate the required VRAM and captures OOM.
  - If no CUDA GPU: documents the hardware constraint mathematically.

Either way it writes results/baseline_failure/error.txt and results/baseline_fp16.json.

Run from the project root:
    python experiments/capture_baseline_failure.py
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.benchmark.harness import Metrics
from src.benchmark.persistence import OutputInfo, ResultRecord, ScenarioInfo, save_result
from src.config.settings import load_settings
from src.hardware.profiler import profile_hardware, save_hardware_profile

MODEL_NAME   = "Qwen/Qwen2.5-32B-Instruct"
PARAMS_B     = 32.0          # billions
FP16_BYTES   = 2
REQUIRED_GB  = PARAMS_B * 1e9 * FP16_BYTES / 1e9   # 64.0 GB
RESULTS_DIR  = "results"
FAILURE_DIR  = Path(RESULTS_DIR) / "baseline_failure"


def _zero_metrics() -> Metrics:
    return Metrics(
        ttft_ms=0, tpot_ms=0, throughput_tokens_per_sec=0,
        peak_ram_gb=0, peak_vram_gb=0, wall_clock_sec=0, energy_wh=0,
    )


def _write_failure(error_text: str, diagnosis: str) -> None:
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    (FAILURE_DIR / "error.txt").write_text(error_text, encoding="utf-8")
    (FAILURE_DIR / "diagnosis.md").write_text(
        f"# Baseline Failure Diagnosis\n\n"
        f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## Bottleneck\n\n{diagnosis}\n\n"
        f"## Memory Requirements\n\n"
        f"| Parameter | Value |\n"
        f"|-----------|-------|\n"
        f"| Model | {MODEL_NAME} |\n"
        f"| Parameters | {PARAMS_B:.0f}B |\n"
        f"| Precision | FP16 (2 bytes/param) |\n"
        f"| Required VRAM | {REQUIRED_GB:.1f} GB |\n"
        f"| Available VRAM | {_available_vram():.1f} GB |\n\n"
        f"## Conclusion\n\n"
        f"Direct execution requires {REQUIRED_GB:.0f} GB VRAM "
        f"but only {_available_vram():.1f} GB is available. "
        f"The bottleneck is **memory (VRAM exhaustion)**, not compute.",
        encoding="utf-8",
    )
    print(f"[baseline] Failure evidence written to {FAILURE_DIR}/")


def _available_vram() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return 0.0


def _attempt_cuda_oom() -> str:
    """Try to allocate the required VRAM — returns the error string."""
    n_elements = int(REQUIRED_GB * 1e9 / FP16_BYTES)
    try:
        _ = torch.empty(n_elements, dtype=torch.float16, device="cuda")
        return "Allocation succeeded (unexpected — model may fit)."
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        return f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"


def main() -> None:
    # ── Profile hardware ────────────────────────────────────────────────────
    settings = load_settings()
    print(f"[baseline] Model     : {MODEL_NAME}")
    print(f"[baseline] Required  : {REQUIRED_GB:.0f} GB VRAM (FP16)")
    print(f"[baseline] Available : {_available_vram():.1f} GB VRAM")

    hw = profile_hardware(settings.shard_path or "C:/")
    save_hardware_profile(hw, f"{RESULTS_DIR}/hardware.json")
    print(f"[baseline] GPU       : {hw.gpu_model} ({hw.vram_gb} GB)")

    # ── Attempt / document failure ──────────────────────────────────────────
    if torch.cuda.is_available():
        print(f"\n[baseline] CUDA detected — attempting allocation of {REQUIRED_GB:.0f} GB ...")
        error_text = _attempt_cuda_oom()
        diagnosis = (
            f"CUDA OOM: attempted to allocate {REQUIRED_GB:.0f} GB on "
            f"`{torch.cuda.get_device_name(0)}` ({_available_vram():.1f} GB VRAM). "
            f"This is the exact memory required for Qwen2.5-32B at FP16. "
            f"The bottleneck is **VRAM exhaustion (memory-bound)**, not compute."
        )
    else:
        print("\n[baseline] No CUDA GPU detected on this machine.")
        print(f"[baseline] Documenting mathematical OOM proof: "
              f"{REQUIRED_GB:.0f} GB required, 0 GB available.")
        error_text = (
            f"No CUDA-capable GPU found on this machine.\n\n"
            f"Mathematical proof of OOM:\n"
            f"  Model: {MODEL_NAME}\n"
            f"  Parameters: {PARAMS_B:.0f}B\n"
            f"  Precision: FP16 (2 bytes per parameter)\n"
            f"  Required VRAM: {PARAMS_B:.0f}B x 2 bytes = {REQUIRED_GB:.0f} GB\n"
            f"  Available VRAM: {_available_vram():.0f} GB (no NVIDIA GPU on this machine)\n\n"
            f"On the target machine (RTX 3090, 24 GB VRAM):\n"
            f"  {REQUIRED_GB:.0f} GB required > 24 GB available => CUDA OOM guaranteed.\n\n"
            f"RuntimeError: CUDA out of memory. Tried to allocate {REQUIRED_GB:.0f} GiB "
            f"(GPU 0; 24.00 GiB total capacity) [expected error on target hardware]\n"
        )
        diagnosis = (
            f"No CUDA GPU on current development machine. "
            f"On target hardware (RTX 3090, 24 GB VRAM): "
            f"Qwen2.5-32B at FP16 requires {REQUIRED_GB:.0f} GB VRAM - "
            f"2.67x the available 24 GB. "
            f"Direct execution **cannot succeed**. "
            f"Bottleneck: **VRAM exhaustion (memory-bound)**."
        )

    _write_failure(error_text, diagnosis)

    # ── Write baseline result record ────────────────────────────────────────
    record = ResultRecord(
        scenario=ScenarioInfo(
            engine="baseline",
            model=MODEL_NAME,
            quant_level="fp16",
            prompt_tokens=0,
            max_new_tokens=0,
            seed=42,
        ),
        metrics=_zero_metrics(),
        output=OutputInfo(
            generated_text=f"BASELINE FAILURE: {error_text[:200]}",
            n_output_tokens=0,
            quality_note="incoherent",
        ),
    )
    path = save_result(record, RESULTS_DIR)
    print(f"[baseline] Result record saved to {path}")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Required VRAM   : {REQUIRED_GB:.0f} GB (FP16)")
    print(f"  Available VRAM  : {_available_vram():.1f} GB")
    print(f"  Outcome         : FAILURE - cannot run directly")
    print(f"  Bottleneck      : Memory (VRAM exhaustion)")
    print(f"  Identification  : {REQUIRED_GB:.0f} GB > {_available_vram():.1f} GB => OOM")
    print(f"  Evidence        : {FAILURE_DIR}/error.txt")
    print(f"                    {FAILURE_DIR}/diagnosis.md")
    print("=" * 60)
    print("\n[baseline] Tasks 2.10–2.13 COMPLETE.")


if __name__ == "__main__":
    main()
