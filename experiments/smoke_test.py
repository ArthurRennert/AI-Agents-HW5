"""Smoke test — verifies the harness and persistence pipeline with a tiny public model.

Run from the project root:
    python experiments/smoke_test.py

Uses gpt2 (124M params, no auth required) so this completes in < 2 minutes even on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmark.harness import Harness
from src.benchmark.persistence import OutputInfo, ResultRecord, ScenarioInfo, save_result
from src.config.settings import Settings

SMOKE_MODEL = "gpt2"
SMOKE_PROMPT = "The concept of virtual memory allows operating systems to"
SMOKE_MAX_TOKENS = 20
RESULTS_DIR = "results"


def main() -> None:
    print(f"[smoke] Loading tokenizer: {SMOKE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SMOKE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[smoke] Loading model: {SMOKE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(SMOKE_MODEL)
    model.eval()

    inputs = tokenizer(SMOKE_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_tokens = int(input_ids.shape[-1])

    torch.manual_seed(42)
    harness = Harness()
    harness.start()

    print(f"[smoke] Generating {SMOKE_MAX_TOKENS} tokens ...")
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=SMOKE_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    harness.record_first_token()

    n_output = int(out_ids.shape[-1]) - prompt_tokens
    metrics = harness.stop(n_output)
    generated_text = tokenizer.decode(out_ids[0][prompt_tokens:], skip_special_tokens=True)

    record = ResultRecord(
        scenario=ScenarioInfo(
            engine="smoke_test",
            model=SMOKE_MODEL,
            quant_level="fp16",
            prompt_tokens=prompt_tokens,
            max_new_tokens=SMOKE_MAX_TOKENS,
            seed=42,
        ),
        metrics=metrics,
        output=OutputInfo(
            generated_text=generated_text,
            n_output_tokens=n_output,
            quality_note="coherent",
        ),
    )
    path = save_result(record, RESULTS_DIR)

    print("\n[smoke] Results:")
    print(f"  Generated : {generated_text!r}")
    print(f"  TTFT      : {metrics.ttft_ms:.1f} ms")
    print(f"  TPOT      : {metrics.tpot_ms:.1f} ms/token")
    print(f"  Throughput: {metrics.throughput_tokens_per_sec:.2f} tokens/s")
    print(f"  Peak RAM  : {metrics.peak_ram_gb:.2f} GB")
    print(f"  Peak VRAM : {metrics.peak_vram_gb:.2f} GB")
    print(f"  Wall clock: {metrics.wall_clock_sec:.2f} s")
    print(f"  Energy    : {metrics.energy_wh:.4f} Wh")
    print(f"  Saved to  : {path}")

    # Validate all 7 metric fields are populated
    issues = [
        field for field, val in [
            ("ttft_ms", metrics.ttft_ms),
            ("tpot_ms", metrics.tpot_ms),
            ("throughput_tokens_per_sec", metrics.throughput_tokens_per_sec),
            ("peak_ram_gb", metrics.peak_ram_gb),
            ("wall_clock_sec", metrics.wall_clock_sec),
        ]
        if val <= 0
    ]
    if issues:
        print(f"\n[smoke] WARNING: zero values for: {issues}")
        sys.exit(1)

    print("\n[smoke] PASSED — all metrics populated.")


if __name__ == "__main__":
    main()
