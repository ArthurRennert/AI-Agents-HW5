"""Smoke test — verifies the harness and persistence pipeline with a tiny public model.

Run from the project root:
    python experiments/smoke_test.py

Uses gpt2 (124M params, no auth required) so this completes in < 2 minutes even on CPU.
Uses TextIteratorStreamer so TTFT is measured on the actual first token, not end-of-generation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.benchmark.harness import Harness
from src.benchmark.persistence import OutputInfo, ResultRecord, ScenarioInfo, save_result

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

    input_ids = tokenizer(SMOKE_PROMPT, return_tensors="pt")["input_ids"]
    prompt_tokens = int(input_ids.shape[-1])

    torch.manual_seed(42)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    harness = Harness()
    harness.start()

    gen_thread = Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=input_ids,
            max_new_tokens=SMOKE_MAX_TOKENS,
            streamer=streamer,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        ),
        daemon=True,
    )
    print(f"[smoke] Generating {SMOKE_MAX_TOKENS} tokens ...")
    gen_thread.start()

    chunks: list[str] = []
    first_token_seen = False
    for chunk in streamer:
        if not first_token_seen and chunk.strip():
            harness.record_first_token()
            first_token_seen = True
        chunks.append(chunk)

    gen_thread.join()
    generated_text = "".join(chunks)
    n_output = len(tokenizer(generated_text, add_special_tokens=False)["input_ids"])
    metrics = harness.stop(n_output)

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
