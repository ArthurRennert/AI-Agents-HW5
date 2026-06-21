"""Baseline runner — attempts direct model loading without AirLLM."""
from __future__ import annotations

import traceback
from pathlib import Path

from src.benchmark.harness import Harness, Metrics
from src.benchmark.persistence import OutputInfo, ResultRecord, ScenarioInfo, save_result
from src.config.settings import Settings


def run_baseline(settings: Settings, results_dir: str = "results") -> ResultRecord:
    """Attempt direct inference; capture OOM or failure evidence gracefully."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name, token=settings.hf_token
    )
    inputs = tokenizer(settings.prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    harness = Harness()
    generated_text = ""
    n_output = 0

    try:
        torch.manual_seed(settings.seed)
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            token=settings.hf_token,
        )
        harness.start()
        input_ids = inputs["input_ids"].to("cuda")
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=settings.max_new_tokens)
        harness.record_first_token()
        n_output = int(out_ids.shape[-1]) - prompt_tokens
        generated_text = tokenizer.decode(
            out_ids[0][prompt_tokens:], skip_special_tokens=True
        )
        metrics = harness.stop(n_output)
        quality_note = "coherent"

    except Exception as exc:
        metrics = harness.stop(0) if harness._t0 else _zero_metrics()
        _save_failure_evidence(exc, results_dir)
        generated_text = f"ERROR: {type(exc).__name__}: {exc}"
        quality_note = "incoherent"

    record = ResultRecord(
        scenario=ScenarioInfo(
            engine="baseline",
            model=settings.model_name,
            quant_level=settings.quant_level,
            prompt_tokens=prompt_tokens,
            max_new_tokens=settings.max_new_tokens,
            seed=settings.seed,
        ),
        metrics=metrics,
        output=OutputInfo(
            generated_text=generated_text,
            n_output_tokens=n_output,
            quality_note=quality_note,
        ),
    )
    save_result(record, results_dir)
    return record


def _save_failure_evidence(exc: Exception, results_dir: str) -> None:
    """Write error text and traceback to results/baseline_failure/."""
    failure_dir = Path(results_dir) / "baseline_failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    (failure_dir / "error.txt").write_text(
        f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    )
    print(f"[baseline] Failure evidence saved to {failure_dir}/error.txt")


def _zero_metrics() -> Metrics:
    """Return a zeroed Metrics object for runs that failed before harness started."""
    return Metrics(
        ttft_ms=0.0,
        tpot_ms=0.0,
        throughput_tokens_per_sec=0.0,
        peak_ram_gb=0.0,
        peak_vram_gb=0.0,
        wall_clock_sec=0.0,
        energy_wh=0.0,
    )
