"""AirLLM runner — layer-by-layer inference for oversized models."""
from __future__ import annotations

from threading import Thread

from src.benchmark.harness import Harness
from src.benchmark.persistence import OutputInfo, ResultRecord, ScenarioInfo, save_result
from src.config.settings import Settings
from src.hardware.profiler import check_disk_space
from src.quantization.levels import estimate_disk_gb, get_bnb_config

_QWEN_32B_PARAMS = 32.0  # billions


def run_experiment(settings: Settings, results_dir: str = "results") -> ResultRecord:
    """Run inference via AirLLM with the configured quantization level."""
    import torch
    from transformers import AutoTokenizer, TextIteratorStreamer

    _preflight(settings)
    torch.manual_seed(settings.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name, token=settings.hf_token
    )
    input_ids = tokenizer(settings.prompt, return_tensors="pt")["input_ids"]
    prompt_tokens = int(input_ids.shape[-1])

    model = _load_model(settings)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    harness = Harness()
    harness.start()

    gen_thread = Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=input_ids,
            max_new_tokens=settings.max_new_tokens,
            streamer=streamer,
            do_sample=False,
        ),
        daemon=True,
    )
    gen_thread.start()

    chunks: list[str] = []
    first_token_seen = False
    for chunk in streamer:
        if not first_token_seen and chunk:
            harness.record_first_token()
            first_token_seen = True
        chunks.append(chunk)

    gen_thread.join()
    generated_text = "".join(chunks)
    n_output = len(tokenizer(generated_text, add_special_tokens=False)["input_ids"])
    metrics = harness.stop(n_output)

    record = ResultRecord(
        scenario=ScenarioInfo(
            engine="airllm",
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
            quality_note="coherent",
        ),
    )
    save_result(record, results_dir)
    return record


def _preflight(settings: Settings) -> None:
    """Validate shard path has enough free disk space before any download."""
    if settings.shard_path:
        required_gb = estimate_disk_gb(settings.quant_level, _QWEN_32B_PARAMS)
        check_disk_space(settings.shard_path, required_gb)


def _load_model(settings: Settings):
    """Load the model via AirLLM AutoModel with the configured quantization."""
    from airllm import AutoModel

    bnb_config = get_bnb_config(settings.quant_level)
    return AutoModel.from_pretrained(
        settings.model_name,
        layer_shards_saving_path=settings.shard_path,
        hf_token=settings.hf_token,
        compression=bnb_config,
    )
