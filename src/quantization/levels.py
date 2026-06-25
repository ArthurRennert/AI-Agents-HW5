"""Quantization level configuration and memory estimation."""
from __future__ import annotations

VALID_LEVELS = ("fp16", "q8", "q4", "q2")
_BITS: dict[str, int] = {"fp16": 16, "q8": 8, "q4": 4, "q2": 2}


def get_airllm_compression(quant_level: str):
    """Map a quant level to AirLLM's `compression` string, or None for fp16.

    AirLLM's bitsandbytes path accepts only '8bit' or '4bit'; 2-bit is not
    available with this engine, so q2 raises.
    """
    _validate(quant_level)
    if quant_level == "fp16":
        return None
    if quant_level == "q8":
        return "8bit"
    if quant_level == "q4":
        return "4bit"
    raise ValueError(
        "AirLLM bitsandbytes compression supports only fp16/8bit/4bit; "
        f"'{quant_level}' (2-bit) is not available with this engine."
    )


def get_bnb_config(quant_level: str):
    """Return a BitsAndBytesConfig for the given level, or None for fp16."""
    _validate(quant_level)
    if quant_level == "fp16":
        return None

    import torch
    from transformers import BitsAndBytesConfig

    if quant_level == "q8":
        return BitsAndBytesConfig(load_in_8bit=True)

    # q4 and q2 both use NF4 4-bit with double quantization
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def estimate_disk_gb(quant_level: str, param_billions: float) -> float:
    """Estimate on-disk model size in GB for the given quantization level."""
    _validate(quant_level)
    bits = _BITS[quant_level]
    return round(param_billions * 1e9 * bits / 8 / 1e9, 1)


def estimate_vram_gb(
    quant_level: str, param_billions: float, n_layers: int
) -> float:
    """Estimate peak VRAM for one layer under AirLLM in GB."""
    _validate(quant_level)
    if n_layers <= 0:
        raise ValueError("n_layers must be > 0")
    total_gb = estimate_disk_gb(quant_level, param_billions)
    return round(total_gb / n_layers, 3)


def _validate(quant_level: str) -> None:
    if quant_level not in VALID_LEVELS:
        raise ValueError(
            f"quant_level must be one of {VALID_LEVELS}, got '{quant_level}'"
        )
