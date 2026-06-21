"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

VALID_QUANT_LEVELS = ("fp16", "q8", "q4", "q2")

_DEFAULT_PROMPT = (
    "Explain the concept of virtual memory in operating systems, "
    "including how page tables and page faults work, "
    "and why this abstraction is useful for modern applications running on hardware "
    "with limited physical RAM."
)


@dataclass
class Settings:
    """All experiment parameters in one place — no hard-coded values anywhere else."""

    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    shard_path: str = ""
    quant_level: str = "fp16"
    seed: int = 42
    max_new_tokens: int = 200
    prompt: str = _DEFAULT_PROMPT
    hf_token: str = ""


def load_settings(**overrides: object) -> Settings:
    """Instantiate Settings from environment variables, with optional overrides."""
    settings = Settings(
        model_name=os.getenv("MODEL_NAME", Settings.model_name),
        shard_path=os.getenv("SHARD_PATH", ""),
        quant_level=os.getenv("QUANT_LEVEL", Settings.quant_level),
        seed=int(os.getenv("SEED", str(Settings.seed))),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", str(Settings.max_new_tokens))),
        prompt=os.getenv("PROMPT", _DEFAULT_PROMPT),
        hf_token=os.getenv("HF_TOKEN", ""),
    )
    for key, value in overrides.items():
        setattr(settings, key, value)

    if settings.quant_level not in VALID_QUANT_LEVELS:
        raise ValueError(
            f"Invalid quant_level '{settings.quant_level}'. "
            f"Must be one of: {VALID_QUANT_LEVELS}"
        )
    return settings
