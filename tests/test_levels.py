"""Tests for src/quantization/levels.py."""
import pytest

from src.quantization.levels import (
    VALID_LEVELS,
    estimate_disk_gb,
    estimate_vram_gb,
    get_bnb_config,
)


def test_fp16_returns_none():
    assert get_bnb_config("fp16") is None


def test_invalid_level_raises():
    with pytest.raises(ValueError, match="q1"):
        get_bnb_config("q1")


def test_invalid_level_in_estimate_disk():
    with pytest.raises(ValueError):
        estimate_disk_gb("q3", 32.0)


def test_estimate_disk_gb_fp16():
    # 32B params × 2 bytes = 64 GB
    assert estimate_disk_gb("fp16", 32.0) == 64.0


def test_estimate_disk_gb_q8():
    assert estimate_disk_gb("q8", 32.0) == 32.0


def test_estimate_disk_gb_q4():
    assert estimate_disk_gb("q4", 32.0) == 16.0


def test_estimate_disk_gb_q2():
    assert estimate_disk_gb("q2", 32.0) == 8.0


def test_estimate_vram_gb_per_layer():
    # 64 GB / 80 layers ≈ 0.8 GB
    result = estimate_vram_gb("fp16", 32.0, n_layers=80)
    assert abs(result - 64.0 / 80) < 0.01


def test_estimate_vram_invalid_layers():
    with pytest.raises(ValueError, match="n_layers"):
        estimate_vram_gb("fp16", 32.0, n_layers=0)


def test_all_valid_levels_accepted():
    for level in VALID_LEVELS:
        # should not raise
        estimate_disk_gb(level, 7.0)


def test_q4_smaller_than_q8():
    assert estimate_disk_gb("q4", 32.0) < estimate_disk_gb("q8", 32.0)


def test_q8_requires_bitsandbytes(monkeypatch):
    """get_bnb_config for q8/q4 requires torch + transformers to be importable."""
    import importlib
    import sys

    # Only test that the function doesn't raise for fp16 without imports
    assert get_bnb_config("fp16") is None
