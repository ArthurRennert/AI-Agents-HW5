"""Tests for src/config/settings.py."""
import pytest

from src.config.settings import VALID_QUANT_LEVELS, Settings, load_settings


def test_defaults_are_sane():
    s = Settings()
    assert s.model_name == "Qwen/Qwen2.5-32B-Instruct"
    assert s.seed == 42
    assert s.max_new_tokens == 200
    assert s.quant_level == "fp16"


def test_valid_load_with_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test123")
    monkeypatch.setenv("SHARD_PATH", "C:/llm_shards")
    monkeypatch.setenv("QUANT_LEVEL", "q4")
    s = load_settings()
    assert s.hf_token == "hf_test123"
    assert s.shard_path == "C:/llm_shards"
    assert s.quant_level == "q4"


def test_invalid_quant_level_raises(monkeypatch):
    monkeypatch.setenv("QUANT_LEVEL", "q1")
    with pytest.raises(ValueError, match="q1"):
        load_settings()


def test_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("QUANT_LEVEL", "q8")
    s = load_settings(quant_level="q4")
    assert s.quant_level == "q4"


def test_all_valid_quant_levels(monkeypatch):
    monkeypatch.delenv("QUANT_LEVEL", raising=False)
    for level in VALID_QUANT_LEVELS:
        s = load_settings(quant_level=level)
        assert s.quant_level == level


def test_seed_override():
    s = load_settings(seed=99)
    assert s.seed == 99


def test_max_new_tokens_from_env(monkeypatch):
    monkeypatch.setenv("MAX_NEW_TOKENS", "50")
    s = load_settings()
    assert s.max_new_tokens == 50
