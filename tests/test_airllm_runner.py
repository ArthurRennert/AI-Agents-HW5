"""Tests for src/runners/airllm_runner.py."""
from unittest.mock import MagicMock, patch

import pytest

from src.hardware.profiler import DiskSpaceError
from src.runners.airllm_runner import _preflight
from src.config.settings import Settings


def _settings(**kwargs) -> Settings:
    base = dict(
        model_name="test/model",
        shard_path="C:/llm_shards",
        quant_level="fp16",
        hf_token="hf_test",
    )
    base.update(kwargs)
    return Settings(**base)


def test_preflight_passes_when_enough_space():
    with patch("src.runners.airllm_runner.check_disk_space") as mock_check:
        _preflight(_settings(quant_level="fp16"))
        # fp16 32B = 64 GB required
        mock_check.assert_called_once_with("C:/llm_shards", 64.0)


def test_preflight_raises_on_insufficient_space():
    with patch(
        "src.runners.airllm_runner.check_disk_space",
        side_effect=DiskSpaceError("Not enough space"),
    ):
        with pytest.raises(DiskSpaceError):
            _preflight(_settings(quant_level="fp16"))


def test_preflight_skipped_when_no_shard_path():
    with patch("src.runners.airllm_runner.check_disk_space") as mock_check:
        _preflight(_settings(shard_path=""))
        mock_check.assert_not_called()


def test_preflight_estimates_correct_size_for_q4():
    with patch("src.runners.airllm_runner.check_disk_space") as mock_check:
        _preflight(_settings(quant_level="q4"))
        # q4 32B = 16 GB required
        mock_check.assert_called_once_with("C:/llm_shards", 16.0)
