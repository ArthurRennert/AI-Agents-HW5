"""Tests for src/runners/baseline_runner.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.benchmark.harness import Metrics
from src.runners.baseline_runner import _save_failure_evidence, _zero_metrics


def test_zero_metrics_has_all_fields():
    m = _zero_metrics()
    assert isinstance(m, Metrics)
    assert m.ttft_ms == 0.0
    assert m.wall_clock_sec == 0.0
    assert m.energy_wh == 0.0


def test_save_failure_evidence_creates_file(tmp_path):
    exc = RuntimeError("CUDA out of memory")
    _save_failure_evidence(exc, str(tmp_path))
    error_file = tmp_path / "baseline_failure" / "error.txt"
    assert error_file.exists()
    content = error_file.read_text()
    assert "RuntimeError" in content
    assert "CUDA out of memory" in content


def test_save_failure_evidence_creates_dir(tmp_path):
    exc = MemoryError("RAM exhausted")
    results_dir = str(tmp_path / "results")
    _save_failure_evidence(exc, results_dir)
    failure_dir = Path(results_dir) / "baseline_failure"
    assert failure_dir.is_dir()
    assert (failure_dir / "error.txt").exists()


def test_run_baseline_oom_writes_result(tmp_path):
    """run_baseline must write a result JSON even when the model raises OOM."""
    torch_mock = MagicMock()
    torch_mock.float16 = "float16"
    torch_mock.manual_seed = MagicMock()

    tokenizer_mock = MagicMock()
    tokenizer_mock.return_value = {"input_ids": MagicMock(shape=(1, 10))}
    tokenizer_mock.decode.return_value = ""

    model_class_mock = MagicMock()
    model_class_mock.from_pretrained.side_effect = RuntimeError("CUDA out of memory")

    auto_tok_mock = MagicMock()
    auto_tok_mock.from_pretrained.return_value = tokenizer_mock

    with patch.dict(
        "sys.modules",
        {
            "torch": torch_mock,
            "transformers": MagicMock(
                AutoModelForCausalLM=model_class_mock,
                AutoTokenizer=auto_tok_mock,
            ),
        },
    ):
        from src.config.settings import Settings
        from src.runners.baseline_runner import run_baseline

        settings = Settings(model_name="test/model", hf_token="hf_test")
        record = run_baseline(settings, results_dir=str(tmp_path))

    assert "ERROR" in record.output.generated_text
    assert record.output.quality_note == "incoherent"
    result_file = tmp_path / "baseline_fp16.json"
    assert result_file.exists()
