"""Tests for src/benchmark/persistence.py."""
import pytest

from src.benchmark.harness import Metrics
from src.benchmark.persistence import (
    OutputInfo,
    ResultRecord,
    ScenarioInfo,
    load_results,
    save_result,
)


def _make_metrics() -> Metrics:
    return Metrics(
        ttft_ms=100.0, tpot_ms=50.0, throughput_tokens_per_sec=0.5,
        peak_ram_gb=8.0, peak_vram_gb=4.0, wall_clock_sec=10.0, energy_wh=0.01,
    )


def _make_record(engine: str = "airllm", quant: str = "fp16",
                 quality: str = "coherent") -> ResultRecord:
    return ResultRecord(
        scenario=ScenarioInfo(
            engine=engine, model="test/model", quant_level=quant,
            prompt_tokens=10, max_new_tokens=20, seed=42,
        ),
        metrics=_make_metrics(),
        output=OutputInfo(
            generated_text="hello world", n_output_tokens=20, quality_note=quality,
        ),
    )


def test_roundtrip_save_load(tmp_path):
    record = _make_record()
    path = save_result(record, str(tmp_path))
    assert path.exists()

    loaded = load_results(str(tmp_path))
    assert len(loaded) == 1
    r = loaded[0]
    assert r.scenario.engine == "airllm"
    assert r.scenario.quant_level == "fp16"
    assert r.metrics.ttft_ms == 100.0
    assert r.output.quality_note == "coherent"
    assert r.output.generated_text == "hello world"


def test_invalid_quality_note_raises():
    with pytest.raises(ValueError, match="quality_note"):
        _make_record(quality="excellent")


def test_load_empty_dir(tmp_path):
    assert load_results(str(tmp_path)) == []


def test_multiple_scenarios(tmp_path):
    save_result(_make_record(engine="airllm", quant="fp16"), str(tmp_path))
    save_result(_make_record(engine="airllm", quant="q8"), str(tmp_path))
    loaded = load_results(str(tmp_path))
    assert len(loaded) == 2
    quant_levels = {r.scenario.quant_level for r in loaded}
    assert quant_levels == {"fp16", "q8"}


def test_timestamp_auto_set():
    record = _make_record()
    assert record.timestamp != ""
    assert "T" in record.timestamp  # ISO 8601 format


def test_all_valid_quality_notes(tmp_path):
    for note in ("coherent", "minor_degradation", "incoherent"):
        record = _make_record(quality=note, quant=note[:2])
        save_result(record, str(tmp_path))
    loaded = load_results(str(tmp_path))
    assert len(loaded) == 3
