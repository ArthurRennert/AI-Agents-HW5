"""Tests for src/scaling (size sweep + figure)."""
from pathlib import Path

import pytest

from src.scaling.size_plot import plot_size_comparison
from src.scaling.size_sweep import (
    REGISTRY,
    ModelSpec,
    analyze_model,
    load_size_results,
    run_size_sweep,
)


def test_small_model_fits_in_ram():
    res = analyze_model(ModelSpec("tiny", 1.5, 28), ram_gb=64.0)
    assert res.fits_baseline is True
    assert res.baseline_outcome == "fits_in_ram"
    assert res.bottleneck.startswith("none")


def test_large_model_oom_on_small_ram():
    res = analyze_model(ModelSpec("big", 32.0, 64), ram_gb=16.8)
    assert res.fits_baseline is False
    assert res.baseline_outcome == "OOM"
    assert res.bottleneck == "disk_io_bandwidth"


def test_airllm_ram_is_bounded_below_baseline():
    res = analyze_model(ModelSpec("big", 32.0, 64), ram_gb=16.8)
    assert res.airllm_peak_ram_gb < res.baseline_required_gb
    assert res.disk_read_per_token_gb == res.fp16_disk_gb


def test_tpot_grows_with_size():
    small = analyze_model(ModelSpec("s", 3.0, 36), ram_gb=16.8)
    large = analyze_model(ModelSpec("l", 32.0, 64), ram_gb=16.8)
    assert large.est_tpot_s > small.est_tpot_s


def test_registry_is_sorted_ascending():
    params = [m.param_billions for m in REGISTRY]
    assert params == sorted(params)


def test_run_and_load_roundtrip(tmp_path):
    results = run_size_sweep(16.8, str(tmp_path))
    assert len(results) == len(REGISTRY)
    files = list(Path(tmp_path).glob("extension_size_*.json"))
    assert len(files) == len(REGISTRY)
    loaded = load_size_results(str(tmp_path))
    assert [r.name for r in loaded] == [r.name for r in results]


def test_crossover_present_on_16gb(tmp_path):
    results = run_size_sweep(16.8, str(tmp_path))
    fits = [r.fits_baseline for r in results]
    assert any(fits) and not all(fits)  # some fit, some do not


def test_plot_size_comparison_creates_file(tmp_path):
    results = run_size_sweep(16.8, str(tmp_path))
    out = plot_size_comparison(results, 16.8, str(tmp_path))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_size_comparison_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_size_comparison([], 16.8, str(tmp_path))
