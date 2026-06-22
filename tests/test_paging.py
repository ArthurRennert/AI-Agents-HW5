"""Tests for src/paging (sampler + mmap demo + figure)."""
import time
from pathlib import Path

import pytest

from src.paging.instrument import PageSampler, Sample, read_faults
from src.paging.layer_demo import _faults_per_layer, run_layer_paging_demo
from src.paging.paging_plot import plot_paging


def test_read_faults_returns_int_pair():
    minflt, majflt = read_faults()
    assert isinstance(minflt, int)
    assert isinstance(majflt, int)
    assert minflt >= 0 and majflt >= 0


def test_sampler_collects_samples():
    sampler = PageSampler(interval_sec=0.005)
    sampler.start()
    sampler.mark_layer(3)
    time.sleep(0.05)
    samples = sampler.stop()
    assert len(samples) >= 1
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[-1].layer == 3


def test_demo_writes_trace_and_faults(tmp_path):
    trace = run_layer_paging_demo(
        n_layers=3, layer_mb=2, results_dir=str(tmp_path)
    )
    assert trace["extension"] == "paging"
    assert trace["n_layers"] == 3
    assert len(trace["per_layer_load_sec"]) == 3
    assert len(trace["per_layer_faults"]) == 3
    assert trace["total_page_faults"] >= 0
    assert (Path(tmp_path) / "extension_paging_trace.json").exists()
    assert len(trace["samples"]) >= 1


def test_sawtooth_has_amplitude(tmp_path):
    trace = run_layer_paging_demo(
        n_layers=4, layer_mb=8, results_dir=str(tmp_path)
    )
    rss = [s["rss_gb"] for s in trace["samples"]]
    assert max(rss) > min(rss)  # RSS rose and fell -> a sawtooth exists


def test_faults_per_layer_handles_sparse_samples():
    s0 = Sample(0.0, 0.1, 100, 0, 0)
    s1 = Sample(0.1, 0.2, 180, 1, 0)
    out = _faults_per_layer([s0, s1], n_layers=2)
    assert out[0] == 81  # (180+1) - (100+0)
    assert out[1] == 0   # no samples tagged layer 1


def test_plot_paging_creates_file(tmp_path):
    trace = run_layer_paging_demo(
        n_layers=3, layer_mb=2, results_dir=str(tmp_path)
    )
    out = plot_paging(trace, str(tmp_path))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_paging_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_paging({"samples": []}, str(tmp_path))


def test_read_faults_psutil_fallback(monkeypatch):
    import src.paging.instrument as inst

    def _boom(*args, **kwargs):
        raise OSError("no /proc on this platform")

    monkeypatch.setattr(inst, "open", _boom, raising=False)
    minflt, majflt = read_faults()  # forces the psutil fallback path
    assert isinstance(minflt, int) and isinstance(majflt, int)
