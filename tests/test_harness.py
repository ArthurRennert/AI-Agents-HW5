"""Tests for src/benchmark/harness.py."""
import time

from src.benchmark.harness import Harness, Metrics


def test_metrics_dataclass_fields():
    m = Metrics(
        ttft_ms=100.0, tpot_ms=50.0, throughput_tokens_per_sec=2.0,
        peak_ram_gb=8.0, peak_vram_gb=4.0, wall_clock_sec=10.0, energy_wh=0.05,
    )
    assert m.ttft_ms == 100.0
    assert m.energy_wh == 0.05


def test_basic_flow_returns_metrics():
    harness = Harness()
    harness.start()
    time.sleep(0.05)
    harness.record_first_token()
    time.sleep(0.1)
    metrics = harness.stop(n_output_tokens=10)

    assert isinstance(metrics, Metrics)
    assert metrics.wall_clock_sec >= 0.1
    assert metrics.ttft_ms > 0
    assert metrics.throughput_tokens_per_sec > 0
    assert metrics.peak_ram_gb > 0


def test_ttft_reflects_first_token_timing():
    harness = Harness()
    harness.start()
    time.sleep(0.1)
    harness.record_first_token()
    time.sleep(0.1)
    metrics = harness.stop(n_output_tokens=5)
    # TTFT should be at least 80ms (slept 100ms before first token)
    assert metrics.ttft_ms >= 80.0


def test_record_first_token_only_once():
    harness = Harness()
    harness.start()
    time.sleep(0.02)
    harness.record_first_token()
    captured = harness._t_first_token
    time.sleep(0.05)
    harness.record_first_token()  # second call must be ignored
    assert harness._t_first_token == captured
    harness.stop(1)


def test_no_first_token_gives_zero_ttft():
    harness = Harness()
    harness.start()
    time.sleep(0.05)
    metrics = harness.stop(n_output_tokens=0)
    assert metrics.ttft_ms == 0.0


def test_throughput_is_tokens_over_time():
    harness = Harness()
    harness.start()
    harness.record_first_token()
    time.sleep(0.2)
    metrics = harness.stop(n_output_tokens=10)
    # rough check: 10 tokens / ~0.2s ≈ 50 tok/s — just verify it's non-zero and sane
    assert 1.0 < metrics.throughput_tokens_per_sec < 10_000


def test_stop_without_power_samples_gives_zero_energy():
    harness = Harness()
    harness.start()
    harness.record_first_token()
    # Don't allow time for power sampling; stop immediately
    metrics = harness.stop(n_output_tokens=1)
    # energy may be 0 if no nvidia-smi available — just check no exception
    assert metrics.energy_wh >= 0.0
