"""Tests for src/viz/plots.py, architecture.py, and roofline.py."""
import json
from pathlib import Path

import pytest

from src.viz.architecture import plot_architecture
from src.viz.plots import (
    load_results_as_df,
    plot_energy_comparison,
    plot_latency_comparison,
    plot_memory_comparison,
    plot_pareto,
)
from src.viz.roofline import plot_roofline


def _write_record(results_dir: Path, engine: str, quant: str, quality: str) -> None:
    d = {
        "scenario": {
            "engine": engine, "model": "m", "quant_level": quant,
            "prompt_tokens": 5, "max_new_tokens": 10, "seed": 42,
        },
        "metrics": {
            "ttft_ms": 100.0, "tpot_ms": 50.0,
            "throughput_tokens_per_sec": 0.5,
            "peak_ram_gb": 8.0, "peak_vram_gb": 1.0,
            "wall_clock_sec": 200.0, "energy_wh": 10.0,
        },
        "output": {
            "generated_text": "hello", "n_output_tokens": 10,
            "quality_note": quality,
        },
        "timestamp": "2026-06-21T00:00:00+00:00",
    }
    (results_dir / f"{engine}_{quant}.json").write_text(json.dumps(d), encoding="utf-8")


def test_load_results_as_df_empty(tmp_path):
    df = load_results_as_df(str(tmp_path))
    assert df.empty


def test_load_results_as_df_columns(tmp_path):
    _write_record(tmp_path, "airllm", "fp16", "coherent")
    df = load_results_as_df(str(tmp_path))
    assert "engine" in df.columns
    assert "ttft_ms" in df.columns
    assert "quality_note" in df.columns
    assert len(df) == 1


def test_load_results_as_df_multiple(tmp_path):
    for q in ("fp16", "q8", "q4", "q2"):
        _write_record(tmp_path, "airllm", q, "coherent")
    df = load_results_as_df(str(tmp_path))
    assert len(df) == 4


def test_load_results_skips_bad_json(tmp_path):
    (tmp_path / "bad.json").write_text("not-json", encoding="utf-8")
    _write_record(tmp_path, "airllm", "fp16", "coherent")
    df = load_results_as_df(str(tmp_path))
    assert len(df) == 1


def _full_df(tmp_path: Path):
    qualities = {"fp16": "coherent", "q8": "coherent", "q4": "minor_degradation", "q2": "incoherent"}
    for q, qual in qualities.items():
        _write_record(tmp_path, "airllm", q, qual)
    return load_results_as_df(str(tmp_path))


def test_plot_latency_creates_file(tmp_path):
    df = _full_df(tmp_path)
    out = tmp_path / "figs"
    plot_latency_comparison(df, str(out))
    assert (out / "latency_comparison.png").exists()


def test_plot_memory_creates_file(tmp_path):
    df = _full_df(tmp_path)
    out = tmp_path / "figs"
    plot_memory_comparison(df, str(out))
    assert (out / "memory_comparison.png").exists()


def test_plot_energy_creates_file(tmp_path):
    df = _full_df(tmp_path)
    out = tmp_path / "figs"
    plot_energy_comparison(df, str(out))
    assert (out / "energy_comparison.png").exists()


def test_plot_pareto_creates_file(tmp_path):
    df = _full_df(tmp_path)
    out = tmp_path / "figs"
    plot_pareto(df, str(out))
    assert (out / "pareto.png").exists()


def test_plot_pareto_empty_df_no_crash(tmp_path):
    import pandas as pd
    plot_pareto(pd.DataFrame(), str(tmp_path / "figs"))


def test_plot_architecture_creates_file(tmp_path):
    plot_architecture(str(tmp_path / "figs"))
    assert (tmp_path / "figs" / "architecture.png").exists()


def test_plot_roofline_creates_file(tmp_path):
    plot_roofline(str(tmp_path / "figs"))
    assert (tmp_path / "figs" / "roofline.png").exists()
