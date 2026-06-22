"""A runnable demonstration of AirLLM's core mechanic: layer-by-layer paging.

We write N disk-backed "layer" shards, then load each one through an mmap, touch
every page (forcing real page-ins / faults), do a small matmul (the "compute"),
and then evict the layer with madvise(MADV_DONTNEED) before loading the next.
This reproduces the RSS sawtooth and the page-fault staircase that AirLLM
produces on any machine, including CPU-only, so the OS-paging narrative is
demonstrated rather than asserted.
"""
from __future__ import annotations

import gc
import json
import mmap
import tempfile
import time
from pathlib import Path

import numpy as np

from src.paging.instrument import PageSampler, Sample


def _build_shards(work_dir: Path, n_layers: int, layer_mb: int) -> list[Path]:
    """Write n_layers disk-backed float32 shards of layer_mb each."""
    floats = layer_mb * 1024 * 1024 // 4
    paths: list[Path] = []
    for i in range(n_layers):
        p = work_dir / f"layer_{i:03d}.dat"
        arr = np.memmap(p, dtype=np.float32, mode="w+", shape=(floats,))
        arr[:] = np.float32(i + 1)
        arr.flush()
        del arr
        paths.append(p)
    gc.collect()
    return paths


def _run_layer(path: Path) -> float:
    """mmap a shard, page it in, compute, then evict it; return elapsed seconds.

    Uses access=ACCESS_READ (portable across Windows and Unix) and always closes
    the mapping in a finally block so the file is never left locked on Windows.
    """
    t0 = time.perf_counter()
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            arr = np.frombuffer(mm, dtype=np.float32)
            _ = float(arr.sum())  # forces every page resident -> real page faults
            side = 256
            block = np.array(arr[: side * side]).reshape(side, side)
            _ = block @ block.T  # the "compute" step
            time.sleep(0.03)      # brief resident plateau so the page-in is visible
            del arr, block
            if hasattr(mm, "madvise"):  # Unix only; pages the layer back out
                try:
                    mm.madvise(mmap.MADV_DONTNEED)
                except (OSError, ValueError):
                    pass
        finally:
            mm.close()
    gc.collect()
    return time.perf_counter() - t0


def run_layer_paging_demo(
    n_layers: int = 24,
    layer_mb: int = 96,
    results_dir: str = "results",
    work_dir: str | None = None,
) -> dict:
    """Execute the paging demo and persist a trace to results/extension_paging_trace.json."""
    tmp = tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True)
    base = Path(tmp.name)
    shards = _build_shards(base, n_layers, layer_mb)

    sampler = PageSampler(interval_sec=0.005)
    sampler.start()
    time.sleep(0.05)  # capture a pre-load baseline
    per_layer_sec: list[float] = []
    for i, path in enumerate(shards):
        sampler.mark_layer(i)
        per_layer_sec.append(round(_run_layer(path), 4))
    samples = sampler.stop()
    tmp.cleanup()

    trace = _summarize(samples, per_layer_sec, n_layers, layer_mb)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extension_paging_trace.json").write_text(json.dumps(trace, indent=2))
    return trace


def _summarize(
    samples: list[Sample], per_layer_sec: list[float], n_layers: int, layer_mb: int
) -> dict:
    """Reduce raw samples + per-layer timings into a persistable trace dict."""
    faults0 = samples[0].minflt + samples[0].majflt if samples else 0
    faults_total = (
        (samples[-1].minflt + samples[-1].majflt) - faults0 if samples else 0
    )
    layer_faults = _faults_per_layer(samples, n_layers)
    return {
        "extension": "paging",
        "n_layers": n_layers,
        "layer_mb": layer_mb,
        "total_page_faults": faults_total,
        "per_layer_load_sec": per_layer_sec,
        "per_layer_faults": layer_faults,
        "samples": [
            {"t": s.t_sec, "rss_gb": s.rss_gb, "layer": s.layer,
             "minflt": s.minflt, "majflt": s.majflt}
            for s in samples
        ],
    }


def _faults_per_layer(samples: list[Sample], n_layers: int) -> list[int]:
    """Compute the fault delta attributable to each layer index."""
    out: list[int] = []
    for layer in range(n_layers):
        rows = [s for s in samples if s.layer == layer]
        if len(rows) >= 2:
            out.append((rows[-1].minflt + rows[-1].majflt) - (rows[0].minflt + rows[0].majflt))
        else:
            out.append(0)
    return out
