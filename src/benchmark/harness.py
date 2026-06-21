"""Measurement harness — wraps an inference run and collects all metrics."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class Metrics:
    """All performance metrics collected during one inference run."""

    ttft_ms: float
    tpot_ms: float
    throughput_tokens_per_sec: float
    peak_ram_gb: float
    peak_vram_gb: float
    wall_clock_sec: float
    energy_wh: float


class Harness:
    """Instruments an inference call and computes TTFT, TPOT, RAM, VRAM, and energy."""

    def __init__(self) -> None:
        self._t0: float = 0.0
        self._t_first_token: float = 0.0
        self._ram_samples: list[float] = []
        self._power_samples: list[float] = []
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Begin timing and start background RAM + power sampling threads."""
        self._t0 = time.perf_counter()
        self._t_first_token = 0.0
        self._ram_samples = []
        self._power_samples = []
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._sample_ram, daemon=True),
            threading.Thread(target=self._sample_power, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def record_first_token(self) -> None:
        """Call immediately when the first output token is emitted."""
        if self._t_first_token == 0.0:
            self._t_first_token = time.perf_counter()

    def stop(self, n_output_tokens: int) -> Metrics:
        """Stop sampling and return all computed metrics."""
        t_end = time.perf_counter()
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)

        wall_clock_sec = t_end - self._t0
        ttft_ms = (
            (self._t_first_token - self._t0) * 1000 if self._t_first_token else 0.0
        )
        decode_tokens = max(n_output_tokens - 1, 1)
        tpot_ms = max(wall_clock_sec * 1000 - ttft_ms, 0.0) / decode_tokens
        throughput = n_output_tokens / wall_clock_sec if wall_clock_sec > 0 else 0.0

        peak_ram_gb = max(self._ram_samples, default=0.0)
        peak_vram_gb = _get_peak_vram()
        mean_power_w = (
            sum(self._power_samples) / len(self._power_samples)
            if self._power_samples
            else 0.0
        )
        energy_wh = mean_power_w * wall_clock_sec / 3600

        return Metrics(
            ttft_ms=round(ttft_ms, 2),
            tpot_ms=round(tpot_ms, 2),
            throughput_tokens_per_sec=round(throughput, 4),
            peak_ram_gb=round(peak_ram_gb, 2),
            peak_vram_gb=round(peak_vram_gb, 2),
            wall_clock_sec=round(wall_clock_sec, 2),
            energy_wh=round(energy_wh, 4),
        )

    def _sample_ram(self) -> None:
        import psutil

        proc = psutil.Process()
        while not self._stop_event.is_set():
            self._ram_samples.append(proc.memory_info().rss / 1e9)
            self._stop_event.wait(timeout=0.5)

    def _sample_power(self) -> None:
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                self._power_samples.append(float(out.stdout.strip()))
            except Exception:
                pass
            self._stop_event.wait(timeout=1.0)


def _get_peak_vram() -> float:
    """Return peak VRAM allocated by PyTorch in GB, or 0.0 if no CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0
