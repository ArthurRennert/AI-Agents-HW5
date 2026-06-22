"""Sample resident memory and page-fault counters over time, cross-platform.

Page faults are the OS signal that a layer's pages were not resident and had to be
brought in — exactly the event AirLLM triggers when it streams the next layer off
disk. On Linux we read /proc/self/stat (minflt/majflt); on Windows we fall back to
psutil's num_page_faults. The sampler runs in a daemon thread like the Phase-2
Harness so it does not perturb the measured workload.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


@dataclass
class Sample:
    """One timestamped snapshot of memory and cumulative page faults."""

    t_sec: float
    rss_gb: float
    minflt: int
    majflt: int
    layer: int


def read_faults() -> tuple[int, int]:
    """Return cumulative (minor, major) page faults for this process."""
    try:
        with open(f"/proc/{os.getpid()}/stat", encoding="ascii") as fh:
            after = fh.read().rsplit(")", 1)[1].split()
        # Fields after the comm/state block: index 7 = minflt, 9 = majflt.
        return int(after[7]), int(after[9])
    except (OSError, IndexError, ValueError):
        pass
    try:
        import psutil

        mem = psutil.Process().memory_info()
        return int(getattr(mem, "num_page_faults", 0)), 0
    except Exception:
        return 0, 0


class PageSampler:
    """Background sampler of RSS (GB) and page-fault counters."""

    def __init__(self, interval_sec: float = 0.02) -> None:
        self._interval = interval_sec
        self._samples: list[Sample] = []
        self._layer = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling in a daemon thread."""
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def mark_layer(self, layer: int) -> None:
        """Tag subsequent samples as belonging to the given layer index."""
        self._layer = layer

    def stop(self) -> list[Sample]:
        """Stop sampling and return the collected trace."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._samples

    def _loop(self) -> None:
        import psutil

        proc = psutil.Process()
        t0 = time.perf_counter()
        while not self._stop.is_set():
            rss_gb = proc.memory_info().rss / 1e9
            minflt, majflt = read_faults()
            self._samples.append(
                Sample(
                    t_sec=round(time.perf_counter() - t0, 4),
                    rss_gb=round(rss_gb, 4),
                    minflt=minflt,
                    majflt=majflt,
                    layer=self._layer,
                )
            )
            self._stop.wait(self._interval)
