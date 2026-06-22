"""Extension A experiment: multi-model size sweep (Tasks 5.2).

Reads the profiled RAM from results/hardware.json (falls back to a default) and
analyzes every Qwen2.5 size, writing results/extension_size_*.json.

Run from the project root:
    python experiments/run_size_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scaling.size_sweep import run_size_sweep

RESULTS_DIR = "results"
_DEFAULT_RAM_GB = 16.0


def _profiled_ram_gb() -> float:
    path = Path(RESULTS_DIR) / "hardware.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text())["ram_gb"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return _DEFAULT_RAM_GB


def main() -> None:
    ram_gb = _profiled_ram_gb()
    print(f"[size] Profiled RAM ceiling: {ram_gb:.1f} GB")
    results = run_size_sweep(ram_gb, RESULTS_DIR)

    print(f"\n[size] {'model':<26}{'fp16 GB':>9}{'baseline':>10}{'fits?':>7}"
          f"{'AirLLM TPOT':>13}")
    crossover = None
    for r in results:
        print(f"  {r.name:<24}{r.fp16_disk_gb:>9.1f}{r.baseline_required_gb:>10.1f}"
              f"{str(r.fits_baseline):>7}{r.est_tpot_s:>11.1f} s")
        if crossover is None and not r.fits_baseline:
            crossover = r.name

    if crossover:
        print(f"\n[size] RAM wall first hit at: {crossover} "
              f"(smaller models fit; this and larger require AirLLM).")
    print(f"[size] Records written to {RESULTS_DIR}/extension_size_*.json")


if __name__ == "__main__":
    main()
