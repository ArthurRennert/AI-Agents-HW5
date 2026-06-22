"""Generate both Phase-5 extension figures from persisted results (Task 5.3).

Run from the project root (after the two experiment scripts):
    python experiments/generate_extension_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paging.paging_plot import plot_paging
from src.scaling.size_plot import plot_size_comparison
from src.scaling.size_sweep import load_size_results

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def main() -> None:
    size_results = load_size_results(RESULTS_DIR)
    if size_results:
        ram_gb = _profiled_ram_gb()
        plot_size_comparison(size_results, ram_gb, FIGURES_DIR)
    else:
        print("[ext] No size results — run experiments/run_size_comparison.py first.")

    trace_path = Path(RESULTS_DIR) / "extension_paging_trace.json"
    if trace_path.exists():
        plot_paging(json.loads(trace_path.read_text()), FIGURES_DIR)
    else:
        print("[ext] No paging trace — run experiments/run_paging_demo.py first.")

    print("[ext] Extension figures complete.")


def _profiled_ram_gb() -> float:
    path = Path(RESULTS_DIR) / "hardware.json"
    try:
        return float(json.loads(path.read_text())["ram_gb"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return 16.0


if __name__ == "__main__":
    main()
