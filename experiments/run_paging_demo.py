"""Extension B experiment: layer-by-layer paging demonstration (Task 5.2).

Runs a real mmap layer-paging workload on this machine (CPU-only friendly) and
records the RSS sawtooth + page-fault trace to results/extension_paging_trace.json.

To instead instrument a *real* AirLLM generation run, wrap the generate() call in
experiments/run_airllm.py with a PageSampler (see reports/extension.md); the demo
here exists so the mechanic is reproducible without downloading a 32B model.

Run from the project root:
    python experiments/run_paging_demo.py [--layers 32 --layer-mb 48]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paging.layer_demo import run_layer_paging_demo

RESULTS_DIR = "results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--layer-mb", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[paging] Paging {args.layers} layers x {args.layer_mb} MB off disk ...")
    trace = run_layer_paging_demo(
        n_layers=args.layers, layer_mb=args.layer_mb, results_dir=RESULTS_DIR
    )
    n = len(trace["samples"])
    peak = max((s["rss_gb"] for s in trace["samples"]), default=0.0)
    print(f"[paging] Collected {n} samples; peak RSS {peak:.2f} GB; "
          f"total page faults {trace['total_page_faults']:,}")
    print(f"[paging] Trace written to {RESULTS_DIR}/extension_paging_trace.json")


if __name__ == "__main__":
    main()
