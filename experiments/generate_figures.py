"""Aggregate results and generate all figures (Tasks 3.8-3.14).

Run from the project root:
    python experiments/generate_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.architecture import plot_architecture
from src.viz.plots import (
    load_results_as_df,
    plot_energy_comparison,
    plot_latency_comparison,
    plot_memory_comparison,
    plot_pareto,
)
from src.viz.roofline import plot_roofline

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def main() -> None:
    print("[figures] Loading results ...")
    df = load_results_as_df(RESULTS_DIR)

    if df.empty:
        print("[figures] No results found — run experiments first.")
        sys.exit(1)

    print(f"[figures] Loaded {len(df)} records: {df[['engine','quant_level']].to_dict('records')}")

    print("\n[figures] Generating charts ...")
    plot_latency_comparison(df, FIGURES_DIR)   # 3.9
    plot_memory_comparison(df, FIGURES_DIR)    # 3.10
    plot_energy_comparison(df, FIGURES_DIR)    # 3.11
    plot_pareto(df, FIGURES_DIR)               # 3.12
    plot_architecture(FIGURES_DIR)             # 3.13
    plot_roofline(FIGURES_DIR)                 # 3.14 (P2)

    print("\n[figures] Summary table:")
    cols = ["engine", "quant_level", "ttft_ms", "tpot_ms",
            "throughput_tokens_per_sec", "peak_ram_gb", "peak_vram_gb",
            "energy_wh", "quality_note"]
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False))

    airllm = df[df["engine"] == "airllm"]
    if not airllm.empty:
        red_line = airllm[airllm["quality_note"] == "incoherent"]
        if not red_line.empty:
            rl = red_line.iloc[0]["quant_level"]
            print(f"\n[figures] Quality RED LINE: {rl} produces incoherent output.")
            print("[figures] Recommendation: use q4 or above for coherent results.")

    print(f"\n[figures] All figures saved to {FIGURES_DIR}/")
    print("[figures] Tasks 3.8-3.14 COMPLETE.")


if __name__ == "__main__":
    main()
