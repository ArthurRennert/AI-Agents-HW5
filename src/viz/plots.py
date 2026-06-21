"""Visualization functions for experiment results (Tasks 3.8-3.12)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

QUANT_ORDER = ["fp16", "q8", "q4", "q2"]
_QUALITY_COLORS = {
    "coherent": "#4C72B0",
    "minor_degradation": "#DD8452",
    "incoherent": "#C44E52",
}


def load_results_as_df(results_dir: str = "results") -> pd.DataFrame:
    """Load all *.json result files into a tidy DataFrame."""
    rows = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            row = {
                **d["scenario"],
                **d["metrics"],
                "quality_note": d["output"]["quality_note"],
                "n_output_tokens": d["output"]["n_output_tokens"],
            }
            rows.append(row)
        except (KeyError, json.JSONDecodeError):
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _airllm_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "engine" not in df.columns:
        return pd.DataFrame()
    sub = df[df["engine"] == "airllm"].copy()
    sub["quant_level"] = pd.Categorical(
        sub["quant_level"], categories=QUANT_ORDER, ordered=True
    )
    return sub.sort_values("quant_level")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_latency_comparison(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """Bar charts: TTFT, TPOT, throughput per quantization level."""
    sub = _airllm_df(df)
    if sub.empty:
        print("[viz] No AirLLM results -- skipping latency chart")
        return
    labels = sub["quant_level"].astype(str).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, ylabel) in zip(axes, [
        ("ttft_ms", "TTFT (ms)"),
        ("tpot_ms", "TPOT (ms/token)"),
        ("throughput_tokens_per_sec", "Throughput (tokens/s)"),
    ]):
        ax.bar(labels, sub[col].tolist(), color="#4C72B0", edgecolor="black", linewidth=0.6)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Quantization")
        ax.set_ylabel(ylabel)
    fig.suptitle("AirLLM Latency by Quantization Level", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(out_dir) / "latency_comparison.png")


def plot_memory_comparison(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """Grouped bar chart: peak RAM and peak VRAM per quantization level."""
    sub = _airllm_df(df)
    if sub.empty:
        print("[viz] No AirLLM results -- skipping memory chart")
        return
    x = range(len(sub))
    labels = sub["quant_level"].astype(str).tolist()
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], sub["peak_ram_gb"].tolist(), w,
           label="Peak RAM (GB)", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar([i + w / 2 for i in x], sub["peak_vram_gb"].tolist(), w,
           label="Peak VRAM (GB)", color="#55A868", edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Peak Memory Usage by Quantization Level")
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(out_dir) / "memory_comparison.png")


def plot_energy_comparison(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """Bar chart: total energy consumed per quantization level."""
    sub = _airllm_df(df)
    if sub.empty:
        print("[viz] No AirLLM results -- skipping energy chart")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sub["quant_level"].astype(str).tolist(), sub["energy_wh"].tolist(),
           color="#C44E52", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Total Energy per 200-Token Run")
    fig.tight_layout()
    _save(fig, Path(out_dir) / "energy_comparison.png")


def plot_pareto(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """Scatter: TPOT vs peak VRAM, colored by quality. Red line at incoherent quant."""
    sub = _airllm_df(df)
    if sub.empty:
        print("[viz] No AirLLM results -- skipping Pareto chart")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in sub.iterrows():
        color = _QUALITY_COLORS.get(str(row["quality_note"]), "#888")
        ax.scatter(row["tpot_ms"], row["peak_vram_gb"], c=color, s=150, zorder=3,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(str(row["quant_level"]),
                    (row["tpot_ms"], row["peak_vram_gb"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    incoherent = sub[sub["quality_note"] == "incoherent"]
    if not incoherent.empty:
        threshold = float(incoherent["tpot_ms"].max())
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label="Quality red line (Q2)")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=9, label=q)
        for q, c in _QUALITY_COLORS.items()
    ]
    if not incoherent.empty:
        legend_handles.append(
            plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Quality red line")
        )
    ax.legend(handles=legend_handles, title="Quality")
    ax.set_xlabel("TPOT (ms/token)  [lower = faster]")
    ax.set_ylabel("Peak VRAM (GB)  [lower = more accessible]")
    ax.set_title("Quality vs Speed vs VRAM -- Pareto Frontier")
    fig.tight_layout()
    _save(fig, Path(out_dir) / "pareto.png")
