"""Economic analysis: API vs on-prem cost comparison (Tasks 4.1-4.6).

Loads token counts from results/*.json, computes costs, produces
figures/break_even.png, and saves results/economics.json.

Run from the project root:
    python experiments/run_economics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.economics.costs import (
    EconomicsConfig,
    api_cost_per_request,
    capex_monthly,
    compute_request_costs,
    cumulative_costs,
    onprem_variable_per_request,
)
from src.viz.plots import load_results_as_df

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _load_workload_tokens() -> tuple[int, int]:
    df = load_results_as_df(RESULTS_DIR)
    q4 = df[(df["engine"] == "airllm") & (df["quant_level"] == "q4")]
    if q4.empty:
        return 20, 200
    row = q4.iloc[0]
    return int(row.get("prompt_tokens", 20)), int(row.get("n_output_tokens", 200))


def _load_q4_wall_clock() -> float:
    df = load_results_as_df(RESULTS_DIR)
    q4 = df[(df["engine"] == "airllm") & (df["quant_level"] == "q4")]
    if q4.empty:
        return 1822.9
    return float(q4.iloc[0].get("wall_clock_sec", 1822.9))


def plot_break_even(cfg: EconomicsConfig, out_dir: str = FIGURES_DIR) -> None:
    n, api_total, onprem_total = cumulative_costs(cfg, max_requests=5_000)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n, api_total, label=f"API ({cfg.api_provider})", color="#4C72B0", linewidth=2)
    ax.plot(n, onprem_total, label="On-Prem (RTX 3090 + AirLLM Q4)",
            color="#C44E52", linewidth=2)

    # On-prem fixed cost baseline
    ax.axhline(capex_monthly(cfg), color="#C44E52", linestyle=":",
               linewidth=1, alpha=0.6, label=f"CAPEX floor (${capex_monthly(cfg):.2f}/mo)")

    ax.fill_between(n, api_total, onprem_total,
                    where=(onprem_total > api_total),
                    alpha=0.08, color="#C44E52", label="On-prem excess cost")

    ax.annotate(
        "API always cheaper\n(AirLLM energy cost > API per-token cost)\n"
        "On-prem value: Privacy + Data Sovereignty",
        xy=(4000, float(api_total[4000])), xytext=(2800, float(onprem_total[2000])),
        fontsize=8, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#CCCC00"),
    )

    ax.set_xlabel("Monthly Request Volume (requests/month)", fontsize=10)
    ax.set_ylabel("Monthly Cost (USD)", fontsize=10)
    ax.set_title(
        "Cumulative Monthly Cost: API vs On-Prem (AirLLM Q4)\n"
        f"Workload: {cfg.prompt_tokens} input + {cfg.output_tokens} output tokens",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = Path(out_dir) / "break_even.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[economics] Saved {out}")


def main() -> None:
    prompt_tokens, output_tokens = _load_workload_tokens()
    wall_clock = _load_q4_wall_clock()

    cfg = EconomicsConfig(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        wall_clock_sec=wall_clock,
    )
    costs = compute_request_costs(cfg)

    print("=" * 60)
    print("ECONOMIC ANALYSIS — API vs On-Prem")
    print("=" * 60)
    print(f"  Provider          : {cfg.api_provider} ({cfg.api_price_date})")
    print(f"  API input price   : ${cfg.api_input_usd_per_m:.2f}/M tokens")
    print(f"  API output price  : ${cfg.api_output_usd_per_m:.2f}/M tokens")
    print(f"  Workload          : {prompt_tokens} in + {output_tokens} out tokens")
    print(f"  API cost/request  : ${costs.api_usd:.6f}")
    print()
    print(f"  GPU               : RTX 3090 (${cfg.gpu_cost_usd:.0f} used, {cfg.amortization_years:.0f}-yr amort.)")
    print(f"  Electricity       : ${cfg.electricity_usd_per_kwh:.2f}/kWh")
    print(f"  Wall clock (Q4)   : {wall_clock:.1f}s per request")
    print(f"  Energy/request    : {costs.onprem_energy_kwh:.4f} kWh")
    print(f"  On-prem CAPEX/mo  : ${costs.onprem_capex_monthly_usd:.2f}")
    print(f"  On-prem var/req   : ${costs.onprem_variable_usd:.6f}")
    print()
    if costs.break_even_requests is None:
        print("  Break-even        : NONE — API is always cheaper per request")
        print("  On-prem advantage : Privacy, data sovereignty, zero per-token metering")
    else:
        print(f"  Break-even        : {costs.break_even_requests:.0f} requests/month")
    print("=" * 60)

    plot_break_even(cfg)

    result = {
        "api_provider": cfg.api_provider,
        "api_price_date": cfg.api_price_date,
        "api_input_usd_per_m_tokens": cfg.api_input_usd_per_m,
        "api_output_usd_per_m_tokens": cfg.api_output_usd_per_m,
        "workload_prompt_tokens": prompt_tokens,
        "workload_output_tokens": output_tokens,
        "api_cost_per_request_usd": round(costs.api_usd, 8),
        "onprem_energy_kwh_per_request": round(costs.onprem_energy_kwh, 6),
        "onprem_variable_usd_per_request": round(costs.onprem_variable_usd, 6),
        "onprem_capex_monthly_usd": round(costs.onprem_capex_monthly_usd, 2),
        "onprem_gpu_cost_usd": cfg.gpu_cost_usd,
        "onprem_amortization_years": cfg.amortization_years,
        "onprem_electricity_usd_per_kwh": cfg.electricity_usd_per_kwh,
        "break_even_requests_per_month": costs.break_even_requests,
        "conclusion": (
            "AirLLM on-prem is not cost-optimal vs API due to energy inefficiency "
            "of layer-paging. On-prem advantage is privacy and data sovereignty."
        ),
    }
    out = Path(RESULTS_DIR) / "economics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[economics] Saved {out}")
    print("[economics] Tasks 4.1-4.6 COMPLETE.")


if __name__ == "__main__":
    main()
