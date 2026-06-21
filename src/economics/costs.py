"""API vs on-prem economic analysis for LLM inference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# GPT-4o pricing as retrieved from platform.openai.com on 2026-06-21
_GPT4O_INPUT_USD_PER_M = 2.50
_GPT4O_OUTPUT_USD_PER_M = 10.00


@dataclass
class EconomicsConfig:
    """All pricing and hardware assumptions for the economic analysis."""
    # API
    api_input_usd_per_m: float = _GPT4O_INPUT_USD_PER_M
    api_output_usd_per_m: float = _GPT4O_OUTPUT_USD_PER_M
    api_provider: str = "OpenAI GPT-4o"
    api_price_date: str = "2026-06-21"
    # On-prem hardware
    gpu_cost_usd: float = 1500.0      # RTX 3090 used, 2026
    amortization_years: float = 3.0
    electricity_usd_per_kwh: float = 0.12
    gpu_power_watts: float = 350.0
    # Workload (one inference call)
    prompt_tokens: int = 20
    output_tokens: int = 200
    wall_clock_sec: float = 1822.9    # Q4 AirLLM (best coherent level)


@dataclass
class RequestCosts:
    """Per-request cost breakdown."""
    api_usd: float
    onprem_energy_kwh: float
    onprem_variable_usd: float
    onprem_capex_monthly_usd: float
    break_even_requests: float | None   # None = API always cheaper


def api_cost_per_request(cfg: EconomicsConfig) -> float:
    """Total API cost for one inference call (input + output tokens)."""
    return (
        cfg.prompt_tokens * cfg.api_input_usd_per_m
        + cfg.output_tokens * cfg.api_output_usd_per_m
    ) / 1_000_000


def onprem_energy_kwh(cfg: EconomicsConfig) -> float:
    """Electricity consumed (kWh) for one on-prem inference call."""
    return cfg.gpu_power_watts * cfg.wall_clock_sec / 3_600 / 1_000


def onprem_variable_per_request(cfg: EconomicsConfig) -> float:
    """Variable electricity cost (USD) for one on-prem inference call."""
    return onprem_energy_kwh(cfg) * cfg.electricity_usd_per_kwh


def capex_monthly(cfg: EconomicsConfig) -> float:
    """Monthly GPU amortization cost (CAPEX / amortization_years / 12)."""
    return cfg.gpu_cost_usd / (cfg.amortization_years * 12)


def break_even_requests(cfg: EconomicsConfig) -> float | None:
    """Monthly request volume where on-prem total cost = API total cost.

    Returns None when the on-prem variable cost per request exceeds the API
    cost per request — meaning on-prem is never cheaper than API.
    """
    api = api_cost_per_request(cfg)
    var = onprem_variable_per_request(cfg)
    fixed = capex_monthly(cfg)
    if api <= var:
        return None  # API dominates even on variable cost alone
    return fixed / (api - var)


def cumulative_costs(
    cfg: EconomicsConfig, max_requests: int = 5_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monthly cumulative costs over 0..max_requests requests.

    Returns:
        n          - request volume array
        api_total  - total API spend at each volume
        onprem_total - total on-prem spend at each volume
    """
    n = np.arange(0, max_requests + 1)
    api_total = n * api_cost_per_request(cfg)
    onprem_total = capex_monthly(cfg) + n * onprem_variable_per_request(cfg)
    return n, api_total, onprem_total


def compute_request_costs(cfg: EconomicsConfig) -> RequestCosts:
    """Compute all per-request cost figures at once."""
    return RequestCosts(
        api_usd=api_cost_per_request(cfg),
        onprem_energy_kwh=onprem_energy_kwh(cfg),
        onprem_variable_usd=onprem_variable_per_request(cfg),
        onprem_capex_monthly_usd=capex_monthly(cfg),
        break_even_requests=break_even_requests(cfg),
    )
