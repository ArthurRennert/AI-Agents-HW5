"""Tests for src/economics/costs.py."""
import pytest

from src.economics.costs import (
    EconomicsConfig,
    api_cost_per_request,
    break_even_requests,
    capex_monthly,
    compute_request_costs,
    cumulative_costs,
    onprem_energy_kwh,
    onprem_variable_per_request,
)


def _cfg(**kwargs) -> EconomicsConfig:
    return EconomicsConfig(**kwargs)


# ── api_cost_per_request ─────────────────────────────────────────────────────

def test_api_cost_formula():
    cfg = _cfg(api_input_usd_per_m=2.50, api_output_usd_per_m=10.00,
               prompt_tokens=20, output_tokens=200)
    expected = (20 * 2.50 + 200 * 10.00) / 1_000_000
    assert abs(api_cost_per_request(cfg) - expected) < 1e-10


def test_api_cost_zero_tokens():
    cfg = _cfg(prompt_tokens=0, output_tokens=0)
    assert api_cost_per_request(cfg) == 0.0


def test_api_cost_scales_with_tokens():
    cfg_small = _cfg(prompt_tokens=10, output_tokens=100)
    cfg_large = _cfg(prompt_tokens=100, output_tokens=1000)
    assert api_cost_per_request(cfg_large) > api_cost_per_request(cfg_small)


# ── onprem_energy_kwh ────────────────────────────────────────────────────────

def test_energy_formula():
    cfg = _cfg(gpu_power_watts=350.0, wall_clock_sec=3600.0)
    # 350W × 1h = 350 Wh = 0.35 kWh
    assert abs(onprem_energy_kwh(cfg) - 0.35) < 1e-9


def test_energy_zero_wall_clock():
    cfg = _cfg(wall_clock_sec=0.0)
    assert onprem_energy_kwh(cfg) == 0.0


# ── onprem_variable_per_request ──────────────────────────────────────────────

def test_variable_cost_formula():
    cfg = _cfg(gpu_power_watts=350.0, wall_clock_sec=3600.0, electricity_usd_per_kwh=0.12)
    expected = 0.35 * 0.12  # 0.042
    assert abs(onprem_variable_per_request(cfg) - expected) < 1e-9


# ── capex_monthly ────────────────────────────────────────────────────────────

def test_capex_formula():
    cfg = _cfg(gpu_cost_usd=1800.0, amortization_years=3.0)
    assert abs(capex_monthly(cfg) - 50.0) < 1e-9


def test_capex_longer_amortization_is_cheaper():
    cfg3 = _cfg(amortization_years=3.0)
    cfg5 = _cfg(amortization_years=5.0)
    assert capex_monthly(cfg5) < capex_monthly(cfg3)


# ── break_even_requests ──────────────────────────────────────────────────────

def test_no_break_even_when_api_cheaper_per_request():
    # api=$0.002, onprem_var=$0.02 → no break-even
    cfg = _cfg(
        api_input_usd_per_m=2.50, api_output_usd_per_m=10.00,
        prompt_tokens=20, output_tokens=200,
        gpu_power_watts=350.0, wall_clock_sec=1822.9,
        electricity_usd_per_kwh=0.12,
    )
    assert break_even_requests(cfg) is None


def test_break_even_exists_when_api_more_expensive():
    # Make API very expensive, on-prem very cheap
    cfg = _cfg(
        api_input_usd_per_m=1000.0, api_output_usd_per_m=1000.0,
        prompt_tokens=20, output_tokens=200,
        gpu_power_watts=10.0, wall_clock_sec=1.0,
        electricity_usd_per_kwh=0.01,
        gpu_cost_usd=1000.0, amortization_years=3.0,
    )
    be = break_even_requests(cfg)
    assert be is not None
    assert be > 0


def test_break_even_formula_correctness():
    # Manually verify: api=$0.5, var=$0.1, capex=$50 → break-even = 50/(0.5-0.1)=125
    cfg = _cfg(
        api_input_usd_per_m=500_000.0, api_output_usd_per_m=0.0,
        prompt_tokens=1, output_tokens=0,   # api=$0.5
        gpu_power_watts=100.0, wall_clock_sec=3600.0,    # 0.1 kWh
        electricity_usd_per_kwh=1.0,                     # var=$0.1
        gpu_cost_usd=1800.0, amortization_years=3.0,     # capex=$50/mo
    )
    be = break_even_requests(cfg)
    assert be is not None
    assert abs(be - 125.0) < 0.01


# ── cumulative_costs ─────────────────────────────────────────────────────────

def test_cumulative_costs_shape():
    cfg = _cfg()
    n, api, onprem = cumulative_costs(cfg, max_requests=100)
    assert len(n) == 101
    assert len(api) == 101
    assert len(onprem) == 101


def test_cumulative_costs_at_zero():
    cfg = _cfg(gpu_cost_usd=600.0, amortization_years=3.0)
    n, api, onprem = cumulative_costs(cfg, max_requests=10)
    assert api[0] == 0.0
    assert abs(onprem[0] - capex_monthly(cfg)) < 1e-9


# ── compute_request_costs ────────────────────────────────────────────────────

def test_compute_request_costs_fields():
    cfg = _cfg()
    rc = compute_request_costs(cfg)
    assert rc.api_usd > 0
    assert rc.onprem_energy_kwh > 0
    assert rc.onprem_variable_usd > 0
    assert rc.onprem_capex_monthly_usd > 0
    # With default config, no break-even
    assert rc.break_even_requests is None
