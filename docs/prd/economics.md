# Mini-PRD: Economics Module

## 1. Description and Theoretical Background

The economics module compares three inference cost models — API, on-premises, and (optionally) cloud GPU rental — to find the **break-even usage volume** at which on-premises inference becomes cheaper than API calls. It surfaces both the cost and the privacy argument.

**API pricing model (variable cost, no upfront):**
```
cost_per_request = (n_input_tokens × price_per_input_token)
                 + (n_output_tokens × price_per_output_token)
```
Scales linearly with usage. No hardware investment. Data is sent to a third-party provider.

**On-premises cost model (high fixed cost, low marginal cost):**
```
CAPEX_per_request  = hardware_cost_usd / (lifespan_years × 365 × 24 × requests_per_hour)
OPEX_per_request   = energy_wh_per_request / 1000 × electricity_rate_usd_per_kwh
total_onprem_cost  = CAPEX_per_request + OPEX_per_request
```
High upfront capital expenditure (hardware), very low marginal cost per additional request at high volume.

**Break-even volume:**
The cumulative cost curves cross when:
```
total_requests × api_cost = hardware_cost + total_requests × onprem_opex
→ break_even = hardware_cost / (api_cost - onprem_opex)
```
Below break-even: API is cheaper (no capital tied up, volume is low).  
Above break-even: On-premises is cheaper (capital amortized, OPEX dominates).

**Context Caching / PagedAttention effect on API costs:**
API providers (Anthropic, OpenAI) offer prompt caching: if the same long system prompt / context prefix is reused across requests, the cached tokens cost ~10% of the full price. This lowers the effective API cost per request for repeated-context workloads, **shifting the break-even point to a higher volume** (making API more competitive at moderate usage). PagedAttention (vLLM) enables similar caching for on-premises deployments but does not apply to the AirLLM setup in this project.

**Privacy and data-security argument:**
On-premises inference keeps all data — prompts, outputs, user context — inside the organization's infrastructure. For regulated industries (healthcare, finance, legal) or any workload with sensitive data, this may be a hard non-negotiable constraint regardless of cost comparisons. The economics module must explicitly surface this in its recommendation section.

---

## 2. Specific Requirements

### Inputs

| Input | Type | Notes |
|-------|------|-------|
| `n_input_tokens` | `int` | Prompt token count; from harness results |
| `n_output_tokens` | `int` | Output token count; from harness results |
| `energy_wh_per_request` | `float` | From harness `energy_wh` field |
| `api_provider` | `str` | Provider name and model, e.g. `"OpenAI gpt-4o (2026-06-21)"` |
| `api_price_in_usd` | `float` | Price per input token in USD |
| `api_price_out_usd` | `float` | Price per output token in USD |
| `hardware_cost_usd` | `float` | Total hardware purchase cost (e.g., $2000 for RTX 3090 system) |
| `lifespan_years` | `float` | Assumed hardware lifespan (e.g., 3.0 years) |
| `monthly_volume` | `int` | Assumed monthly inference request volume (baseline assumption) |
| `electricity_rate_usd_per_kwh` | `float` | Local electricity rate; stated explicitly in output |

### Outputs

| Output | Type | Notes |
|--------|------|-------|
| `api_cost_per_request_usd` | `float` | Computed from token counts × prices |
| `onprem_capex_per_request_usd` | `float` | CAPEX amortized over lifespan and volume |
| `onprem_opex_per_request_usd` | `float` | Electricity cost per request |
| `onprem_total_per_request_usd` | `float` | CAPEX + OPEX |
| `break_even_volume_requests` | `int` | Total requests at crossover |
| `break_even_chart` | `figures/break_even.png` | Cumulative cost vs volume; crossover point labelled |
| `recommendation` | `str` | When API wins, when On-Prem wins; includes privacy argument |
| `economics.json` | file | All inputs and outputs persisted for reproducibility |

### Performance Metrics

- All assumptions must be explicitly stated in both `results/economics.json` and the report.
- The break-even chart must label the crossover volume and show both cost curves clearly.
- Prices must be cited with the provider name and retrieval date (prices change over time).

---

## 3. Constraints and Limitations

- API prices change over time; this analysis is a point-in-time snapshot. Prices must be cited with their retrieval date.
- Hardware lifespan is an assumption and results are sensitive to it; a brief sensitivity note (e.g., comparing 2-year vs 3-year vs 5-year lifespan) strengthens the analysis.
- The on-prem cost model omits maintenance, cooling, and operator time costs; these simplifications must be stated explicitly.
- The cloud GPU rental line is optional (P2); if included, use a real hourly rate × time-per-request.
- PagedAttention / context caching analysis is qualitative estimation, not empirical measurement.

---

## 4. Alternatives Considered

| Alternative | Reason Not Chosen |
|-------------|------------------|
| Live API cost measurement (real calls with billing) | Requires spending real money; out of scope. Public pricing pages are sufficient. |
| Full TCO (Total Cost of Ownership) with maintenance + cooling | Adds too many hard-to-verify assumptions; would reduce transparency. Keep model simple and explicit. |
| Multi-provider comparison (OpenAI + Anthropic + Groq) | Useful as an extension; core analysis uses one provider for clarity. |
| Monte Carlo sensitivity analysis | Overkill; a simple sensitivity table on lifespan and volume suffices. |

---

## 5. Success Criteria

- `api_cost_per_request_usd` and `onprem_total_per_request_usd` both computed from real inputs.
- Break-even volume derived mathematically and clearly visible on the chart.
- Every assumption stated explicitly in `results/economics.json` and in the report.
- Recommendation section explicitly addresses both cost dimension and privacy/data-security dimension.
- `results/economics.json` written with all inputs and outputs (reproducible by any reader who changes the assumptions).

---

## 6. Specific Test Scenarios

| Test | Input | Expected Output |
|------|-------|----------------|
| API cheaper at low volume | `monthly_volume=10` | `api_cost_per_request < onprem_total`; `break_even_volume` >> 10 |
| On-prem cheaper at high volume | `monthly_volume=100000` | `onprem_total < api_cost`; `break_even_volume` << 100000 |
| Zero electricity rate | `electricity_rate=0.0` | Only CAPEX in on-prem cost; no exception raised |
| Missing required input | `hardware_cost_usd=None` | `ValueError` raised before any computation |
| Chart generation | Any valid inputs | `figures/break_even.png` created; non-zero file size |
| Sensitivity: lifespan | Run with 2yr vs 5yr lifespan | `onprem_capex_per_request` is inversely proportional to lifespan |
| Sensitivity: volume | Run with `monthly_volume` × 10 | Break-even volume unchanged (it depends on cost rates, not assumed volume) |
| JSON persistence | Any valid inputs | `results/economics.json` written; loadable with `json.load()` |
