# Product Requirements Document (PRD)

## Project Overview and Context

This project is part of the AI Agents course (L08) and investigates the practical challenge of running state-of-the-art large language models (LLMs) on consumer-grade hardware. The lecture "LoRA, AirLLM & Running a Massive LLM Locally" motivates the problem: modern frontier LLMs (30B+ parameters) require tens of gigabytes of GPU memory, far exceeding what is available on a single consumer GPU.

**Central Claim:** Model `Qwen2.5-32B-Instruct` (~65 GB in FP16) cannot run directly on the available hardware (AMD Ryzen 9 5950X, 128 GB RAM, NVIDIA RTX 3090 24 GB VRAM), but successfully executes under AirLLM with quantization at a measurable latency cost.

**Hardware Profile:**

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 5950X (16 cores / 32 threads) |
| RAM | 128 GB DDR4 |
| GPU | NVIDIA RTX 3090, 24 GB VRAM |
| Storage | NVMe SSD |

---

## User Problem Description

Practitioners and researchers frequently encounter a fundamental constraint: large, high-quality LLMs do not fit in the VRAM of consumer GPUs. Direct loading of a 32B-parameter model in FP16 requires ~65 GB of VRAM — nearly 3× the RTX 3090's 24 GB capacity. The common workarounds are:

1. Renting expensive cloud GPU instances, which incurs ongoing cost.
2. Using paid API inference, which raises privacy and cost concerns.
3. Abandoning the model in favour of smaller, lower-quality alternatives.

**AirLLM** solves this by splitting the model into per-layer SafeTensors shards and loading one layer at a time into GPU memory, enabling execution at the price of high disk I/O and increased latency. This project measures the exact cost of that trade-off.

---

## Market Analysis and Target Audience

**Primary audience:** ML practitioners, AI researchers, and students who own consumer or workstation GPUs (8–24 GB VRAM) and want to run frontier-scale models locally without cloud costs.

**Secondary audience:** Organizations evaluating whether on-premises inference is economically viable compared to API providers (OpenAI, Anthropic, etc.) — particularly for privacy-sensitive workloads where data cannot leave the organization.

**Market context:** The inference cost landscape splits into three competing options:

| Option | Cost Model | Privacy | Latency |
|--------|-----------|---------|---------|
| **API inference** (OpenAI, Anthropic) | Pay-per-token; no upfront cost | Data leaves org | Low (data-center GPU) |
| **Cloud GPU rental** (AWS, RunPod, Lambda Labs) | Pay-per-hour | Data leaves org | Low-medium |
| **On-premises** (local GPU + AirLLM) | High CAPEX; low OPEX at volume | Full data privacy | High (disk I/O bound) |

This project quantifies the break-even threshold and helps practitioners make informed infrastructure decisions.

---

## Measurable Goals, KPIs, and Acceptance Criteria

| Goal | KPI | Acceptance Criterion |
|------|-----|---------------------|
| Prove direct execution failure | OOM or crash log produced | Model load attempt terminates with OOM or process kill |
| Enable AirLLM execution | Model completes generation | ≥ 1 successful inference completing 200 output tokens |
| Measure latency | TTFT (ms), TPOT (ms/token) | Both values recorded for every (engine × quant) cell |
| Measure throughput | Tokens / second | Recorded and compared across all scenarios |
| Measure resource usage | Peak RAM (GB), Peak VRAM (GB) | Recorded per scenario; VRAM ≤ 24 GB at all times |
| Measure energy | Estimated Wh per request | Derived from power draw × wall-clock time; method stated |
| Quantization sweep | ≥ 3 quant levels tested (FP16, Q8, Q4) | Metrics collected at each level; quality note recorded |
| Identify accuracy "red line" | Qualitative quality notes | One quant level identified where text coherence breaks |
| Economic analysis | Cost/request for API vs local; break-even volume | Break-even plot produced with all assumptions stated |
| Test coverage | Line coverage % | ≥ 85% via pytest + coverage |

---

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | **Baseline Runner:** Load and execute `Qwen2.5-32B-Instruct` via HuggingFace `transformers` without AirLLM; capture OOM / failure evidence. |
| FR-2 | **AirLLM Runner:** Execute the same model via AirLLM layer-by-layer sharding. Use `AutoModel` for Qwen-family to avoid class-mismatch errors. |
| FR-3 | **Measurement Harness:** Collect TTFT, TPOT, throughput, peak RAM, peak VRAM, wall-clock time, and energy estimate for every inference run. |
| FR-4 | **Quantization Sweep:** Execute the AirLLM runner at FP16, Q8, Q4 (and optionally Q2) using `bitsandbytes`; record all metrics and quality notes at each level. |
| FR-5 | **Results Persistence:** Save all raw metric data to `results/*.json` tagged by scenario (engine, quant level, model). Never hand-edit raw results. |
| FR-6 | **Data Aggregation:** Load results into pandas, produce tidy comparison tables, and export for visualization. |
| FR-7 | **Visualization:** Generate comparison figures (TTFT, TPOT, throughput, peak RAM/VRAM, energy) across all scenarios; produce a quality-vs-memory-vs-speed Pareto chart. |
| FR-8 | **Economic Analysis:** Compute API cost/request (token counting × provider prices), on-prem cost/request (CAPEX amortization + OPEX), and break-even plot. |
| FR-9 | **Technical Report:** Document all findings, methodology, concept analysis, and answers to the six research questions in `reports/`. |
| FR-10 | **README:** External-reader-facing with all figures embedded and clear, copy-pasteable reproduction instructions. |

---

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Each source file ≤ 150 lines (modularity). |
| NFR-2 | No secrets (HF token, API keys) in committed code; use `.env` + `.env.example`. |
| NFR-3 | Test coverage ≥ 85% (pytest + coverage report). |
| NFR-4 | All experiments reproducible: fixed prompt, fixed seed, fixed `max_new_tokens`, warm-up run discarded. |
| NFR-5 | All raw results stored unmodified in `results/`. |
| NFR-6 | Python 3.11 (pinned in `.python-version`); all deps pinned in `pyproject.toml`. |
| NFR-7 | Consistent code style (PEP 8 / ruff); docstrings on all public functions. |
| NFR-8 | `layer_shards_saving_path` must point to the NVMe drive (not the OS drive). |

---

## User Stories

- **As a student**, I want to run a 32B-parameter model on my gaming PC so that I can experiment with frontier-scale LLMs without cloud costs.
- **As an ML engineer**, I want to see latency and throughput measurements for AirLLM + quantization so that I can make informed infrastructure decisions.
- **As a data privacy officer**, I want to understand when on-premises inference is economically justified compared to API inference so that I can recommend the right solution for sensitive workloads.
- **As a course grader**, I want to see every L08 concept demonstrated with real experimental data so that I can verify understanding beyond theory.

---

## Use Cases

### UC-1: Direct Execution Attempt (Baseline)
**Actor:** Baseline runner  
**Precondition:** Model weights available; no AirLLM sharding.  
**Flow:** Load model via `transformers` → OOM / process killed → capture logs, error text, memory state.  
**Postcondition:** Failure evidence saved in `results/baseline_failure/`; bottleneck (memory vs compute) identified.

### UC-2: AirLLM Inference
**Actor:** AirLLM runner  
**Precondition:** Shards on fast NVMe path (`layer_shards_saving_path`); sufficient disk space.  
**Flow:** Load model shard-by-shard → forward pass layer-by-layer → collect tokens → record metrics via harness.  
**Postcondition:** Generated text + full metric set persisted to `results/airllm_<quant>.json`.

### UC-3: Quantization Sweep
**Actor:** Researcher (via experiment scripts)  
**Precondition:** AirLLM runner verified on FP16.  
**Flow:** For each quant level in {FP16, Q8, Q4, Q2}: configure bitsandbytes → run AirLLM with harness → record metrics + quality note.  
**Postcondition:** One quant level flagged as accuracy "red line".

### UC-4: Economic Analysis
**Actor:** Economics module  
**Precondition:** Token counts and energy measurements from UC-2 available in `results/`.  
**Flow:** Count tokens → multiply by API price → compute on-prem CAPEX + OPEX per request → plot break-even curve.  
**Postcondition:** Break-even volume identified; recommendation (API vs On-Prem, including privacy argument) written.

---

## Assumptions

- Hardware: AMD Ryzen 9 5950X, 128 GB RAM, RTX 3090 24 GB VRAM, NVMe SSD with ≥ 70 GB free space available.
- Model: `Qwen2.5-32B-Instruct` available in SafeTensors format on HuggingFace.
- AirLLM is compatible with the chosen model version and Python 3.11.
- `bitsandbytes` CUDA kernels support Q8/Q4/Q2 on RTX 3090 (Ampere architecture).
- Electricity rate and hardware prices are stated explicitly in the economics analysis (local values as of project date).
- A single warm-up run is sufficient to prime any caches before metric collection begins.

---

## Dependencies

| Dependency | Version (pinned) | Purpose |
|-----------|-----------------|---------|
| `airllm` | latest stable | Layer-by-layer model execution |
| `transformers` | ≥ 4.40 | HuggingFace model loading and tokenization |
| `torch` | ≥ 2.2 | Tensor ops, CUDA memory tracking |
| `accelerate` | ≥ 0.27 | Model offloading utilities |
| `bitsandbytes` | ≥ 0.43 | Quantization kernels (Q8/Q4/Q2) |
| `psutil` | ≥ 5.9 | RAM monitoring |
| `pandas` | ≥ 2.0 | Results aggregation and analysis |
| `matplotlib` | ≥ 3.8 | Visualization and figures |
| `python-dotenv` | ≥ 1.0 | Loading HF token from `.env` |

---

## Constraints

- **VRAM hard limit:** 24 GB (RTX 3090). Any approach requiring > 24 GB for a single layer will fail.
- **Disk space:** NVMe free space must exceed model on-disk size (~65 GB for FP16; ~16 GB for Q4).
- **Shard path:** `layer_shards_saving_path` must point to the NVMe drive; using the OS drive will cause a severe I/O bottleneck.
- **Time budget:** Full quantization sweep may take several hours; plan for overnight runs.
- **FP8:** Not supported on RTX 3090 (requires Ada Lovelace / Hopper); excluded from scope.

---

## Out of Scope

- Training or fine-tuning the model (inference only).
- Multi-GPU setups (single-GPU constraint throughout).
- Deployment to production endpoints or serving infrastructure.
- Models other than `Qwen2.5-32B-Instruct` (unless chosen as the multi-model-size extension).
- Automated quality metrics (perplexity evaluation) — quality assessment is qualitative unless chosen as an extension.
- Browser or mobile inference.

---

## Timeline and Milestones

| Milestone | Deliverables | Phase |
|-----------|-------------|-------|
| M1 — Planning Complete | PRD, PLAN, TODO, CONCEPTS, PROMPTS, 4 mini-PRDs approved | Phase 1 |
| M2 — Environment Ready | uv env, pinned deps, `.env`, hardware profile recorded in README | Phase 1 |
| M3 — Harness Verified | Measurement harness smoke-tested on tiny model + Q2 | Phase 2 |
| M4 — Baseline Captured | OOM/failure evidence + bottleneck diagnosis documented | Phase 2 |
| M5 — AirLLM Running | Successful FP16 inference + full metric set in `results/` | Phase 3 |
| M6 — Quantization Sweep Done | FP16/Q8/Q4 metrics collected; accuracy red line identified | Phase 3 |
| M7 — Graphs Generated | All comparison figures + Pareto chart in `figures/` | Phase 3 |
| M8 — Economic Analysis Done | Break-even plot + recommendation written | Phase 4 |
| M9 — Extension Complete | 1–2 extensions documented with figures | Phase 5 |
| M10 — Submission Ready | Report, README, ≥ 85% coverage, final QA gate passed, repo submitted | Phase 6 |
