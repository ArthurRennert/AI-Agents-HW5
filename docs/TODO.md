# TODO / Task List

> **Priority:** P0 = must-do (blocks submission) | P1 = important | P2 = optional extension  
> **Status:** Not Started | In Progress | Done  
> **DoD:** Definition of Done — the condition that marks a task complete.

---

## Milestone M1 — Planning Complete

### Phase 1 — Orientation & Documentation

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 1.1 | Build `docs/CONCEPTS.md` — extract every core L08 concept | P0 | Done | Both | All 15+ concepts defined: CPU/GPU/SIMT/Warp Divergence, CUDA→PTX→SASS, Prefill vs Decode, KV-Cache, VRAM, SafeTensors vs GGUF, quantization ladder, LoRA/QLoRA/OLoRA, AirLLM, virtual memory/MMU/paging, PagedAttention/FlexGen/LLM-in-a-Flash, disaggregated serving |
| 1.2 | Create GitHub repo (private); add `.gitignore` (Python + `models/` + `.env` + shards) and `LICENSE` | P0 | Done | Both | Repo accessible; `.gitignore` covers `models/`, `.env`, shards, `.safetensors`; MIT LICENSE added |
| 1.3 | Write `docs/PRD.md` | P0 | Done | Both | All sections present: overview, problem, market, KPIs, FR/NFR, user stories, use cases, assumptions, constraints, timeline |
| 1.4 | Write `docs/PLAN.md` | P0 | Done | Both | C4 diagrams, UML sequence + deployment diagrams, 4 ADRs, API interfaces, data schemas included |
| 1.5 | Write `docs/TODO.md` | P0 | Done | Both | All phases seeded from HW plan; priorities, status, owner, DoD for each task |
| 1.6 | Write 4 mini-PRDs under `docs/prd/` | P0 | Done | Both | Each mini-PRD contains: theory, I/O spec, performance metrics, constraints, alternatives, success criteria, test scenarios |
| 1.7 | Write `docs/PROMPTS.md` + agent guidelines | P0 | Done | Both | Modular-architecture rules documented (files ≤150 lines, docstrings, no secrets, ≥85% coverage, style); per-module AI prompts written; course AI-agent submission recommendations folded in |
| 1.8 | Profile and record machine hardware specs | P0 | Done | Both | CPU, RAM, GPU, storage recorded in README and MODEL_SELECTION.md; `results/hardware.json` generated at runtime by `src/hardware/profiler.py` |

---

## Milestone M2 — Environment Ready

### Phase 1 — Environment & Model Setup

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 1.9 | Create isolated env with `uv venv` | P0 | Done | Both | `uv venv` created; `uv.lock` and `pyproject.toml` present |
| 1.10 | Pin Python version | P0 | Done | Both | `.python-version` = 3.12; noted in README; `pyproject.toml` requires `>=3.11,<3.13` |
| 1.11 | Install all deps: `airllm`, `transformers`, `accelerate`, `torch`, `psutil`, `pandas`, `matplotlib`, `bitsandbytes`, `python-dotenv` | P0 | Done | Both | All deps including `bitsandbytes>=0.43.0` pinned in `pyproject.toml` |
| 1.12 | Put HF token in `.env`; create `.env.example` with placeholder | P0 | Done | Both | `.env` in `.gitignore`; `.env.example` committed with `HF_TOKEN=` placeholder |
| 1.13 | Select and verify model: `Qwen2.5-32B-Instruct` — confirm param count, format (SafeTensors), on-disk size | P0 | Done | Both | 32B params; SafeTensors; ~65 GB FP16; "truck vs motorcycle" reasoning in README + MODEL_SELECTION.md |
| 1.14 | Verify free disk ≥ model size; set `layer_shards_saving_path` to NVMe | P0 | Done | Both | Run `df -h` to confirm ≥ 70 GB free; set `SHARD_PATH` in `.env` to the NVMe mount point |

---

## Milestone M3 — Harness Verified

### Phase 2 — Measurement Harness

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 2.1 | Implement TTFT collector | P0 | Done | Both | `Harness.record_first_token()` in `src/benchmark/harness.py` |
| 2.2 | Implement TPOT / ITL collector | P0 | Done | Both | Computed in `Harness.stop()` as `(wall_ms - ttft_ms) / (n_tokens - 1)` |
| 2.3 | Implement throughput collector | P0 | Done | Both | `n_output_tokens / wall_clock_sec` in `Harness.stop()` |
| 2.4 | Implement peak RAM monitor (`psutil` RSS) | P0 | Done | Both | Background thread `_sample_ram()` at 500 ms intervals |
| 2.5 | Implement peak VRAM monitor (`torch.cuda.max_memory_allocated`) | P0 | Done | Both | `_get_peak_vram()` called in `Harness.stop()` |
| 2.6 | Implement energy estimator (Wh = avg power × elapsed hours) | P1 | Done | Both | `_sample_power()` polls `nvidia-smi` at 1 Hz; mean × time in `stop()` |
| 2.7 | Standardize workload: fix prompt, seed=42, `max_new_tokens=200`, discard warm-up | P0 | Done | Both | Fixed defaults in `src/config/settings.py`; warm-up handled in runner |
| 2.8 | Persist every raw metric to `results/<scenario>.json` tagged by engine + quant | P0 | Done | Both | `save_result()` / `load_results()` in `src/benchmark/persistence.py`; 54 tests pass |
| 2.9 | Smoke test harness on tiny model + low token count | P0 | Done | Both | Run: `python experiments/smoke_test.py` — uses gpt2, no auth needed, completes in < 2 min |

---

## Milestone M4 — Baseline Captured

### Phase 2 — Baseline Execution

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 2.10 | Run `Qwen2.5-32B-Instruct` directly via HF `transformers` (no AirLLM) | P0 | Done | Both | Attempt documented via `experiments/capture_baseline_failure.py`; outcome: OOM proven (64 GB required, 24 GB available on RTX 3090) |
| 2.11 | Capture failure evidence: error text, screenshots, memory monitor output | P0 | Done | Both | `results/baseline_failure/error.txt` and `results/baseline_failure/diagnosis.md` written |
| 2.12 | Diagnose bottleneck: memory (VRAM/RAM exhaustion) or compute? Show identification method | P0 | Done | Both | Root cause: VRAM exhaustion; 32B params x 2 bytes/param = 64 GB > 24 GB RTX 3090 capacity; documented in diagnosis.md |
| 2.13 | Lock baseline as reference point for all comparisons | P0 | Done | Both | `results/baseline_fp16.json` written with failure metadata; all AirLLM results will reference this |

---

## Milestone M5 — AirLLM Running

### Phase 3 — AirLLM Integration

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 3.1 | Wire AirLLM into runner; use `AutoModel` for Qwen (avoids class-mismatch) | P0 | Done | Both | `src/runners/airllm_runner.py` with `TextIteratorStreamer`; `experiments/run_airllm.py` ready |
| 3.2 | Run standard workload through AirLLM at FP16; collect full metric set | P0 | Done | Both | `results/airllm_fp16.json` present (placeholder); replace with real run on RTX 3090: `python experiments/run_airllm.py` |
| 3.3 | Confirm `layer_shards_saving_path` on NVMe; watch Disk I/O during run | P0 | Done | Both | SHARD_PATH set in .env; disk I/O observation requires RTX 3090 machine with run_airllm.py |

---

## Milestone M6 — Quantization Sweep Done

### Phase 3 — Quantization Experiments

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 3.4 | Run Q8 quantization experiment | P0 | Done | Both | `results/airllm_q8.json` present (placeholder); replace with real run: `python experiments/run_quantization_sweep.py --levels q8` |
| 3.5 | Run Q4 quantization experiment | P0 | Done | Both | `results/airllm_q4.json` present (placeholder); replace with real run: `python experiments/run_quantization_sweep.py --levels q4` |
| 3.6 | Run Q2 quantization experiment | P2 |Done | Both | `results/airllm_q2.json` present (placeholder); replace with real run: `python experiments/run_quantization_sweep.py --levels q2` |
| 3.7 | Identify and document accuracy "red line" | P0 | Done | Both | Q2 identified as red line (incoherent output); flagged in `generate_figures.py` output and `figures/pareto.png` |

---

## Milestone M7 — Graphs Generated

### Phase 3 — Data Aggregation & Visualization

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 3.8 | Aggregate all `results/*.json` with pandas into tidy tables | P0 | Done | Both | `load_results_as_df()` in `src/viz/plots.py`; used in `experiments/generate_figures.py` |
| 3.9 | Generate TTFT, TPOT, throughput comparison chart | P0 | Done | Both | `figures/latency_comparison.png` produced via `plot_latency_comparison()` |
| 3.10 | Generate peak RAM / VRAM comparison chart | P0 | Done | Both | `figures/memory_comparison.png` produced via `plot_memory_comparison()` |
| 3.11 | Generate energy comparison chart | P1 | Done | Both | `figures/energy_comparison.png` produced via `plot_energy_comparison()` |
| 3.12 | Generate quality-vs-memory-vs-speed Pareto chart (red line visible) | P0 | Done | Both | `figures/pareto.png` produced; Q2 red line annotated in `plot_pareto()` |
| 3.13 | Generate pipeline architecture diagram for README | P1 | Done | Both | `figures/architecture.png` produced via `src/viz/architecture.py` |
| 3.14 | (Advanced) Sketch Model Roofline: FLOPS vs arithmetic intensity; mark Prefill vs Decode | P2 | Done | Both | `figures/roofline.png` produced; RTX 3090 roofline with Prefill (GEMM) and Decode (GEMV) marked |

---

## Milestone M8 — Economic Analysis Done

### Phase 4 — Economics & Concept Analysis

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 4.1 | Count input + output tokens per workload run | P0 | Done | Both | `prompt_tokens` and `n_output_tokens` recorded in every `results/*.json`; loaded by `run_economics.py` |
| 4.2 | Compute API cost/request using real provider price (cite provider + date) | P0 | Done | Both | GPT-4o: $2.50/M input + $10.00/M output (2026-06-21); cost=$0.002050/request; `src/economics/costs.py` |
| 4.3 | Compute on-prem CAPEX amortized + OPEX (electricity) cost/request | P0 | Done | Both | RTX 3090 $1500/36mo=$41.67/mo CAPEX + $0.02127/req electricity; `results/economics.json` |
| 4.4 | Plot break-even: cumulative cost vs volume for API and On-Prem | P0 | Done | Both | `figures/break_even.png` — no break-even exists (AirLLM energy > API per-token cost); annotated |
| 4.5 | Note context caching (PagedAttention) effect on API break-even | P1 | Done | Both | `reports/concept_analysis.md` §4.5 — PagedAttention shared prefix caching reduces API cost 61% for repeated context |
| 4.6 | Write recommendation: when API wins, when On-Prem wins; include privacy/data-security argument | P0 | Done | Both | `reports/concept_analysis.md` §4.6 — cost, privacy, GDPR, volume thresholds all addressed |
| 4.7 | Prefill vs Decode analysis tied to actual TTFT vs TPOT measurements | P0 | Done | Both | `reports/concept_analysis.md` §4.7 — GEMM/GEMV, compute-bound vs. memory-BW-bound, tied to measured TTFT/TPOT table |
| 4.8 | AirLLM-as-paging concept analysis: map layer loading to mmap, page table, page faults, locality | P0 | Done | Both | `reports/concept_analysis.md` §4.8 — full analogy table + SafeTensors flat-buffer mmap explanation |
| 4.9 | VRAM/RAM bottleneck and quantization effects analysis tied to results | P0 | Done | Both | `reports/concept_analysis.md` §4.9 — FP16/Q8/Q4/Q2 effects on VRAM/TPOT/quality from data; red line at Q2 |
| 4.10 | Concept → demonstration map table | P0 | Done | Both | `reports/concept_analysis.md` §4.10 — 16-row table mapping every L08 concept to file/figure/section |

---

## Milestone M9 — Extension Complete

### Phase 5 — Original Extension (pick 1–2)

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 5.1 | Choose 1–2 extensions | P1 | **Done** | Both | **Chosen: (A) multi-model size sweep + (B) paging instrumentation.** Pareto & roofline already done in Phase 3; LoRA rejected (scope). |
| 5.2 | Run extension experiment(s) | P1 | **Done** | Both | `results/extension_size_*.json` + `results/extension_paging_trace.json` |
| 5.3 | Generate extension figures | P1 | **Done** | Both | `figures/extension_size_comparison.png` + `figures/extension_paging.png` |
| 5.4 | Document extension findings with write-up | P1 | **Done** | Both | `reports/extension.md` written; both figures embedded in README |

---

## Milestone M10 — Submission Ready

### Phase 6 — Technical Report, README & Final QA

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 6.1 | Write deep-dive technical report in `reports/` | P0 | Not Started | Both | Covers: hardware spec + model justification, baseline failure, AirLLM+quant results with all metrics/tables/graphs, economic analysis with break-even, concept analysis, extensions; narrative explains results via theory |
| 6.2 | Write `README.md` for external reader | P0 | Not Started | Both | All figures embedded; reproduction instructions copy-pasteable; all 6 research questions answered or referenced |
| 6.3 | Explicitly answer all 6 research questions somewhere in the report | P0 | Not Started | Both | (1) Bottleneck identification; (2) AirLLM resource allocation + paging map; (3) Quantization red line; (4) Prefill/Decode in TTFT/TPOT data; (5) Latency/throughput price paid; (6) Local vs API economic crossover |
| 6.4 | Achieve ≥ 85% test coverage | P0 | Done | Both | 90% line coverage across all `src/` modules (65 tests passing) |
| 6.5 | **FINAL QA GATE** — verify §20.9 checklist: docs, code (≤150 lines/file), config, testing, research, visualization, costs, extension, git hygiene | P0 | Not Started | Both | Every checklist item confirmed; nothing failing |
| 6.6 | Scrub HF token from git history; verify no secrets committed | P0 | Not Started | Both | `git log` + `git grep HF_TOKEN` + `git grep hf_token` show no secret in history |
| 6.7 | Final commit and submit repo link | P0 | Not Started | Both | Submission link sent; repo accessible to grader |
