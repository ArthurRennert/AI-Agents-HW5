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
| 1.8 | Profile and record machine hardware specs | P0 | Done | Both | CPU, RAM, GPU, storage recorded in README and MODEL_SELECTION.md; `results/hardware.json` generated at runtime by `src/hardware/profiler.py` (RTX 3090 / 24 GB VRAM / Ryzen 9 5950X / 32 GB RAM / NVMe) |

---

## Milestone M2 — Environment Ready

### Phase 1 — Environment & Model Setup

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 1.9 | Create isolated env with `uv venv` | P0 | Done | Both | `uv venv` created; `uv.lock` and `pyproject.toml` present |
| 1.10 | Pin Python version | P0 | Done | Both | `.python-version` = 3.12; noted in README; `pyproject.toml` requires `>=3.11,<3.13` |
| 1.11 | Install all deps: `airllm`, `transformers`, `accelerate`, `torch` (CUDA cu128), `psutil`, `pandas`, `matplotlib`, `bitsandbytes`, `python-dotenv`, `sentencepiece` | P0 | Done | Both | All deps pinned in `pyproject.toml`; `transformers==4.44.2` (AirLLM-2.11 compatible); `torch>=2.11.0` from cu128 index for the RTX 3090 |
| 1.12 | Put HF token in `.env`; create `.env.example` with placeholder | P0 | Done | Both | `.env` in `.gitignore` (verified: never committed); `.env.example` committed with `HF_TOKEN=` placeholder |
| 1.13 | Select and verify model: `Qwen2.5-14B-Instruct` — confirm param count, format (SafeTensors), on-disk size | P0 | Done | Both | ~14.7B params; SafeTensors; ~29 GB FP16; "truck vs motorcycle" reasoning (29 GB > 24 GB VRAM) in README + MODEL_SELECTION.md |
| 1.14 | Verify free disk ≥ model size; set `layer_shards_saving_path` to NVMe | P0 | Done | Both | ~231 GB free confirmed; `SHARD_PATH=C:/airllm_shards` set in `.env` (NVMe) |

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
| 2.7 | Standardize workload: fix prompt, seed=42, `max_new_tokens=50`, discard warm-up | P0 | Done | Both | Fixed defaults in `src/config/settings.py`; all runs use 50 output tokens |
| 2.8 | Persist every raw metric to `results/<scenario>.json` tagged by engine + quant | P0 | Done | Both | `save_result()` / `load_results()` in `src/benchmark/persistence.py` |
| 2.9 | Smoke test harness on tiny model + low token count | P0 | Done | Both | `experiments/smoke_test.py` — uses gpt2, no auth needed, completes in < 2 min (artifact not part of final results) |

---

## Milestone M4 — Baseline Captured

### Phase 2 — Baseline Execution

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 2.10 | Run `Qwen2.5-14B-Instruct` directly via HF `transformers` (no AirLLM) | P0 | Done | Both | `experiments/run_baseline.py` (`device_map="cuda"`); outcome: 14B (~29 GB) exceeds 24 GB VRAM |
| 2.11 | Capture failure evidence (memory monitor output) | P0 | Done | Both | Peak VRAM 29.6 GB recorded in `results/baseline_fp16.json`; no hard crash — driver spills overflow to shared RAM, so there is no `error.txt` (the evidence is the VRAM-overflow + 44 s TTFT, not an exception) |
| 2.12 | Diagnose bottleneck: memory (VRAM) or compute? Show identification method | P0 | Done | Both | Root cause: **VRAM** — peak 29.6 GB > 24 GB ceiling; identified via peak-VRAM monitor + 44 s first-token latency cliff from driver sysmem fallback; compute never the limiter |
| 2.13 | Lock baseline as reference point for all comparisons | P0 | Done | Both | `results/baseline_fp16.json` is the reference; all AirLLM results compare against it |

---

## Milestone M5 — AirLLM Running

### Phase 3 — AirLLM Integration

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 3.1 | Wire AirLLM into runner; use `AutoModel` for Qwen | P0 | Done | Both | `src/runners/airllm_runner.py` with `TextIteratorStreamer`; compat shim in `src/runners/airllm_compat.py`; `use_cache=False` for transformers-4.44.2 path |
| 3.2 | Run standard workload through AirLLM at FP16; collect full metric set | P0 | Done | Both | **Real run** on RTX 3090 → `results/airllm_fp16.json` (peak VRAM 2.38 GB, ~37 s/token, coherent) |
| 3.3 | Confirm `layer_shards_saving_path` on NVMe; watch Disk I/O during run | P0 | Done | Both | Shards split to `C:/airllm_shards` (51 per-layer SafeTensors); per-token latency dominated by NVMe reads |

---

## Milestone M6 — Quantization Sweep Done

### Phase 3 — Quantization Experiments

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 3.4 | Run Q8 quantization experiment | P0 | Done | Both | **Real run** → `results/airllm_q8.json` (peak VRAM 3.16 GB, ~13 s/token, coherent) |
| 3.5 | Run Q4 quantization experiment | P0 | Done | Both | **Real run** → `results/airllm_q4.json` (peak VRAM 3.93 GB, ~11 s/token, coherent) |
| 3.6 | Run Q2 quantization experiment | P2 | N/A | Both | **Not available** — AirLLM's bitsandbytes `compression` supports only 8-bit/4-bit; 2-bit cannot be requested. Engine quantization floor = Q4. Documented in report §5/§9 |
| 3.7 | Identify and document accuracy "red line" | P0 | Done | Both | Output coherent at every available level down to the Q4 floor; the red line was **not reached** (Q2 unavailable). Documented in report §5 |

---

## Milestone M7 — Graphs Generated

### Phase 3 — Data Aggregation & Visualization

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 3.8 | Aggregate all `results/*.json` with pandas into tidy tables | P0 | Done | Both | `load_results_as_df()` in `src/viz/plots.py`; used in `experiments/generate_figures.py` (4 real records: baseline + FP16/Q8/Q4) |
| 3.9 | Generate TTFT, TPOT, throughput comparison chart | P0 | Done | Both | `figures/latency_comparison.png` |
| 3.10 | Generate peak RAM / VRAM comparison chart | P0 | Done | Both | `figures/memory_comparison.png`; report headline figure `figures/results_vram_comparison.png` |
| 3.11 | Generate energy comparison chart | P1 | Done | Both | `figures/energy_comparison.png` |
| 3.12 | Generate quality-vs-memory-vs-speed Pareto chart | P0 | Done | Both | `figures/pareto.png`; coherent across all real levels (no incoherent point — Q2 unavailable) |
| 3.13 | Generate pipeline architecture diagram for README | P1 | Done | Both | `figures/architecture.png` |
| 3.14 | (Advanced) Sketch Model Roofline: FLOPS vs arithmetic intensity; mark Prefill vs Decode | P2 | Done | Both | `figures/roofline.png`; RTX 3090 roofline with Prefill (GEMM) and Decode (GEMV) marked |

---

## Milestone M8 — Economic Analysis Done

### Phase 4 — Economics & Concept Analysis

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 4.1 | Count input + output tokens per workload run | P0 | Done | Both | `prompt_tokens` and `n_output_tokens` recorded in every `results/*.json`; loaded by `run_economics.py` |
| 4.2 | Compute API cost/request using real provider price (cite provider + date) | P0 | Done | Both | GPT-4o: $2.50/M input + $10.00/M output (2026-06-21); **$0.000963/request** (37 in + 87 out); `src/economics/costs.py` |
| 4.3 | Compute on-prem CAPEX amortized + OPEX (electricity) cost/request | P0 | Done | Both | RTX 3090 $1000/36mo = $27.78/mo CAPEX + $0.0166/req electricity (0.0258 kWh × $0.64/kWh); `results/economics.json` |
| 4.4 | Plot break-even: cumulative cost vs volume for API and On-Prem | P0 | Done | Both | `figures/break_even.png` — **no break-even** (API always cheaper per request); annotated |
| 4.5 | Note context caching (PagedAttention) effect on API break-even | P1 | Done | Both | `reports/technical_report.md` §6 — shared-prefix caching lowers effective API cost, pushing break-even further toward API |
| 4.6 | Write recommendation: when API wins, when On-Prem wins; include privacy/data-security argument | P0 | Done | Both | `reports/technical_report.md` §6 — API wins on cost; on-prem wins on privacy/data sovereignty + latency-tolerant batch |
| 4.7 | Prefill vs Decode analysis tied to actual TTFT vs TPOT measurements | P0 | Done | Both | `reports/technical_report.md` §7 — memory-bound regime; `use_cache=False` ⇒ TTFT ≈ TPOT, both NVMe-BW-bound |
| 4.8 | AirLLM-as-paging concept analysis: map layer loading to mmap, page table, page faults, locality | P0 | Done | Both | `reports/technical_report.md` §7 — full analogy table + SafeTensors flat-buffer mmap explanation |
| 4.9 | VRAM/RAM bottleneck and quantization effects analysis tied to results | P0 | Done | Both | `reports/technical_report.md` §5/§7 — FP16/Q8/Q4 effects on VRAM/latency from real data; Q2 unavailable |
| 4.10 | Concept → demonstration map table | P0 | Done | Both | `reports/technical_report.md` §7 — table mapping each L08 concept to file/figure/section |

---

## Milestone M9 — Extension Complete

### Phase 5 — Original Extension (pick 1–2)

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 5.1 | Choose 1–2 extensions | P1 | Done | Both | Chosen: (A) multi-model size sweep + (B) paging instrumentation. Pareto & roofline already done in Phase 3; LoRA rejected (scope). |
| 5.2 | Run extension experiment(s) | P1 | Done | Both | `results/extension_size_*.json` + paging trace |
| 5.3 | Generate extension figures | P1 | Done | Both | `figures/extension_size_comparison.png` + `figures/extension_paging.png` |
| 5.4 | Document extension findings with write-up | P1 | Done | Both | `reports/extension.md` written; both figures embedded in README |

---

## Milestone M10 — Submission Ready

### Phase 6 — Technical Report, README & Final QA

| # | Task | Priority | Status | Owner | Definition of Done |
|---|------|----------|--------|-------|--------------------|
| 6.1 | Write deep-dive technical report in `reports/` | P0 | Done | Both | `reports/technical_report.md` — hardware + model justification, baseline VRAM failure, AirLLM+quant results with metrics/tables/figures, economic analysis with break-even, concept analysis, extensions; results explained via theory |
| 6.2 | Write `README.md` for external reader | P0 | Done | Both | Figures embedded; copy-pasteable reproduction; all 6 research questions answered/referenced |
| 6.3 | Explicitly answer all 6 research questions in the report | P0 | Done | Both | `reports/technical_report.md` §8 — (1) VRAM bottleneck; (2) AirLLM↔paging; (3) quantization, Q4 floor; (4) Prefill/Decode in TTFT/TPOT; (5) latency/throughput price; (6) local vs API crossover |
| 6.4 | Achieve ≥ 85% test coverage | P0 | Done | Both | 91% line coverage; 99 tests passing (`uv run --extra dev pytest -q`) |
| 6.5 | **FINAL QA GATE** — verify §20.9 checklist: docs, code (≤150 lines/file), config, testing, research, visualization, costs, extension, git hygiene | P0 | Done | Both | All items confirmed; 99 tests green; figures + report + economics from real data |
| 6.6 | Scrub HF token from git history; verify no secrets committed | P0 | Done | Both | `git log --all --full-history -- .env` → empty (`.env` never committed); no secret in history |
| 6.7 | Final commit and submit repo link | P0 | In Progress | Both | Commit + push, then submission link sent; repo accessible to grader |
