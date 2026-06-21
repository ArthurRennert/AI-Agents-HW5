# PROMPTS.md — Agent Guidelines & Per-Module Prompt Book

> This document defines the rules given to AI coding agents for this project, plus the exact per-module prompts used to generate code. It satisfies §2.5 ("define guidelines for the agents") and incorporates the course's AI-agent software submission recommendations.

---

## Part 1 — Modular Architecture Rules (given to every agent session)

These rules are included at the top of every agent prompt. They are non-negotiable.

```
ARCHITECTURE RULES — follow these in every file you write:

1. FILE SIZE: Each .py file must be ≤ 150 lines. If a module exceeds this, split it.
2. DOCSTRINGS: Every public function, class, and module must have a one-line docstring
   (or multi-line if the purpose is non-obvious). Private helpers (_name) may omit it.
3. TYPE HINTS: All function signatures must include parameter and return type hints.
4. SEPARATION OF CONCERNS: One responsibility per module.
   - runners/ → execution only (no metric logic, no visualization)
   - benchmark/ → metric collection only (no model loading, no I/O)
   - viz/ → plotting only (reads from results/, writes to figures/)
   - economics/ → cost math only
   - config/ → settings only (no business logic)
5. NO SECRETS IN CODE: Never hard-code HF_TOKEN, API keys, or passwords.
   Always load them from environment variables via python-dotenv.
6. TEST COVERAGE: Every module you write must be accompanied by a test file in tests/.
   Target ≥ 85% line coverage (enforced by pytest-cov).
7. CONSISTENT STYLE: Follow PEP 8. Use ruff for linting. No unused imports.
8. ERROR HANDLING: Raise specific, descriptive exceptions at system boundaries
   (missing files, auth errors, disk space). Do not silently swallow errors.
9. NO GLOBAL STATE: Pass configuration explicitly via function arguments or dataclasses.
   Do not use module-level mutable globals.
10. RESULTS ARE IMMUTABLE: Never modify files under results/. Write once; read many times.
```

---

## Part 2 — Course AI-Agent Software Submission Recommendations

The following requirements come from the course's AI-agent submission guidelines and are graded explicitly:

- **Docs first:** Planning documents (PRD, PLAN, TODO, mini-PRDs) must exist and be approved *before* any production code is written. The AI agent should be given these documents as context.
- **Agent prompts are reviewable:** Every prompt given to an AI agent that produced committed code must be recorded here (Part 3 of this document). The grader may check that the code matches the prompts.
- **Critical review:** Do not blindly accept AI-generated code. Review every output for correctness, style, and security before committing. Note any manual fixes made.
- **Modular by default:** AI agents tend to produce monolithic code. Always ask for the smallest possible unit — one function, one class — and integrate manually.
- **Test together:** Ask the agent to write the test file alongside the implementation, not after.
- **No hard-coded values:** Remind the agent of this rule in every prompt. Agents frequently hard-code model names, paths, and tokens.
- **Vibe-coding warning:** Long manual debug loops with the agent are a signal that the prompt was too vague. Invest more time in precise specs; let the agent do the mechanical work.

---

## Part 3 — Per-Module Prompts

### Module: `src/config/settings.py`

```
Write a Python module src/config/settings.py (≤ 150 lines).

It must define a Settings dataclass with these fields:
  - model_name: str = "Qwen/Qwen2.5-32B-Instruct"
  - shard_path: str  (read from env var SHARD_PATH, no default — must be set)
  - quant_level: str = "fp16"  (one of: fp16, q8, q4, q2)
  - seed: int = 42
  - max_new_tokens: int = 200
  - prompt: str = (a fixed ~40-token English test prompt)
  - hf_token: str  (read from env var HF_TOKEN via python-dotenv — never hard-coded)

Add a load_settings() function that instantiates Settings from environment variables
and validates that quant_level is one of the allowed values (raise ValueError if not).

Rules: ≤ 150 lines, type hints, one-line docstrings on all public symbols, no secrets.
Also write tests/test_settings.py covering: valid load, invalid quant_level, missing HF_TOKEN.
```

---

### Module: `src/hardware/profiler.py`

```
Write a Python module src/hardware/profiler.py (≤ 150 lines).

It must define a HardwareProfile dataclass and a function profile_hardware() → HardwareProfile
that collects:
  - cpu_model: str (from platform or psutil)
  - cpu_cores: int (physical cores)
  - cpu_threads: int (logical cores)
  - ram_gb: float (total system RAM via psutil)
  - gpu_model: str (from torch.cuda.get_device_name(0), or "none" if no GPU)
  - vram_gb: float (from torch.cuda.get_device_properties(0).total_memory / 1e9, or 0.0)
  - storage_type: str ("NVMe" or "SSD" — detect via psutil.disk_partitions or hardcode based on known hardware)
  - free_disk_gb: float (free space on the shard_path drive via shutil.disk_usage)

Add a save_hardware_profile(profile, path) function that writes the profile as JSON.
Add a check_disk_space(shard_path, required_gb) function that raises DiskSpaceError if insufficient.

Rules: ≤ 150 lines, type hints, docstrings, no hard-coded values.
Write tests/test_hardware.py: mock psutil and torch.cuda; test check_disk_space with pass and fail cases.
```

---

### Module: `src/benchmark/harness.py`

```
Write a Python module src/benchmark/harness.py (≤ 150 lines).

It must define a Harness class with these methods:
  - start() → None: record t0 using time.perf_counter(); start background threads
    that sample psutil.Process().memory_info().rss every 0.5s and
    poll nvidia-smi for GPU power every 1s.
  - record_first_token() → None: called immediately when the first output token is emitted.
    Computes ttft_ms = (time.perf_counter() - t0) * 1000.
  - stop(n_output_tokens: int) → Metrics: stop background threads; compute and return
    a Metrics dataclass with fields:
      ttft_ms, tpot_ms, throughput_tokens_per_sec,
      peak_ram_gb, peak_vram_gb, wall_clock_sec, energy_wh.
    - tpot_ms = (wall_clock_sec * 1000 - ttft_ms) / max(n_output_tokens - 1, 1)
    - throughput = n_output_tokens / wall_clock_sec
    - peak_ram_gb = max of sampled RSS values / 1e9
    - peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 (0.0 if no CUDA)
    - energy_wh = mean_power_w * wall_clock_sec / 3600

Also define a Metrics dataclass. Persist nothing here — the runner calls save_result().

Rules: ≤ 150 lines, type hints, docstrings, thread-safe sampling.
Write tests/test_harness.py: mock time, psutil, torch.cuda; verify all metric computations.
```

---

### Module: `src/benchmark/persistence.py`

```
Write a Python module src/benchmark/persistence.py (≤ 150 lines).

It must define:
  - ResultRecord dataclass: fields for scenario (engine, model, quant_level, prompt_tokens,
    max_new_tokens, seed), metrics (all Metrics fields), output (generated_text,
    n_output_tokens, quality_note), and timestamp (ISO 8601 string).
  - save_result(record: ResultRecord, results_dir: str) → Path: serialize to JSON and write
    to results/<engine>_<quant_level>.json. Return the written path.
  - load_results(results_dir: str) → list[ResultRecord]: load all *.json files in results_dir.

quality_note must be one of: "coherent", "minor_degradation", "incoherent".

Rules: ≤ 150 lines, type hints, docstrings, no silent overwrites (warn if file exists).
Write tests/test_persistence.py: round-trip save/load; invalid quality_note raises ValueError.
```

---

### Module: `src/quantization/levels.py`

```
Write a Python module src/quantization/levels.py (≤ 150 lines).

It must define:
  - get_bnb_config(quant_level: str) → BitsAndBytesConfig | None:
    - "fp16" → None (no bitsandbytes config needed)
    - "q8"   → BitsAndBytesConfig(load_in_8bit=True)
    - "q4"   → BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
    - "q2"   → BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, ...)  [closest available]
    - anything else → raise ValueError listing valid options
  - estimate_disk_gb(quant_level: str, param_billions: float) → float:
    Bits-per-weight: fp16=16, q8=8, q4=4, q2=2. Formula: params * bits / 8 / 1e9.
  - estimate_vram_gb(quant_level: str, param_billions: float, n_layers: int) → float:
    Estimate VRAM for one layer: estimate_disk_gb / n_layers.

Rules: ≤ 150 lines, type hints, docstrings.
Write tests/test_levels.py: test each quant level config; test estimate functions; test ValueError.
```

---

### Module: `src/runners/airllm_runner.py`

```
Write a Python module src/runners/airllm_runner.py (≤ 150 lines).

It must define run_experiment(settings: Settings) → ResultRecord:
  1. Load HF_TOKEN from settings (already loaded via python-dotenv).
  2. Pre-flight: call check_disk_space(settings.shard_path, estimate_disk_gb(...)).
  3. Get bnb_config = get_bnb_config(settings.quant_level).
  4. Load model using AirLLM's AutoModel (not LlamaForCausalLM) with compression=bnb_config.
  5. Set torch.manual_seed(settings.seed).
  6. Tokenize settings.prompt; count prompt_tokens.
  7. Start harness: harness.start()
  8. Generate tokens in a streaming loop; call harness.record_first_token() on the first token.
  9. Call harness.stop(n_output_tokens) → metrics.
  10. Decode generated_text.
  11. Build and return a ResultRecord (quality_note left as "coherent" — human fills in later).

Do not print anything during generation (no tqdm, no progress bars).
Rules: ≤ 150 lines, type hints, docstrings, no hard-coded model name or paths.
Write tests/test_airllm_runner.py: mock AirLLM, harness, and disk check; test token counting.
```

---

### Module: `src/runners/baseline_runner.py`

```
Write a Python module src/runners/baseline_runner.py (≤ 150 lines).

It must define run_baseline(settings: Settings) → ResultRecord:
  1. Load model via AutoModelForCausalLM.from_pretrained(settings.model_name,
     torch_dtype=torch.float16, device_map="cuda"). Wrap in try/except RuntimeError/torch.cuda.OutOfMemoryError.
  2. On OOM: write a result JSON with error="OOM" and partial metrics (peak RAM/VRAM at time of failure).
  3. On success: run inference (unlikely for 32B, but handle it) and return full metrics.
  4. Save error evidence (exception message, traceback) as a text file in results/baseline_failure/.

Rules: ≤ 150 lines, type hints, docstrings, catch OOM gracefully.
Write tests/test_baseline_runner.py: mock OOM; verify error JSON is written.
```

---

### Module: `src/economics/analysis.py`

```
Write a Python module src/economics/analysis.py (≤ 150 lines).

It must define:
  - EconomicsInputs dataclass: all fields listed in docs/prd/economics.md (Section 2, Inputs).
  - EconomicsResult dataclass: all output fields listed in docs/prd/economics.md (Section 2, Outputs).
  - compute_costs(inputs: EconomicsInputs) → EconomicsResult: implement the formulas from
    docs/prd/economics.md Section 1 (api_cost, capex_per_request, opex_per_request, break_even).
  - save_economics(result: EconomicsResult, path: str) → None: write JSON.

Validate all required inputs; raise ValueError with a descriptive message if any is None or negative.
Rules: ≤ 150 lines, type hints, docstrings, no pyplot calls (visualization is in src/viz/).
Write tests/test_economics.py: test break-even formula; test validation; test JSON round-trip.
```

---

### Module: `src/viz/plots.py`

```
Write a Python module src/viz/plots.py (≤ 150 lines).

It must define these functions (each saves a file to figures/ and returns the Path):
  - plot_latency_comparison(records: list[ResultRecord], figures_dir: str) → Path
    Bar chart: TTFT and TPOT for each (engine × quant) scenario.
  - plot_memory_comparison(records: list[ResultRecord], figures_dir: str) → Path
    Grouped bar chart: peak_ram_gb and peak_vram_gb per scenario.
  - plot_pareto(records: list[ResultRecord], figures_dir: str) → Path
    Scatter plot: x=peak_vram_gb, y=tpot_ms, color/label=quant_level.
    Annotate the point identified as the accuracy "red line".
  - plot_break_even(result: EconomicsResult, figures_dir: str) → Path
    Line chart: cumulative API cost and on-prem cost vs. number of requests.
    Mark the break-even volume with a vertical dashed line.

Use matplotlib only. No interactive/notebook-only APIs. Save as PNG at 150 dpi.
Rules: ≤ 150 lines, type hints, docstrings.
Write tests/test_plots.py: use matplotlib's Agg backend; verify output files are created.
```
