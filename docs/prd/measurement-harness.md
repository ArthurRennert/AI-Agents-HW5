# Mini-PRD: Measurement Harness

## 1. Description and Theoretical Background

The measurement harness is a lightweight instrumentation layer that wraps any model inference run to collect performance metrics consistently. It must work identically for both the baseline runner and the AirLLM runner so that all comparisons are fair.

**Key metrics and their theoretical meaning:**

| Metric | Definition | What it proxies |
|--------|-----------|----------------|
| **TTFT** (Time To First Token) | Wall-clock ms from request submission to first output token | **Prefill latency**: time to process the prompt, build the KV-cache, and complete the first forward pass. Compute-bound — GEMM-dominated (matrix × matrix over all prompt positions). |
| **TPOT / ITL** (Time Per Output Token / Inter-Token Latency) | Mean gap (ms) between consecutive output tokens after the first | **Decode latency**: each step reads the full KV-cache for one new position. Memory-bandwidth-bound — GEMV-dominated (matrix × vector for one token). |
| **Throughput** | Total output tokens / total wall-clock time | Aggregate production rate (tokens/sec); combines prefill + decode. |
| **Peak RAM** | Maximum resident set size (RSS) during the run | Whether system RAM is the binding constraint. |
| **Peak VRAM** | Maximum GPU memory allocated during the run | Whether GPU VRAM is the binding constraint. |
| **Wall-clock time** | Total elapsed time from request to final token | Practical latency as perceived by a user or calling application. |
| **Energy (Wh)** | Average power draw (W) × elapsed time (h) | Operational cost proxy; feeds directly into the economics analysis. |

**Why TTFT is compute-bound and TPOT is memory-bound:** During prefill, the full prompt is processed as a batched GEMM — all token positions computed in parallel, yielding high arithmetic intensity and saturating the GPU's FLOPS. During decode, each step performs a GEMV against the KV-cache (one row per layer per attention head), yielding low arithmetic intensity and saturating memory bandwidth instead. AirLLM amplifies the TPOT bottleneck: every decode step must reload all transformer layers from NVMe, so TPOT is further limited by disk I/O (NVMe sequential read: ~3–7 GB/s) in addition to memory bandwidth.

**Energy measurement method:** GPU power is sampled from `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` at 1 Hz in a background thread during the run. Mean power × elapsed hours gives Wh. For CPU-only or no-GPU paths, a TDP-fraction estimate is used and explicitly stated in the results JSON.

---

## 2. Specific Requirements

### Inputs

| Input | Type | Notes |
|-------|------|-------|
| `prompt` | `str` | Fixed across all scenarios (standardized workload) |
| `max_new_tokens` | `int` | Fixed (e.g., 200); same for every scenario |
| `seed` | `int` | Fixed (e.g., 42); passed to `torch.manual_seed` and `transformers.set_seed` |
| `engine` | `str` | `"baseline"` or `"airllm"` — for scenario tagging in results |
| `quant_level` | `str` | `"fp16"` \| `"q8"` \| `"q4"` \| `"q2"` |

### Outputs (per run)

| Output | Type | Unit | Notes |
|--------|------|------|-------|
| `ttft_ms` | `float` | ms | Time from request to first token |
| `tpot_ms` | `float` | ms/token | Mean inter-token latency after first token |
| `throughput_tokens_per_sec` | `float` | tokens/s | `n_output_tokens / wall_clock_sec` |
| `peak_ram_gb` | `float` | GB | Max RSS during run |
| `peak_vram_gb` | `float` | GB | Max GPU allocation during run (0.0 if no GPU) |
| `wall_clock_sec` | `float` | s | Total elapsed time |
| `energy_wh` | `float` | Wh | Estimated watt-hours consumed |

### Performance Requirements of the Harness Itself

- Sampling overhead: < 1% of total wall-clock time.
- Power sampling interval: 1 second (sufficient given AirLLM run durations of minutes to hours).
- Memory sampling interval: 500 ms (`psutil` RSS + `torch.cuda.max_memory_allocated`).

### Workload Standardization

- **Warm-up run:** One full run discarded before metric collection begins.
- **Fixed prompt:** A single multi-sentence English prompt (~40 tokens).
- **Fixed seed:** `seed=42` set before every run.
- **Fixed output length:** `max_new_tokens=200`.
- **No batching:** Single-sample inference only (`batch_size=1`).

---

## 3. Constraints and Limitations

- `torch.cuda.max_memory_allocated()` tracks only PyTorch-managed allocations; CUDA driver overhead may push actual VRAM usage slightly higher than what is reported.
- Energy estimation accuracy is ±10–20% for GPU due to 1 Hz sampling. CPU energy uses a TDP-fraction estimate and is noted as approximate.
- TTFT under AirLLM is far larger than under native GPU inference because the first token requires loading all N layers sequentially — this is expected behavior, not a measurement error.
- The harness stores per-token timestamps in memory; for very long outputs (> 1000 tokens) memory usage of the harness itself may become non-negligible.

---

## 4. Alternatives Considered

| Alternative | Reason Rejected |
|-------------|----------------|
| NVIDIA Nsight / GPU performance counters | Too invasive; requires root / driver-level access; out of scope for this project's complexity. |
| External wall-clock only (no power sampling) | Energy is an explicit required metric per the HW plan; must estimate it. |
| 100 ms power sampling interval | Overkill given run durations of minutes; 1 s is sufficient and reduces overhead. |
| Integrating with MLflow / WandB | Adds a heavy external dependency; simple JSON persistence is sufficient and reproducible. |

---

## 5. Success Criteria

- All 7 metric fields populated (non-zero, non-null) for every successfully completed run.
- Results persisted as valid JSON immediately after each run; file must be loadable by the analysis module without modification.
- Smoke test (tiny model + Q2 + 20 output tokens) completes in under 5 minutes with all metrics populated.
- For a baseline OOM run: harness catches the exception, records partial metrics (peak RAM at time of OOM), and writes a result JSON with an `error` field.

---

## 6. Specific Test Scenarios

| Test | Input | Expected Output |
|------|-------|----------------|
| Smoke test — tiny model, Q2, 20 tokens | Small model + fixed prompt | All 7 metrics populated; valid JSON written to `results/` |
| TTFT accuracy | Run on GPU with known model | TTFT within ±50 ms of a manual `time.perf_counter()` wrapper |
| TPOT accuracy | 10-token run; compare to manual per-token timestamps | TPOT within ±5 ms of manual measurement |
| OOM handling (baseline) | Trigger OOM by loading oversized model | Exception caught; partial JSON written with `error` field; no unhandled crash |
| No-GPU environment | Run without CUDA available | `peak_vram_gb = 0.0` written; no exception raised |
| Result persistence | Any completed run | JSON file present in `results/<scenario>.json` with correct engine + quant tags |
| Warm-up discard | Run twice; check only second run is persisted | Only one result JSON per scenario name; warm-up metrics discarded |
