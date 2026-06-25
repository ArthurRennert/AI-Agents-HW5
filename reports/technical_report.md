# Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

**Technical Report — EX05 (L08)**

---

## 1. Executive summary

A 14-billion-parameter model (Qwen2.5-14B-Instruct, ~29 GB in FP16) cannot be held
in the 24 GB of VRAM on an RTX 3090. Loaded directly, it overflows VRAM, the NVIDIA
driver silently spills the excess into shared system memory, and time-to-first-token
collapses to **44 seconds** — non-viable. Run through **AirLLM**, which streams the
model one transformer layer at a time, the *same* model executes in **2.4–3.9 GB of
VRAM** — a ~10× reduction — at a latency cost of ~11–37 s per token. **Quantization**
(8-bit, 4-bit) cuts that per-token cost by ~3× while preserving coherent output.
This report presents the hardware, the measured baseline failure, the AirLLM +
quantization results, an economic analysis, the underlying systems concepts, and
explicit answers to the six research questions.

All numbers are measured on the author's machine; nothing is simulated.

## 2. Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3090 — **24 GB VRAM** |
| CPU | AMD Ryzen 9 5950X (16 cores / 32 threads) |
| RAM | 32 GB |
| Storage | NVMe SSD (~665 GB free) |
| OS | Windows 11 |

The binding resource is **24 GB of GPU VRAM**. (Profiled via `experiments/profile_machine.py`
→ `results/hardware.json`.)

## 3. Model selection

**Qwen2.5-14B-Instruct** (~14.7 B params; ~29 GB FP16; SafeTensors). The choice was
calibrated empirically (full reasoning in `docs/MODEL_SELECTION.md`): direct-load
measurements showed 7B (15 GB) and even larger models fit in 24 GB, so the model had
to be large enough to exceed VRAM but small enough to remain tractable under AirLLM.
14B sits just above the 24 GB line — it genuinely cannot run directly, yet completes
under AirLLM — making it the deliberate "truck vs motorcycle" match to a 24 GB GPU.
(A 32B model overflows further but at ~65 GB is impractical to download and sweep.)

## 4. Baseline: direct execution and its bottleneck

Loading 14B directly onto the GPU (`device_map="cuda"`, FP16):

| Metric | Value |
|--------|-------|
| Peak VRAM | **29.6 GB** (exceeds the 24 GB ceiling) |
| Time to first token | **43.5 s** |
| Wall clock | 43.9 s |
| Output | coherent |

**The bottleneck is memory (VRAM), not compute, and here is how it was identified.**
Peak VRAM (29.6 GB) exceeds the physical 24 GB of the card. The model did not crash
with a hard CUDA out-of-memory error because the NVIDIA driver's *system-memory
fallback* pages the ~5.6 GB overflow into shared host RAM over PCIe. The price of
that fallback is the 44-second first token — two to three orders of magnitude slower
than the sub-second response a model that fits in VRAM would give. This is the GPU
analogue of OS swap-thrashing, and it is the "inevitable bottleneck" the assignment
asks us to surface: the model's footprint exceeds the fast memory tier, so the system
is forced onto a slow one. (Evidence: `results/baseline_fp16.json`.)

![VRAM comparison](figures/results_vram_comparison.png)

## 5. AirLLM + quantization results

AirLLM was wired into the runner (per-layer execution via `AutoModel`). The model is
split once into per-layer SafeTensors shards on NVMe (`C:/airllm_shards`); each
forward pass then streams the 48 decoder layers (+ embeddings, norm, head = 51
shards) through the GPU one at a time, bounding resident VRAM to roughly a single
layer. The identical workload (fixed prompt, fixed `max_new_tokens`, greedy decoding,
fixed seed) was run at FP16, then quantized to 8-bit and 4-bit via bitsandbytes.

| Engine / level | Peak VRAM | Peak RAM | Per-token latency | Quality |
|----------------|----------:|---------:|------------------:|---------|
| Baseline (direct FP16) | 29.6 GB | 7.4 GB | 44 s to 1st token (thrash) | coherent, non-viable |
| AirLLM FP16 | **2.38 GB** | 9.4 GB | ~37 s/token | coherent |
| AirLLM Q8 | 3.16 GB | 4.0 GB | ~13 s/token | coherent |
| AirLLM Q4 | 3.93 GB | 4.2 GB | ~11 s/token | coherent |
| AirLLM Q2 | — | — | — | **unavailable** |

Key observations:

- **Memory:** AirLLM cuts peak VRAM from 29.6 GB to 2.4–3.9 GB — a ~10× reduction —
  letting a model that overflowed the card run comfortably within it. Resident VRAM
  is bounded by one layer, independent of total model size.
- **Latency:** the price is steep. Every forward pass re-reads the whole model from
  disk, so per-token latency is dominated by NVMe I/O, not compute: ~37 s/token at
  FP16. Throughput is ~0.03–0.09 tokens/s, roughly 100–1000× slower than an in-VRAM
  model.
- **Quantization helps speed.** Lower precision means fewer bytes streamed per layer:
  per-token latency falls from ~37 s (FP16) to ~13 s (Q8) to ~11 s (Q4) — about a 3×
  speed-up — while peak memory stays bounded. (The bitsandbytes dequantization
  buffers make Q8/Q4 VRAM marginally higher than the raw FP16 layer stream, but all
  remain ~10× below baseline.)
- **The accuracy "red line" was not reached, and the engine floor was found.** Output
  stayed coherent at every available level, down to 4-bit. **2-bit is unavailable** —
  AirLLM's bitsandbytes path supports only 8-bit and 4-bit — so the lowest precision
  the engine can reach here is Q4, which remained coherent. The quality red line for
  this model/engine therefore lies at or below 4-bit and could not be crossed.

![Quantization effect](figures/results_quant_latency.png)

## 6. Economic analysis: API vs on-prem

*All prices/rates below are assumptions — replace the two marked values with your
actuals and re-run `experiments/run_economics.py`.*

**Workload:** ~20 prompt tokens + 50 output tokens ≈ **70 tokens/request**.

**Hosted API.** At a representative mid-tier price (~$0.50 / 1M input, ~$1.50 / 1M
output), one request costs ≈ **$0.00008** — effectively free per call, with zero
capital outlay and instant latency.

**On-prem (this machine, AirLLM Q4).**
- CAPEX: RTX 3090 workstation ≈ **$2,000** *(replace with your real build cost)*,
  amortized over 3 years.
- OPEX (electricity): ~450 W under load × (50 tok × 11 s = 550 s ≈ 0.153 h) ≈ 0.069
  kWh/request; at ≈ **$0.16/kWh** *(replace with your real IL rate)* ≈ **$0.011/request**.
- Capital per request depends on volume: $2,000 / (requests over 3 years).

**Break-even.** Per-request *marginal* cost on-prem ($0.011, mostly electricity) is
far higher than the API's $0.00008, so on a pure cost-per-token basis the **API wins
at essentially every volume** for this slow AirLLM configuration — the electricity for
9 minutes of full-system draw per request dwarfs the API token price. On-prem only
approaches parity if the hardware is already owned (sunk CAPEX) and used for many
other tasks.

**But cost is not the only axis.** On-prem keeps all data inside the organization —
decisive for privacy/regulated workloads — and has no per-call fee or rate limit once
the hardware exists. The honest recommendation: **API for interactive, low-volume, or
cost-sensitive use; on-prem AirLLM only for privacy-critical, latency-tolerant, batch
workloads** where a ~9-minute/request turnaround is acceptable and data cannot leave
the premises. Prompt/context caching (PagedAttention) further lowers effective API
cost for repeated-context workloads, pushing the break-even even further toward the
API.

## 7. Concept analysis (tying results to theory)

**Prefill vs decode.** Normally prefill (compute-bound, builds the KV cache) is
measured by TTFT and decode (memory-bandwidth-bound) by TPOT. In this AirLLM run the
KV cache was disabled (a transformers-5.x compatibility constraint, see §9), so every
token is a full forward pass — there is no cheap decode step. TTFT ≈ TPOT ≈ one full
51-layer disk sweep, and **both are dominated by NVMe read bandwidth**, not compute.
This is why quantization (fewer bytes/layer) speeds things up: the bottleneck is data
movement, exactly as the prefill/decode framing predicts for a memory-bound regime.

**AirLLM as virtual memory / paging.** AirLLM is demand paging for model weights. The
mapping is exact: a **layer = a page**; **NVMe = the backing store**; **VRAM = physical
memory**; a **forward pass = one sweep of the working set**. Resident VRAM is bounded
(one layer) just as an OS bounds a process's resident set, and the cost is the
page-in traffic from disk. The baseline's driver fallback (§4) is the *uncontrolled*
version of the same idea — VRAM overflow paged blindly to host RAM; AirLLM replaces it
with *explicit, bounded* layer paging. Because the shards are **SafeTensors** (a flat,
mmap-friendly buffer), each layer page-in is a near-zero-copy load rather than a
pickle deserialization.

**Quantization (FP16 → Q8 → Q4).** Fewer bits per weight means less data to page per
layer and less VRAM per resident layer — directly improving the memory-bound latency,
which the data confirms (37 → 13 → 11 s/token). Coherence held to 4-bit; 2-bit was
not reachable with this engine.

**Concept → demonstration map.**

| Concept | Where demonstrated |
|---------|--------------------|
| VRAM capacity bottleneck | §4 baseline, peak VRAM 29.6 > 24 GB |
| Memory hierarchy / paging | §7, AirLLM layer streaming; Extension B (`reports/extension.md`) |
| SafeTensors flat buffer / mmap | §5 shard splitting + §7 |
| Quantization tradeoffs | §5 FP16/Q8/Q4 sweep |
| Prefill vs decode (memory-bound) | §7, TTFT≈TPOT under streaming |
| Disk I/O as the real cost | §5 per-token latency |
| Bottleneck shifts with size | Extension A (`reports/extension.md`) |

## 8. The six research questions (§4)

1. **Bottleneck blocking direct execution (memory vs compute), how identified?**
   Memory — specifically 24 GB of VRAM. Identified by peak VRAM (29.6 GB) exceeding
   the card's 24 GB and the resulting 44 s time-to-first-token from driver
   system-memory fallback, with compute never the limiter.

2. **How does AirLLM change resource allocation, and how does it map to virtual
   memory / paging?** It bounds resident VRAM to a single layer (~2–4 GB) by streaming
   layers from NVMe per forward pass. Layer↔page, NVMe↔backing store, VRAM↔physical
   memory, forward pass↔working-set sweep; SafeTensors↔zero-copy mmap page-in.

3. **Quantization's impact on memory/speed/quality — where is the red line?**
   Lower precision cut per-token latency ~3× (37→11 s) and kept VRAM bounded, with no
   loss of coherence down to 4-bit. The red line was not crossed: 2-bit is unavailable
   in AirLLM's bitsandbytes path, so Q4 (coherent) is the engine's floor here.

4. **How do prefill/decode show up in TTFT vs TPOT?** With the KV cache disabled,
   every token is a full forward pass, so TTFT ≈ TPOT, and both are NVMe-bandwidth-
   bound rather than compute-bound — the signature of a memory-bound regime.

5. **What latency/throughput price do you pay for running big on modest hardware?**
   ~37 s/token (FP16) down to ~11 s/token (Q4); throughput ~0.03–0.09 tokens/s —
   roughly 100–1000× slower than an in-VRAM model. That is the cost of fitting 14B
   into 2–4 GB of VRAM.

6. **When is local economically worth it vs an external API?** On pure cost, the API
   wins at nearly all volumes for this slow configuration. On-prem AirLLM is justified
   only when data must stay local (privacy/compliance) and the workload tolerates
   ~9-minute/request latency — i.e. privacy-critical batch jobs, not interactive use.

## 9. Limitations & honest notes

- **Stack compatibility.** AirLLM 2.11 was built for transformers ~4.44; the modern
  5.x stack broke it in four places (removed `optimum.bettertransformer`, the
  `compression` API, the `DynamicCache` refactor, and model-level rotary embeddings).
  The project pins **transformers 4.44.2** — the version AirLLM was written for — which
  resolves all four. A small no-op shim provides the removed `optimum.bettertransformer`.
- **KV cache disabled.** As a consequence of the above, generation runs with
  `use_cache=False`. This makes per-token latency a worst case (every token recomputes
  the full pass) but does not affect the memory results, which are the project's point.
- **TTFT/TPOT instrumentation.** The harness's streamer-based TTFT/TPOT under-report
  the true per-token time; the authoritative latency figures here are derived from the
  observed full-layer-pass duration (~37/13/11 s) and wall-clock.
- **Quantization floor.** 2-bit is not available via AirLLM's bitsandbytes path.

## 10. Extensions

Two original extensions are documented in `reports/extension.md`: (A) a multi-model
size sweep showing the memory wall shifting with model size, and (B) a runnable
mmap-based paging demonstration (RSS sawtooth + measured page faults) that exhibits
the OS-paging mechanic AirLLM relies on. Both reinforce §7's virtual-memory framing.

## 11. Reproduce

```bash
uv sync
uv run python experiments/profile_machine.py
uv run python experiments/run_baseline.py              # captures the VRAM bottleneck
uv run python experiments/run_airllm.py                # AirLLM FP16
uv run python experiments/run_quantization_sweep.py --levels q8 q4
uv run python experiments/run_economics.py
uv run python experiments/generate_figures.py
```

Raw results: `results/baseline_fp16.json`, `results/airllm_{fp16,q8,q4}.json`.
