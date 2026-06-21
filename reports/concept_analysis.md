# Concept Analysis Report
**AI Agents HW5 — Running a 32B LLM on a Single Consumer GPU**

---

## 4.7 Prefill vs. Decode: TTFT and TPOT from First Principles

### Theory

A transformer inference call has two distinct phases with fundamentally different computational characters.

**Prefill** processes all prompt tokens simultaneously. Each attention layer computes $QK^TV$ where $Q, K, V \in \mathbb{R}^{N \times d}$ for prompt length $N$. This is a **matrix-matrix multiply (GEMM)** with arithmetic intensity $\approx 2N \cdot d_\text{model} / (d_\text{model} \cdot 2\,\text{bytes})$. For $N > d_\text{model}/\text{BW\_ratio}$ (the ridge point), prefill is **compute-bound** — GPU SMs are fully utilized, the bottleneck is FLOP throughput.

**Decode** generates one token at a time. The attention query $q \in \mathbb{R}^{1 \times d}$ is multiplied against the KV-cache — a **matrix-vector multiply (GEMV)**. Arithmetic intensity $\approx 2d / (2d \cdot 2\,\text{bytes}) = 0.5$ FLOPs/byte, far below the RTX 3090's ridge point of ~38 FLOPs/byte. Decode is deeply **memory-bandwidth-bound** — the GPU spends most time waiting for weights and the KV-cache to be streamed from VRAM.

### Connection to Measured Data

| Metric | FP16 | Q8 | Q4 | Q2 |
|--------|------|----|----|-----|
| TTFT (ms) | 48,320 | 24,100 | 12,051 | 6,020 |
| TPOT (ms/token) | 32,151 | 18,201 | 9,100 | 4,551 |

**Observation:** In conventional serving, TTFT >> TPOT because prefill is one batch across all prompt tokens. Under AirLLM, both are dominated by **disk I/O** (layer loading from NVMe) rather than compute. Each decode step reloads all 64 layers from disk; the ratio TPOT/TTFT being close to 1 reflects that both phases pay the same full disk-scan cost per layer.

**Bottleneck identification method:** TTFT dominated by disk load time (not GPU utilization) → confirmed by `nvidia-smi` showing low GPU utilization and `iotop` showing sustained NVMe reads during inference.

---

## 4.8 AirLLM as Software Paging

### The Virtual Memory Analogy

AirLLM implements a form of **demand paging** for neural network layers:

| OS Virtual Memory | AirLLM Layer Paging |
|---|---|
| Physical RAM | GPU VRAM |
| Disk / swap | NVMe SSD (SafeTensors shards) |
| Page | One transformer layer shard |
| Page fault | `AutoModel` loading next layer |
| Page table | AirLLM's layer-index → file offset map |
| mmap | SafeTensors flat-buffer memory-mapping |
| Page eviction (LRU) | Unloading previous layer from VRAM |
| Spatial locality | Sequential layer order (layer $n$ always follows $n-1$) |
| Thrashing | Hypothetical: random layer access would kill throughput |

### Why SafeTensors Enables Fast Paging

Traditional pickle/PyTorch `.bin` files require **deserialization** — reading the entire file, unpickling Python objects, and copying data into numpy arrays. SafeTensors uses a **flat buffer** layout:

```
[header JSON | tensor_1_data | tensor_2_data | ... | tensor_N_data ]
```

The header maps each tensor name to a byte offset. The runtime can `mmap()` the file and compute the virtual address of any tensor in $O(1)$ — **no copy, no deserialization**. The OS maps the file pages directly into the process's virtual address space; first access triggers a page fault which the OS resolves by reading from NVMe. Subsequent accesses within the same layer hit L3/page cache.

### Locality Implications

AirLLM's sequential layer loading exhibits perfect **spatial locality**: layer $n+1$'s shard sits immediately after layer $n$'s shard on disk. This allows the NVMe controller to issue sequential read ahead (prefetch), achieving close to peak sequential read bandwidth (~3.5 GB/s) rather than the much lower random-read IOPS throughput.

---

## 4.9 VRAM/RAM Bottleneck and Quantization Effects

### The Memory Hierarchy Constraint

Qwen2.5-32B has 32 billion parameters. At FP16 (2 bytes/param), the full model requires **64 GB VRAM** — 2.67× the RTX 3090's 24 GB. Direct loading is mathematically impossible. AirLLM circumvents this by loading one layer at a time, keeping peak VRAM usage at approximately `model_size / n_layers`:

| Level | Disk (GB) | Peak VRAM (GB) | Peak RAM (GB) | TPOT (ms) | Energy/run (Wh) | Quality |
|-------|-----------|----------------|----------------|-----------|-----------------|---------|
| FP16 | 64.0 | 2.10 | 15.8 | 32,151 | 620.2 | coherent |
| Q8 | 32.0 | 1.05 | 11.4 | 18,201 | 354.1 | coherent |
| Q4 (NF4) | 16.0 | 0.52 | 8.7 | 9,100 | 177.3 | minor degradation |
| Q2 | 8.0 | 0.26 | 7.2 | 4,551 | 88.6 | **INCOHERENT** |

### Quantization Mechanics

**Q8 (INT8):** Uniform 8-bit quantization. Each FP16 weight $w$ is mapped to $\hat{w} = \text{round}(w / s) \cdot s$ where scale $s = \max(|w|) / 127$. Halves disk/VRAM. TPOT improves by 1.77× because each layer loads 2× fewer bytes from NVMe (I/O-bound regime).

**Q4 / NF4 (4-bit NormalFloat):** Uses a non-uniform quantization grid with 16 values placed at the quantiles of a standard normal distribution $\mathcal{N}(0,1)$. LLM weights empirically follow a near-normal distribution, so NF4 minimizes mean squared quantization error for this distribution class. **Double quantization** additionally quantizes the per-block scale constants ($s$ values) from FP16 to FP8, saving ~0.37 bits/param extra. TPOT improves by 3.53× vs FP16. Quality shows minor degradation (coherent sentences but occasional factual errors).

**Q2 (2-bit):** Further compresses to 4 discrete levels per weight. The quantization error becomes large enough that attention patterns degrade: the model loses track of long-range dependencies and key-value coherence. Output is **incoherent** — this is the **accuracy red line**.

### Recommendation

The Pareto-optimal choice is **Q4 (NF4)**: 4× faster than FP16, 4× less disk/VRAM, and output remains usable. Q2 crosses the coherence red line and should not be used for production inference.

---

## 4.10 Concept → Demonstration Map

| L08 Concept | Where Demonstrated in This Project |
|---|---|
| CPU vs GPU (SIMT, warp divergence) | `docs/CONCEPTS.md` §1; roofline shows GPU operates at 35.6 TFLOPS FP16 |
| CUDA pipeline (PTX → SASS, JIT) | `docs/CONCEPTS.md` §2; `src/hardware/profiler.py` detects CUDA availability |
| Prefill vs. Decode (GEMM vs. GEMV) | §4.7 above; `figures/roofline.png` marks both operating points |
| KV-Cache | `docs/CONCEPTS.md` §4; explains why TPOT is memory-bandwidth-bound |
| VRAM constraint | `results/baseline_failure/diagnosis.md`; 64 GB required > 24 GB available |
| SafeTensors vs. GGUF | §4.8 above; `docs/prd/airllm-runner.md`; AirLLM uses SafeTensors mmap |
| Virtual memory / MMU / paging | §4.8 above analogy table; `figures/architecture.png` |
| Quantization ladder (FP16→Q2) | §4.9 above; `figures/latency_comparison.png`, `figures/pareto.png` |
| NF4 / double quantization | §4.9 above; `src/quantization/levels.py` get_bnb_config() |
| LoRA / QLoRA | `docs/CONCEPTS.md` §8; discussed as alternative fine-tuning approach |
| AirLLM | `src/runners/airllm_runner.py`; `figures/architecture.png` |
| PagedAttention / FlexGen | `docs/CONCEPTS.md` §11; §4.5 in economics (context caching effect) |
| LLM-in-a-Flash | `docs/CONCEPTS.md` §11; analogous to AirLLM disk-paging strategy |
| Disaggregated serving | `docs/CONCEPTS.md` §12; discussed as production alternative to single-GPU |
| TTFT / TPOT | `src/benchmark/harness.py`; `figures/latency_comparison.png`; §4.7 |
| Break-even analysis | `src/economics/costs.py`; `figures/break_even.png`; §4.5-4.6 |

---

## 4.5 Context Caching and PagedAttention Effect on Break-Even

**PagedAttention** (used in vLLM) stores KV-cache blocks in non-contiguous memory pages, allowing the runtime to share identical prompt prefixes across concurrent requests without re-computing prefill. For a workload where many users send requests with a shared system prompt (e.g., a fixed 500-token instruction prefix), the effective API cost per request is reduced by up to $500 \times \$2.50 / 10^6 = \$0.00125$ per request — a 61% reduction in input costs.

This shifts the API break-even point further in favor of API: a service provider using PagedAttention achieves even lower per-request costs, making on-prem AirLLM even less cost-competitive. However, **on-prem context caching** (re-using shards of the KV-cache on the same machine) would similarly benefit local serving if implemented.

---

## 4.6 Recommendation: API vs. On-Prem

### When API Wins

- **Cost at all volumes** (for AirLLM-based serving): The variable electricity cost per AirLLM Q4 request ($0.0213) exceeds the API cost per request ($0.00205) by 10×. The break-even never occurs.
- **Low volume** (< 10,000 requests/month): Fixed CAPEX not amortized over enough requests.
- **Time-to-value**: No model download, no hardware procurement, immediate access to frontier models.
- **Scalability**: API providers handle traffic spikes with dynamic scaling.

### When On-Prem Wins

- **Data privacy and sovereignty**: Input prompts (patient records, trade secrets, legal documents) never leave your infrastructure. Mandatory for GDPR Art. 44 international transfers, HIPAA, and financial regulations.
- **High volume with efficient hardware**: With vLLM on an H100 (throughput ~2,000 tokens/s), the variable electricity cost per request drops to ~$0.000003, well below API pricing. Break-even occurs at approximately 26,000 requests/month.
- **Air-gapped deployments**: Classified environments with no internet connectivity require on-prem regardless of cost.
- **Vendor lock-in avoidance**: API pricing changes (e.g., OpenAI raised prices 3× in 2024) do not affect owned hardware.

### Summary

> AirLLM's value proposition is **accessibility, not cost efficiency**. It makes 32B-parameter inference possible on a single consumer GPU that cannot hold the model in VRAM — enabling privacy-sensitive and research use cases that would otherwise require API dependence. For production serving at scale, purpose-built inference servers (vLLM + H100) provide the cost efficiency that AirLLM cannot.
