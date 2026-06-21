# Mini-PRD: AirLLM Runner

## 1. Description and Theoretical Background

The AirLLM runner executes a large language model that exceeds available GPU VRAM by splitting it into per-layer SafeTensors shards and loading one layer at a time into GPU memory. This directly maps to OS virtual memory / paging concepts:

**AirLLM as a paging system:**

| OS Virtual Memory Concept | AirLLM Equivalent |
|--------------------------|------------------|
| Pages stored on disk (swap) | Layer shards stored on NVMe SSD |
| Page fault → page load | Layer needed → shard read from NVMe into RAM → GPU VRAM |
| Page table | AirLLM's internal shard index (which layer is in which file) |
| Locality of reference | Sequential layer access (0 → N) during forward pass; high spatial locality |
| `mmap` (memory-mapped file I/O) | SafeTensors' flat buffer layout enables zero-copy `mmap`: OS maps the file into virtual address space without copying bytes |

**Why SafeTensors enables fast layer loads:** SafeTensors stores tensors in a flat binary format at a fixed byte offset per tensor. This allows `mmap`-based access: the OS maps the file into virtual address space, and the tensor data is read into physical memory (and then GPU VRAM) only when accessed — a zero-copy load path. This is fundamentally different from pickle-based formats (e.g., old PyTorch `.pt` files), which require a full deserialization pass.

**Why `AutoModel` is required for Qwen:** The `Qwen2.5` architecture uses `AutoModelForCausalLM` internally, not `LlamaForCausalLM` or `MistralForCausalLM`. Passing the wrong class to AirLLM's loader causes a class-mismatch error at weight assignment time. AirLLM's `AutoModel` wrapper resolves the correct class from the model's `config.json` automatically.

**Layer-by-layer execution flow:**
1. Split model weights into one SafeTensors shard per transformer layer (done once; cached on NVMe).
2. For each forward pass:
   - Load shard i from NVMe → system RAM → GPU VRAM.
   - Compute layer i on GPU.
   - Offload shard i from VRAM (free GPU memory).
   - Load shard i+1.
3. Repeat for all N layers.
4. Sampling head generates one token; repeat for `max_new_tokens` decode steps.

**The dominant bottleneck under AirLLM:** NVMe sequential read speed (~3–7 GB/s). Every decode step reads the full model (~65 GB FP16 or ~16 GB Q4) from disk. This, not compute or VRAM bandwidth, caps throughput.

---

## 2. Specific Requirements

### Inputs

| Input | Type | Notes |
|-------|------|-------|
| `model_name` | `str` | HuggingFace model ID, e.g. `"Qwen/Qwen2.5-32B-Instruct"` |
| `prompt` | `str` | Input text (fixed across scenarios) |
| `max_new_tokens` | `int` | Maximum output tokens (e.g., 200) |
| `seed` | `int` | Reproducibility seed (e.g., 42) |
| `quant_level` | `str` | `"fp16"` \| `"q8"` \| `"q4"` \| `"q2"` |
| `shard_path` | `str` | Absolute path to NVMe directory for layer shards |
| `hf_token` | `str` | Loaded from `.env` via `python-dotenv`; never hard-coded |

### Outputs

| Output | Type | Notes |
|--------|------|-------|
| `generated_text` | `str` | Model's full response |
| `n_output_tokens` | `int` | Actual number of tokens generated |
| `metrics` | `Metrics` | Full metric set from the harness (TTFT, TPOT, throughput, RAM, VRAM, time, energy) |

### Performance Metrics

| Metric | Hard Constraint | Expected Range (FP16) |
|--------|----------------|----------------------|
| Peak VRAM | ≤ 24 GB | ~22–24 GB (one layer at a time) |
| Peak RAM | ≤ 128 GB | ~40–60 GB |
| TPOT | No hard limit | ~5,000–15,000 ms/token (disk-I/O-bound) |
| Disk space (shards) | ≥ model size | ~65 GB FP16 / ~16 GB Q4 |

---

## 3. Constraints and Limitations

- **Latency trade-off:** AirLLM's TPOT is orders of magnitude higher than native GPU inference (seconds vs milliseconds per token) because every decode step re-reads all N layers from NVMe. This is the explicit cost the project is measuring.
- **Shard path location:** `layer_shards_saving_path` must reside on a fast NVMe drive. Using a slow HDD or the OS system drive will dramatically worsen performance and may cause timeouts.
- **Disk space per quant level:**
  - FP16: ~65 GB
  - Q8: ~33 GB
  - Q4: ~17 GB
  - Q2: ~9 GB
- **Quantization backend:** `bitsandbytes` CUDA kernels required for Q8/Q4/Q2; CPU-only quantization is not supported.
- **One run at a time:** AirLLM holds GPU memory for the duration of a run; no concurrent experiments.

---

## 4. Alternatives Considered

| Alternative | Reason Not Used |
|-------------|----------------|
| Ollama | GGUF-centric; less transparent Python-side metric instrumentation; harder to hook into custom harness. |
| llama.cpp Python bindings | CPU-focused; does not demonstrate the AirLLM/paging mechanism required by L08. |
| DeepSpeed ZeRO-Infinity | Designed for multi-GPU training NVMe offloading; single-GPU inference setup is complex and not the assignment's target. |
| HF `device_map="auto"` with disk offload | Supported but coarser-grained than AirLLM's explicit per-layer sharding; less directly illustrative of the paging analogy. |

---

## 5. Success Criteria

- AirLLM runner completes at least 200 tokens of generation for `Qwen2.5-32B-Instruct` on the target hardware without OOM.
- Peak VRAM remains ≤ 24 GB throughout the entire run.
- All harness metrics are populated and written to `results/airllm_<quant>.json`.
- The runner performs a pre-flight check: raises an error if `shard_path` disk space is insufficient before beginning download or sharding.
- The runner uses `AutoModel` (not a hardcoded class) for Qwen-family models.

---

## 6. Specific Test Scenarios

| Test | Input | Expected Output |
|------|-------|----------------|
| Successful FP16 generation | Standard prompt, `max_new_tokens=200`, `quant_level="fp16"` | `generated_text` non-empty; all metrics populated; JSON written |
| VRAM budget respected | FP16 run on RTX 3090 | `peak_vram_gb ≤ 24.0` at all times |
| Missing HF token | `hf_token = ""` | Raises `AuthenticationError` with clear message before any download |
| Insufficient disk space | Free space < model on-disk size | `DiskSpaceError` raised in pre-flight; no partial download started |
| Wrong class guard | Qwen model loaded | Verify `AutoModel` is used (not `LlamaForCausalLM`) via mocked import check |
| Q4 VRAM reduction | `quant_level="q4"` vs `"fp16"` | `peak_vram_gb` for Q4 is < 50% of FP16 peak |
| Shard cache reuse | Run same quant level twice | Second run skips download; completes faster (shards already on disk) |
| Seed reproducibility | Two runs with `seed=42` | Generated text is identical between runs |
