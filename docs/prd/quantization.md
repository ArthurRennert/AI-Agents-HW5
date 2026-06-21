# Mini-PRD: Quantization Module

## 1. Description and Theoretical Background

Quantization reduces the numerical precision of model weights (and optionally activations) from higher-precision floating-point formats to lower-precision integers or smaller floats. This shrinks memory footprint at the cost of representational accuracy.

**Precision levels and their memory impact:**

| Level | Bits per weight | Memory vs FP32 | Hardware requirement | Notes |
|-------|----------------|----------------|---------------------|-------|
| FP32  | 32 | 1× | Any | Full precision; never used for 32B inference |
| FP16  | 16 | 0.5× | Any CUDA GPU | Default inference format; negligible quality loss |
| FP8   | 8  | 0.25× | H100 / RTX 4090 | Emerging standard; **not supported on RTX 3090** |
| Q8 (INT8) | 8 | 0.25× | RTX 3090 via `bitsandbytes` | Near-identical quality to FP16 at half the memory |
| Q4 (NF4) | 4 | 0.125× | RTX 3090 via `bitsandbytes` | NF4 grid optimized for LLM weight distributions |
| Q2   | 2  | 0.0625× | RTX 3090 via `bitsandbytes` | Extreme compression; significant quality loss expected |

**Why NF4 (Normal Float 4) outperforms linear INT4:**  
Pre-trained LLM weights empirically follow a near-Gaussian (normal) distribution. NF4 uses a non-uniform 4-bit quantization grid with grid points placed to minimize expected quantization error under a normal distribution. This preserves more information than a uniform INT4 grid at the same 4-bit width. NF4 was introduced in the QLoRA paper and is the default `bitsandbytes` 4-bit format.

**Double quantization (from QLoRA):**  
The quantization scaling constants (one per quantization block) are themselves quantized from FP32 to FP8, saving an additional ~0.5 bits per weight on average. Enabled via `bnb_4bit_use_double_quant=True` in `bitsandbytes`.

**Memory formula for `Qwen2.5-32B-Instruct` (32 billion parameters):**

| Level | Formula | Approximate size |
|-------|---------|----------------|
| FP16 | 32B × 2 bytes | ~64 GB |
| Q8   | 32B × 1 byte  | ~32 GB |
| Q4   | 32B × 0.5 bytes | ~16 GB |
| Q2   | 32B × 0.25 bytes | ~8 GB |

*(Actual on-disk size includes config files, tokenizer weights, etc. — add ~1–2 GB overhead.)*

**Accuracy "red line":**  
As precision drops, weights are rounded to fewer representable values. Below a critical threshold (experimentally observed near Q2–Q3 for large models), the model loses text coherence — outputs become repetitive, factually inaccurate, or grammatically broken. The red line is identified by qualitative review of generated text at each quantization level.

**Quantization effects on the AirLLM pipeline:**  
Lower precision → smaller shards → fewer bytes to read from NVMe per layer → lower TPOT (faster decode). This creates a quality-vs-speed-vs-memory Pareto trade-off that is the core experimental finding.

---

## 2. Specific Requirements

### Inputs

| Input | Type | Notes |
|-------|------|-------|
| `quant_level` | `str` | `"fp16"` \| `"q8"` \| `"q4"` \| `"q2"` |
| `model_name` | `str` | HuggingFace model ID |
| `shard_path` | `str` | NVMe path; must have sufficient space for this level's shards |

### Outputs

| Output | Type | Notes |
|--------|------|-------|
| `bnb_config` | `BitsAndBytesConfig \| None` | Passed to AirLLM runner; `None` for FP16 (default dtype) |
| `expected_disk_gb` | `float` | Pre-flight estimate of shard storage required |
| `expected_vram_gb` | `float` | Pre-flight estimate of peak VRAM for one layer |

### Performance Metrics (expected ranges for `Qwen2.5-32B-Instruct`)

| Level | Expected Peak VRAM | Expected TPOT | Expected Quality |
|-------|--------------------|--------------|-----------------|
| FP16 | ~22–24 GB | Highest (most bytes/layer read from disk) | Reference quality |
| Q8   | ~12–14 GB | ~50% lower than FP16 | Near-identical to FP16 |
| Q4   | ~7–9 GB   | ~75% lower than FP16 | Minor degradation |
| Q2   | ~4–5 GB   | ~87% lower than FP16 | Significant degradation |

---

## 3. Constraints and Limitations

- **FP8 not supported** on RTX 3090 (Ampere); requires Ada Lovelace (RTX 4090) or Hopper (H100).
- `bitsandbytes` Q4/Q2 requires CUDA; CPU-only quantization is not in scope.
- Quantization is applied at model load time, not post-hoc. The model must be re-sharded (and re-downloaded if the shard cache is per-quant-level) when switching quantization levels.
- Quality evaluation in this project is qualitative (human review of generated text). Automated perplexity metrics are out of scope unless chosen as an extension.
- `bitsandbytes` Q2 may not be stable on all model architectures; testing on Q2 is P2 priority.

---

## 4. Alternatives Considered

| Alternative | Reason Not Chosen |
|-------------|------------------|
| GPTQ (post-training quantization) | Requires a calibration dataset and a separate calibration step; more complex setup. `bitsandbytes` is simpler and sufficient for this project. |
| AWQ (Activation-aware Weight Quantization) | Achieves better quality than GPTQ at the same bit width; but not directly integrated into AirLLM's `bitsandbytes` path. Better quality at the cost of integration complexity. |
| llama.cpp GGUF quantization | Excellent for CPU inference with GGUF models; out of scope since the target mechanism is AirLLM + bitsandbytes. |
| FP8 via `transformer-engine` | Not supported on RTX 3090 hardware; excluded. |

---

## 5. Success Criteria

- `get_bnb_config()` returns a valid `BitsAndBytesConfig` for Q8, Q4, Q2 and `None` for FP16.
- Disk space estimates are within 20% of actual shard sizes on disk.
- FP16, Q8, and Q4 experiments all complete successfully on the target hardware without error.
- Q4 peak VRAM is measurably lower than FP16 peak VRAM (expected: > 50% reduction).
- One quantization level is explicitly identified and documented in the report as the accuracy red line.

---

## 6. Specific Test Scenarios

| Test | Input | Expected Output |
|------|-------|----------------|
| Config for Q8 | `quant_level="q8"` | Returns `BitsAndBytesConfig(load_in_8bit=True)` |
| Config for Q4 | `quant_level="q4"` | Returns config with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True` |
| Config for FP16 | `quant_level="fp16"` | Returns `None` (no bitsandbytes config needed) |
| Invalid level | `quant_level="q1"` | Raises `ValueError` listing valid options |
| Disk estimate accuracy | `quant_level="q4"`, 32B params | Estimate within 20% of actual Q4 shard size on disk |
| VRAM reduction (FP16 vs Q4) | Compare `results/airllm_fp16.json` vs `results/airllm_q4.json` | `peak_vram_gb` for Q4 < 50% of FP16 peak |
| TPOT reduction (FP16 vs Q4) | Same comparison | `tpot_ms` for Q4 < FP16 (fewer bytes per layer to read from disk) |
| Accuracy red line | Q2 generated text reviewed | Manual review confirms degradation; `quality_note` in results JSON set to `"incoherent"` or `"minor_degradation"` |
