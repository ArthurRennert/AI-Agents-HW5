# Model Selection

## Selected Model

**Qwen2.5-32B-Instruct**

## Model Information

* Parameters: ~32.8 Billion
* Format: SafeTensors (flat buffer enables zero-copy `mmap` layer loads)
* On-disk size (FP16): ~65 GB
* Source: Hugging Face
* Model type: Instruction-tuned LLM (Qwen family — the brief flags `AutoModel`)

## Hardware Used (profiled — see `results/hardware.json`)

* GPU: **none** (0 GB VRAM)
* CPU: Intel 10-core / 12-thread (Family 6 Model 154, Alder Lake)
* RAM: **32 GB**
* Storage: NVMe SSD (~665 GB free)

## Reasoning ("truck vs motorcycle") — empirically calibrated

This machine has **no GPU**, so the binding resource is system RAM. The model must
be too large to hold in 32 GB of RAM, yet still have a realistic chance under
AirLLM's layer-by-layer streaming.

The RAM wall was located by direct measurement, not estimation:

| Model | FP16 weights | Direct-load peak RAM (measured) | Fits in 32 GB? | Direct-load TTFT |
|-------|-------------:|--------------------------------:|:--------------:|-----------------:|
| 7B    | ~15 GB | 14.25 GB | ✅ yes | 107 s (slow CPU compute) |
| 14B   | ~29 GB | 28.05 GB | ✅ yes (≈4 GB headroom) | 298 s |
| 32B   | ~65 GB | — (≈2× RAM) | ❌ **no** | — |

* **7B and 14B both fit and ran directly** — slowly (CPU FP16 inference is
  un-accelerated), but they fit in 32 GB, so neither can demonstrate the
  "cannot run directly" requirement.
* **32B is the correct choice.** At ~65 GB it is roughly twice the physical RAM,
  so a direct load cannot be held resident: it spills to the Windows page file
  and either thrashes to an unusable latency or is OOM-killed — the genuine
  **memory-capacity bottleneck** the project is built around.
* **AirLLM rescues it.** Streaming one layer at a time keeps peak RAM bounded
  (~3 GB), so the same 32B workload completes with controlled memory at a
  disk-bandwidth latency cost.

The 14B measurement is a useful calibration point: the analytical estimate
(Extension A) applied a 1.2× activation/overhead factor and predicted a larger
footprint than the 28 GB actually observed, which means the true RAM wall sits
**between 14B and 32B** for this machine. Measurement refined the prediction.

## Expected Challenges

* Direct 32B load exceeds RAM → page-file thrash or OOM-kill (the captured failure).
* High, sustained NVMe read under AirLLM (every layer re-read per forward pass).
* High per-token latency; use `MAX_NEW_TOKENS=50` to keep each sweep level tractable.
* Quantization (Q8 → Q4 → Q2) lowers per-token disk read and memory, with a quality
  "red line" to be located empirically.

## Expected Outcome

Direct execution fails on memory (page-file thrash / OOM); AirLLM completes the same
workload with bounded RAM at a disk-bandwidth-shaped latency cost, and quantization
trades output quality for lower memory/latency until coherence breaks down.
