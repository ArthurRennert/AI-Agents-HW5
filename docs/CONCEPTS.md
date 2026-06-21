# Concepts Glossary (L08 — LoRA, AirLLM & Running a Massive LLM Locally)

---

## CPU vs GPU

**CPU (Central Processing Unit):** A general-purpose processor with a small number of powerful cores (e.g., 16 cores / 32 threads on the Ryzen 9 5950X). Optimized for low-latency sequential execution and complex control flow.

**GPU (Graphics Processing Unit):** A massively parallel processor with thousands of smaller cores (e.g., 10,496 CUDA cores on the RTX 3090). Optimized for throughput on regular, data-parallel workloads like matrix multiplication.

**SIMT (Single Instruction, Multiple Threads):** The GPU execution model. Groups of threads (a *warp*, typically 32 threads on NVIDIA) execute the same instruction in lockstep. This allows the GPU to amortize instruction-fetch overhead across many threads.

**Warp Divergence:** When threads within a warp take different branches (if/else), the GPU must execute both paths serially while masking inactive threads — degrading throughput. Well-structured tensor operations avoid this.

**Independent Thread Scheduling (Volta+):** Starting with the Volta architecture, each thread in a warp has its own program counter, enabling true inter-thread synchronization and reducing certain classes of warp-divergence penalties.

---

## CUDA → PTX → SASS (Compilation Pipeline)

**CUDA:** NVIDIA's parallel computing platform and programming model. CUDA code (`.cu` files) is written in C++ with GPU-specific extensions.

**PTX (Parallel Thread Execution):** An intermediate virtual ISA produced by the CUDA compiler (`nvcc`). PTX is architecture-agnostic within the NVIDIA ecosystem — it is compiled at runtime or install time to the actual GPU binary.

**SASS (Shader Assembly):** The actual machine code that runs on a specific GPU microarchitecture (e.g., Ampere for RTX 3090). PTX is compiled to SASS by the driver.

**JIT Compilation:** The CUDA driver can compile PTX to SASS *just-in-time* at first use. This adds a one-time startup delay but allows the same PTX binary to run on future GPU architectures.

**Fat Binary:** A compiled artifact containing both PTX and pre-compiled SASS for one or more GPU architectures, bundled together. PyTorch wheels ship fat binaries covering common CUDA architectures.

---

## Prefill vs Decode

**Prefill:** The first stage of LLM inference. The full input prompt is processed in a single forward pass, producing the KV-cache and the first output token. This stage is **compute-bound** (dominated by GEMM — matrix × matrix over all prompt tokens in parallel). Proxied by **TTFT (Time To First Token)**.

**Decode:** The autoregressive stage. After prefill, tokens are generated one at a time: each step attends to the entire KV-cache plus the new token. This stage is **memory-bandwidth-bound** (dominated by GEMV — matrix × vector, one row per attention head). Proxied by **TPOT (Time Per Output Token)**.

**Why the distinction matters:** These two stages have fundamentally different hardware bottlenecks. Prefill is limited by the GPU's FLOPS; decode is limited by how fast the GPU can read VRAM. Under AirLLM, decode is further limited by NVMe read speed because the KV-cache and all model weights are read from disk for every token.

---

## KV-Cache (Key-Value Cache)

**What it is:** During transformer attention, each layer computes Key (K) and Value (V) projections for every input token. These are cached in GPU VRAM so that subsequent decode steps can attend to all prior tokens without recomputing them.

**GEMM vs GEMV in the KV-cache context:**
- **Prefill:** All prompt positions are processed in parallel → attention is a matrix-matrix multiply (GEMM) → high arithmetic intensity → compute-bound.
- **Decode:** Only one new token is processed per step → attention is a matrix-vector multiply (GEMV against the cached K/V) → low arithmetic intensity → memory-bandwidth-bound.

**KV-cache memory cost:** Grows linearly with sequence length and number of layers. For a 32B model with a long context, KV-cache alone can consume several GB of VRAM.

---

## VRAM

**VRAM (Video RAM):** High-bandwidth memory physically located on the GPU (HBM2 on data-center GPUs; GDDR6X on the RTX 3090). The RTX 3090 has **24 GB GDDR6X** with ~936 GB/s bandwidth. All tensors used in a GPU forward pass must reside in VRAM. A model that does not fit in VRAM cannot run without offloading.

---

## SafeTensors vs GGUF

**SafeTensors:**
- A serialization format for tensors developed by HuggingFace.
- Uses a **flat binary buffer**: all tensor data is stored at fixed byte offsets described by a JSON header.
- Supports **zero-copy `mmap`**: the OS can map the file directly into virtual address space; tensor data is never copied — it is read directly from the mapped memory when accessed.
- Does **not** use Python pickle → safe from arbitrary code execution on load.
- Default format for most HuggingFace models, including `Qwen2.5-32B-Instruct`.

**GGUF (GPT-Generated Unified Format):**
- Serialization format used by `llama.cpp` and Ollama.
- Supports its own quantization encoding (Q2_K, Q4_K_M, Q8_0, etc.) baked into the file.
- Also supports `mmap` for fast loading.
- Optimized for CPU inference with SIMD; less native Python/PyTorch integration than SafeTensors.

**Why AirLLM uses SafeTensors:** AirLLM splits a SafeTensors model into per-layer shards. Each shard is a SafeTensors file for one transformer layer, exploiting the flat-buffer layout and `mmap` to load layers quickly and release them from VRAM after use.

---

## Quantization

**What it is:** Reducing the number of bits used to represent each model weight, shrinking memory footprint at the cost of representational precision.

**Precision ladder:**

| Format | Bits | Memory (32B model) | Notes |
|--------|------|--------------------|-------|
| FP32   | 32 | ~128 GB | Training default; never used for 32B inference |
| FP16   | 16 | ~64 GB  | Standard inference; negligible quality loss |
| FP8    | 8  | ~32 GB  | Emerging; requires RTX 4090 / H100 |
| Q8 (INT8) | 8 | ~32 GB | Supported via `bitsandbytes` on RTX 3090 |
| Q4 (NF4) | 4 | ~16 GB | Non-uniform grid; best 4-bit quality for LLMs |
| Q2     | 2  | ~8 GB  | Extreme compression; notable quality loss |

**NF4 (Normal Float 4):** A 4-bit quantization data type from the QLoRA paper. Uses a non-uniform grid with points optimally placed for data following a zero-mean normal distribution — which pre-trained LLM weights empirically approximate. Outperforms uniform INT4 at the same bit width.

**Double quantization:** A QLoRA technique where the quantization scaling constants (one per block) are themselves quantized from FP32 to FP8, saving ~0.5 additional bits per weight on average.

**Accuracy "red line":** The quantization level below which generated text loses coherence. Typically observed around Q2–Q3 for large models. Identifying this threshold is a core objective of the project.

---

## LoRA / QLoRA / OLoRA

**LoRA (Low-Rank Adaptation):**
- A parameter-efficient fine-tuning (PEFT) method.
- Instead of updating all W weights, it adds a low-rank factorization: `W' = W + AB` where `A ∈ ℝ^{d×r}`, `B ∈ ℝ^{r×k}`, and `r ≪ min(d, k)`.
- Only `A` and `B` are trained (millions of parameters vs billions); `W` is frozen.
- At inference, `BA` can be merged back into `W` with no added latency.

**QLoRA:**
- Combines LoRA with quantization: the base model `W` is loaded in 4-bit NF4; adapters `A`, `B` are trained in BF16.
- Introduces double quantization and paged optimizers to reduce training memory further.
- Enables fine-tuning 65B-parameter models on a single 48 GB GPU.

**OLoRA (Orthonormal LoRA):**
- Initializes adapter matrices via QR decomposition of a slice of the original weight matrix, rather than random initialization.
- Provides better-conditioned gradients at the start of fine-tuning, potentially faster convergence.

---

## AirLLM

**What it is:** An inference framework that enables running LLMs that exceed available VRAM by loading the model one layer (transformer block) at a time rather than all at once.

**How it works:**
1. Split the full model into per-layer SafeTensors shards stored on disk.
2. For each forward pass: load shard i into VRAM → compute layer i → unload shard i → load shard i+1.
3. Peak VRAM = memory for one layer + KV-cache, not the full model.

**The cost:** Every decode step reads all N layers from the NVMe SSD. Disk I/O becomes the dominant bottleneck (not VRAM, not compute). TPOT is orders of magnitude higher than native GPU inference.

**Why `AutoModel` for Qwen:** The Qwen2.5 architecture must be instantiated with `AutoModelForCausalLM`, not a hardcoded class like `LlamaForCausalLM`. AirLLM's `AutoModel` wrapper resolves the correct class from `config.json` automatically.

---

## Virtual Memory / MMU / Page Table / Page Fault / Locality

**Virtual Memory:** An OS abstraction that gives each process a private, contiguous address space larger than physical RAM. The OS uses disk (swap) as an extension of RAM.

**MMU (Memory Management Unit):** Hardware component that translates virtual addresses to physical addresses using the page table. Sits between the CPU and RAM.

**Page Table:** A data structure (managed by the OS) that maps virtual pages to physical frames. Each entry records whether a page is in RAM or on disk.

**Page Fault:** When a process accesses a virtual address whose page is not currently in physical RAM, the MMU raises a page fault. The OS then reads the page from disk into RAM and updates the page table. This is the OS-level equivalent of AirLLM loading a model layer from NVMe into VRAM.

**Locality of Reference:**
- *Spatial locality:* Accessing data near recently accessed data (e.g., sequential layer loading).
- *Temporal locality:* Reusing recently accessed data (e.g., the KV-cache).
- AirLLM's sequential layer access (layer 0 → layer N) exhibits high spatial locality, making NVMe prefetching effective.

**AirLLM as paging:** Each model layer shard is a "page on disk." Loading a shard is a "page fault." The `mmap`-based SafeTensors loading is the zero-copy equivalent of mapping a page into the address space.

---

## PagedAttention / FlexGen / LLM-in-a-Flash

**PagedAttention (vLLM):**
- Manages the KV-cache using OS-inspired paging: KV-cache is divided into fixed-size blocks ("pages") allocated non-contiguously in VRAM.
- Eliminates memory fragmentation from variable-length sequences; enables efficient batching.
- Also supports KV-cache sharing across requests (e.g., common system prompts), reducing effective cost per request.

**FlexGen:**
- An offloading framework that aggressively moves weights, activations, and KV-cache between GPU, CPU, and disk to maximize throughput on resource-constrained hardware.
- Optimizes the offloading schedule to hide I/O latency behind compute.
- More complex than AirLLM; designed for high-throughput batch inference rather than interactive use.

**LLM-in-a-Flash (Apple):**
- A technique for running LLMs on devices with limited DRAM by storing weights in flash (NAND) and streaming them to DRAM on demand.
- Uses *windowed access* and *row-column bundling* to amortize flash read latency.
- Conceptually similar to AirLLM but at the device/mobile level and with hardware-specific flash access optimizations.

---

## Disaggregated Serving

**What it is:** Separating the *prefill* and *decode* stages of LLM inference onto different hardware pools, because they have different resource profiles.

- **Prefill machines:** Need high compute (FLOPS) → beefy GPUs with fast tensor cores.
- **Decode machines:** Need high memory bandwidth → GPUs with wide HBM or multiple smaller GPUs.

**Systems:** Splitwise and DistServe are research systems demonstrating disaggregated serving. They route prefill requests to compute-optimized nodes and decode requests to bandwidth-optimized nodes, improving overall cluster utilization.

**Relevance to this project:** The TTFT vs TPOT split in our measurements directly reflects the prefill/decode distinction — demonstrating why a one-size-fits-all GPU may not be optimal at scale.
