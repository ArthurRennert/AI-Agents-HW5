# Phase 5 — Original Extensions

Two extensions were chosen (the brief asks for 1–2). They are deliberately
complementary: Extension A locates **where** the bottleneck sits as a model
grows, and Extension B demonstrates **why** AirLLM pays the latency it pays.
Together they close the loop on the project's central claim — that AirLLM
converts a *memory-capacity* wall into a *disk-bandwidth* latency cost.

Both were run on the profiled machine (`results/hardware.json`): a CPU-only,
10-core / 12-thread host with **16.8 GB RAM** and NVMe storage. There is no GPU,
so the relevant ceiling throughout is system RAM, not VRAM.

> **Selection rationale.** Quality-vs-quant Pareto and the roofline were already
> built in Phase 3, so they were excluded here. A LoRA/QLoRA fine-tune was
> considered and rejected: the brief explicitly warns against the extension
> ballooning into a final project, and a training run adds little to the
> memory-hierarchy story these two experiments tell.

---

## Extension A — Multi-model size sweep: the bottleneck shifts with size

**Question.** As the model grows, does the same bottleneck persist, or does it
move? This is the "truck vs motorcycle" point from L08 made quantitative.

**Method.** `src/scaling/size_sweep.py` analyzes the Qwen2.5 family
(1.5B → 3B → 7B → 14B → 32B). For each model it estimates the FP16 baseline RAM
requirement (`params × 2 bytes × 1.2` overhead for activations/KV/framework),
checks it against the profiled RAM ceiling, and estimates the AirLLM per-token
latency as `weight_bytes / NVMe_read_bandwidth + compute`. Every physical
assumption (2 bytes/param FP16, 3.5 GB/s NVMe sustained read, 0.30 s/token CPU
compute) is a named constant in the module so the analysis is reproducible.
Run it with `python experiments/run_size_comparison.py`.

**Result** (`figures/extension_size_comparison.png`):

| Model | FP16 on disk | Baseline RAM needed | Fits in 16.8 GB? | AirLLM TPOT |
|-------|-------------:|--------------------:|:----------------:|------------:|
| 1.5B  | 3.1 GB  | 3.7 GB  | ✅ | 1.2 s |
| 3B    | 6.2 GB  | 7.4 GB  | ✅ | 2.1 s |
| 7B    | 15.2 GB | 18.3 GB | ❌ | 4.7 s |
| 14B   | 29.5 GB | 35.5 GB | ❌ | 8.7 s |
| 32B   | 65.5 GB | 78.6 GB | ❌ | 19.0 s |

**Interpretation.** The RAM wall is first crossed at **7B** on this machine:
1.5B and 3B run directly, while 7B and above cannot be held resident at all.
Crucially, the bottleneck does not merely intensify — it *changes kind*. Below
the wall the constraint is whether the weights fit; above it, AirLLM keeps peak
RAM **bounded and flat** (~3 GB, one layer plus runtime, the blue line) so the
constraint becomes **NVMe read bandwidth**: TPOT scales almost linearly with
model size because every forward pass must stream the full weight set from disk.
The 32B model is ~16× the 1.5B model's TPOT for exactly this reason. This is the
disaggregation the lecture describes: capacity is decoupled from the device and
served from the storage tier, at a bandwidth-shaped latency price.

---

## Extension B — Paging instrumentation: AirLLM *is* OS virtual memory

**Question.** Research question §4.2 asks how AirLLM maps onto virtual memory and
paging. Rather than assert the analogy, this extension *exhibits* it on real
hardware.

**Method.** `src/paging/layer_demo.py` writes N disk-backed "layer" shards, then
processes them exactly the way AirLLM streams transformer layers: `mmap` the
shard, touch every page (forcing real page-ins), run a small matmul (the
"compute"), then evict the layer with `madvise(MADV_DONTNEED)` before loading the
next. `src/paging/instrument.py` samples RSS and the process page-fault counters
(`/proc/<pid>/stat` minflt/majflt on Linux; psutil's `num_page_faults` on
Windows) in a background thread, the same pattern as the Phase-2 harness. This is
CPU-only friendly and needs no model download, so the mechanic is reproducible by
any reader. Run it with `python experiments/run_paging_demo.py`.

**Result** (`figures/extension_paging.png`) — a real run of 24 layers × 96 MB:

- A clean **RSS sawtooth**: each layer pages in (RSS climbs ~102 MB), holds
  during compute, then drops back to the ~32 MB baseline on eviction. Twenty-four
  teeth, one per layer — the unmistakable signature of demand paging.
- **14,099 total page faults** across the run, with the per-layer bars showing
  the first layer faulting hardest (**746 faults**, a cold page cache) and later
  layers settling to a steady **~205 faults** each as the kernel's read-ahead
  warms.

**Interpretation.** This is the OS memory hierarchy from L08 in miniature. The
`mmap` of a SafeTensors-style flat buffer gives zero-copy, demand-paged access:
pages are not resident until touched, each touch that misses is a page fault, and
`MADV_DONTNEED` is the explicit page-out that keeps the resident set bounded —
precisely how AirLLM holds a 32B model's peak memory near one layer's worth. The
sawtooth is the visual proof of the latency/capacity trade: the system never
holds more than one layer, but it pays a fault-and-fetch tax on every layer of
every token. Under a real 32B AirLLM run that tax is the dominant cost, which is
why Extension A's TPOT is bandwidth-shaped. To instrument an actual generation
run, wrap the `model.generate()` call in `experiments/run_airllm.py` with a
`PageSampler` (the sampler is engine-agnostic).

---

## Mapping to the research questions (§4)

- **§4.1 (bottleneck: memory vs compute).** Extension A shows the bottleneck is
  memory capacity below the wall and disk bandwidth above it — never compute on
  this hardware.
- **§4.2 (AirLLM ↔ virtual memory / paging).** Extension B demonstrates the
  page-in → compute → page-out cycle directly, with measured faults and RSS.
- **§4.5 (latency/throughput price).** Extension A quantifies the price as a
  near-linear TPOT-vs-size curve (1.2 s → 19.0 s); Extension B explains its
  origin as per-layer paging.

## Reproduce

```bash
python experiments/run_size_comparison.py        # writes results/extension_size_*.json
python experiments/run_paging_demo.py            # writes results/extension_paging_trace.json
python experiments/generate_extension_figures.py # writes figures/extension_*.png
pytest tests/test_size_sweep.py tests/test_paging.py
```

All numbers above are emitted by the scripts; the figures embed the same data.
