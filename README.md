# Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

**Course:** AI Agents — L08  
**Model under test:** `Qwen/Qwen2.5-32B-Instruct`

---

## Central Claim

`Qwen2.5-32B-Instruct` (~65 GB in FP16) **cannot run directly** on the hardware below — but **successfully executes** under AirLLM with quantization, at a measurable latency and disk-I/O cost. This project measures that cost precisely.

---

## Hardware Profile

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 5950X — 16 cores / 32 threads @ 3.4 GHz base |
| RAM | 128 GB DDR4 |
| GPU | NVIDIA RTX 3090 — 24 GB GDDR6X VRAM, ~936 GB/s bandwidth |
| Storage | NVMe SSD (used for AirLLM layer shards) |

---

## Model Choice

**`Qwen/Qwen2.5-32B-Instruct`** — 32 billion parameters, SafeTensors format, HuggingFace.

| Property | Value |
|----------|-------|
| Parameters | 32 B |
| Format | SafeTensors |
| On-disk size (FP16) | ~65 GB |
| On-disk size (Q4) | ~16 GB |

**Why this model ("truck vs motorcycle" logic):**  
The model must be large enough to OOM on direct loading (32B at FP16 requires ~65 GB VRAM — nearly 3× the RTX 3090's 24 GB), but small enough to complete inference under AirLLM in a reasonable time. A 70B model would take too long per token; a 7B model fits natively and doesn't exercise the bottleneck. 32B is the sweet spot. Qwen2.5 is well-supported by AirLLM, has a documented `AutoModel` workaround for class-mismatch issues, and is instruction-tuned for evaluating generation quality across quantization levels.

---

## Project Structure

```
AI-Agents-HW5/
├── README.md
├── pyproject.toml          # pinned dependencies
├── .python-version         # Python 3.12
├── .gitignore
├── .env.example            # HF_TOKEN= (no secret)
├── LICENSE
├── docs/
│   ├── PRD.md              # requirements, KPIs, use cases
│   ├── PLAN.md             # C4 diagrams, ADRs, data schemas
│   ├── TODO.md             # task list with priorities and status
│   ├── CONCEPTS.md         # L08 concept glossary
│   ├── MODEL_SELECTION.md  # model choice rationale
│   ├── PROMPTS.md          # agent guidelines and per-module prompts
│   └── prd/
│       ├── measurement-harness.md
│       ├── airllm-runner.md
│       ├── quantization.md
│       └── economics.md
├── src/
│   ├── config/             # settings dataclass
│   ├── hardware/           # hardware profiler
│   ├── runners/            # baseline and AirLLM runners
│   ├── benchmark/          # measurement harness + persistence
│   ├── quantization/       # bitsandbytes config generator
│   ├── economics/          # cost analysis
│   └── viz/                # matplotlib figures
├── experiments/            # one script per scenario
├── results/                # raw JSON/CSV — never hand-edited
├── reports/                # deep-dive technical report
├── figures/                # generated charts and diagrams
└── tests/                  # pytest; target ≥ 85% coverage
```

---

## Experiment Phases

| Phase | Description |
|-------|-------------|
| Phase 1 | Planning docs, environment setup, hardware profiling, model selection |
| Phase 2 | Measurement harness, baseline execution (capture OOM/failure) |
| Phase 3 | AirLLM runner, quantization sweep (FP16 → Q8 → Q4 → Q2), graphs |
| Phase 4 | Economic analysis (API vs on-prem break-even), concept analysis |
| Phase 5 | Original extension (TBD) |
| Phase 6 | Technical report, README completion, final QA |

---

## Reproduction Instructions

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd AI-Agents-HW5
uv venv
uv pip install -e .
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and set your HuggingFace token:
# HF_TOKEN=hf_...
```

### 3. Set the shard path

```bash
# In .env, also set the path to your fast NVMe drive:
# SHARD_PATH=/path/to/nvme/shards
```

### 4. Run experiments

```bash
# Smoke test (tiny model, verify harness)
python experiments/smoke_test.py

# Baseline failure capture
python experiments/run_baseline.py

# AirLLM + quantization sweep
python experiments/run_airllm.py --quant fp16
python experiments/run_airllm.py --quant q8
python experiments/run_airllm.py --quant q4

# Economic analysis
python experiments/run_economics.py

# Generate all figures
python experiments/generate_figures.py
```

### 5. Run tests

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Key Findings

*(To be filled in after experiments complete — Phase 6)*

- **Baseline result:** ...
- **AirLLM FP16:** TTFT = ... ms, TPOT = ... ms/token, Peak VRAM = ... GB
- **Best quality/memory trade-off:** ...
- **Accuracy red line:** ...
- **Economic break-even:** ... requests/month

---

## Economic Summary

*(To be filled in after Phase 4)*

---

## Research Questions Answered

*(To be filled in after Phase 6 — answers in `reports/`)*

1. What was the bottleneck blocking direct execution, and how was it identified?
2. How does AirLLM change resource allocation, and how does it map to virtual memory / paging?
3. Quantization's impact on memory/speed/quality — where is the accuracy "red line"?
4. How do Prefill/Decode show up in the TTFT vs TPOT measurements?
5. What latency/throughput price is paid for running big on modest hardware?
6. When is local inference economically worth it vs an external API?
