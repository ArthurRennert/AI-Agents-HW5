"""Multi-model size-sweep analysis: where the bottleneck shifts as a model grows.

For each Qwen2.5 size we estimate (1) whether direct (baseline) execution fits in
physical RAM and (2) the per-token latency AirLLM pays to stream the weights off
NVMe. All physical assumptions are explicit constants so the analysis is
reproducible. Records are written with a self-describing schema that the Phase-3
loader ignores (it raises KeyError on the missing 'scenario' key and skips them).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Physical assumptions (stated for reproducibility) ------------------------
BYTES_PER_PARAM_FP16 = 2
BASELINE_OVERHEAD = 1.20          # weights + ~20% for activations / KV / framework
AIRLLM_BASE_RAM_GB = 2.5          # runtime + tokenizer + one activation set
NVME_READ_GB_S = 3.5              # sustained sequential read, consumer NVMe SSD
CPU_COMPUTE_S_PER_TOKEN = 0.30    # rough per-token compute on a 10-core CPU


@dataclass(frozen=True)
class ModelSpec:
    """A single model in the size sweep."""

    name: str
    param_billions: float
    n_layers: int


@dataclass
class SizeResult:
    """Per-model bottleneck analysis result."""

    name: str
    param_billions: float
    n_layers: int
    fp16_disk_gb: float
    baseline_required_gb: float
    fits_baseline: bool
    baseline_outcome: str
    airllm_peak_ram_gb: float
    disk_read_per_token_gb: float
    est_tpot_s: float
    bottleneck: str


# Qwen2.5-Instruct family — param counts and decoder-layer counts.
REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen2.5-1.5B-Instruct", 1.54, 28),
    ModelSpec("Qwen2.5-3B-Instruct", 3.09, 36),
    ModelSpec("Qwen2.5-7B-Instruct", 7.62, 28),
    ModelSpec("Qwen2.5-14B-Instruct", 14.77, 48),
    ModelSpec("Qwen2.5-32B-Instruct", 32.76, 64),
)


def analyze_model(spec: ModelSpec, ram_gb: float) -> SizeResult:
    """Estimate baseline feasibility and AirLLM latency for one model."""
    fp16_disk_gb = round(spec.param_billions * BYTES_PER_PARAM_FP16, 2)
    baseline_required_gb = round(fp16_disk_gb * BASELINE_OVERHEAD, 2)
    fits_baseline = baseline_required_gb <= ram_gb

    per_layer_gb = fp16_disk_gb / spec.n_layers
    airllm_peak_ram_gb = round(per_layer_gb + AIRLLM_BASE_RAM_GB, 2)

    # AirLLM re-reads every layer from disk on each forward pass.
    disk_read_per_token_gb = fp16_disk_gb
    est_tpot_s = round(
        disk_read_per_token_gb / NVME_READ_GB_S + CPU_COMPUTE_S_PER_TOKEN, 2
    )

    if fits_baseline:
        outcome, bottleneck = "fits_in_ram", "none (direct execution viable)"
    else:
        outcome, bottleneck = "OOM", "disk_io_bandwidth"

    return SizeResult(
        name=spec.name,
        param_billions=spec.param_billions,
        n_layers=spec.n_layers,
        fp16_disk_gb=fp16_disk_gb,
        baseline_required_gb=baseline_required_gb,
        fits_baseline=fits_baseline,
        baseline_outcome=outcome,
        airllm_peak_ram_gb=airllm_peak_ram_gb,
        disk_read_per_token_gb=disk_read_per_token_gb,
        est_tpot_s=est_tpot_s,
        bottleneck=bottleneck,
    )


def run_size_sweep(ram_gb: float, results_dir: str = "results") -> list[SizeResult]:
    """Analyze every registry model and persist one record per model."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[SizeResult] = []
    for spec in REGISTRY:
        res = analyze_model(spec, ram_gb)
        results.append(res)
        payload = {
            "extension": "size_sweep",
            "ram_gb": ram_gb,
            "assumptions": {
                "bytes_per_param_fp16": BYTES_PER_PARAM_FP16,
                "baseline_overhead": BASELINE_OVERHEAD,
                "nvme_read_gb_s": NVME_READ_GB_S,
                "cpu_compute_s_per_token": CPU_COMPUTE_S_PER_TOKEN,
            },
            **asdict(res),
        }
        slug = res.name.lower().replace("/", "-")
        (out_dir / f"extension_size_{slug}.json").write_text(
            json.dumps(payload, indent=2)
        )
    return results


def load_size_results(results_dir: str = "results") -> list[SizeResult]:
    """Load all persisted size-sweep records, sorted by parameter count."""
    out: list[SizeResult] = []
    for path in Path(results_dir).glob("extension_size_*.json"):
        d = json.loads(path.read_text())
        out.append(SizeResult(**{k: d[k] for k in SizeResult.__annotations__}))
    return sorted(out, key=lambda r: r.param_billions)
