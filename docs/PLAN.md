# Architecture Plan

## High-Level Pipeline

Model Runner
    ↓
Benchmark Harness
    ↓
Results Collection
    ↓
Analysis
    ↓
Visualization
    ↓
Technical Report

## Modules

### hardware/

Collect hardware specifications.

Responsibilities:

- CPU information
- RAM information
- GPU information
- Disk information

### runners/

Responsible for model execution.

Submodules:

- baseline_runner.py
- airllm_runner.py

### benchmark/

Collects performance metrics.

Metrics:

- TTFT
- TPOT
- Throughput
- Runtime
- Peak RAM
- Peak VRAM
- Energy estimate

### quantization/

Runs different quantization levels.

Scenarios:

- FP16
- Q8
- Q4
- Q2

### economics/

Cost analysis.

Calculates:

- API costs
- Local execution costs
- Break-even point

### viz/

Creates graphs.

Outputs:

- Throughput graphs
- Memory graphs
- Latency graphs
- Cost graphs

## Data Flow

Prompt
    ↓
Runner
    ↓
Metrics
    ↓
Results JSON
    ↓
Pandas Analysis
    ↓
Figures
    ↓
Report

## Technologies

- Python 3.11
- AirLLM
- Transformers
- Torch
- Accelerate
- Pandas
- Matplotlib
- Psutil
- BitsAndBytes

## Testing Strategy

Unit tests for:

- Metric collectors
- Result serialization
- Cost calculations

Target coverage:

85%+