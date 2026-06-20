# Measurement Harness PRD

## Purpose

Collect performance metrics during model execution.

## Inputs

- Prompt
- Generated output
- Runtime information

## Outputs

- TTFT
- TPOT
- Throughput
- Peak RAM
- Peak VRAM
- Total runtime

## Edge Cases

- OOM failures
- Interrupted execution
- Missing GPU