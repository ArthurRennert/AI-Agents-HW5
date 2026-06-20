# Concepts

## AirLLM
Framework that loads model layers dynamically from disk instead of keeping the entire model in memory.

## Quantization
Technique that reduces memory usage by storing weights with lower precision.

## VRAM
Video memory located on the GPU.

## RAM
Main system memory used by the operating system and applications.

## TTFT
Time To First Token.

## TPOT
Time Per Output Token.

## Prefill
Initial stage where the model processes the prompt and builds the KV cache.

## Decode
Stage where the model generates tokens one by one.

## KV Cache
Cached attention values used to speed up generation.

## SafeTensors
Secure tensor storage format commonly used on Hugging Face.

## GGUF
Quantized model format commonly used by llama.cpp.