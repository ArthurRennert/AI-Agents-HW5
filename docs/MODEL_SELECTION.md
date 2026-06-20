# Model Selection

## Selected Model

Qwen2.5-32B-Instruct

## Model Information

* Parameters: 32 Billion
* Format: SafeTensors
* Source: Hugging Face
* Model Type: Instruction-tuned Large Language Model

## Hardware Used

* GPU: NVIDIA RTX 3090 (24GB VRAM)
* CPU: AMD Ryzen 9 5950X
* RAM: 32GB

## Reasoning

The goal of this project is to evaluate whether a large language model that is difficult to execute directly can be executed using AirLLM and quantization techniques.

Qwen2.5-32B was selected because:

* It is significantly larger than typical consumer-grade models.
* It places substantial pressure on available VRAM and RAM resources.
* It is supported by the Transformers ecosystem and AirLLM.
* It provides a realistic scenario for evaluating memory-saving techniques and their performance tradeoffs.

## Expected Challenges

* High VRAM requirements.
* High RAM consumption.
* Increased disk I/O during AirLLM execution.
* Longer inference latency compared to smaller models.

## Expected Outcome

Direct execution is expected to stress hardware resources significantly, while AirLLM and quantization should reduce memory requirements at the cost of additional latency and disk activity.
