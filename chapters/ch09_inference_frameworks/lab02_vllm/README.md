# Lab 02: vLLM Serving

## Objective

Deploy and serve LLMs with vLLM for high-throughput inference.

## What You'll Build

Functions and utilities for:
- Loading models with vLLM
- Offline batch inference
- Online serving with OpenAI-compatible API
- Measuring throughput and latency

## Prerequisites

Read these docs first:
- `../docs/01_paged_attention.md`
- `../docs/02_vllm_architecture.md`

## Instructions

1. Open `src/vllm_serving.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

**Note**: Some tests require a GPU. Tests will be skipped if no GPU is available.

## Functions to Implement

### `create_llm(model_name, **kwargs)`
Create a vLLM LLM instance.
- Set reasonable defaults for gpu_memory_utilization, max_model_len
- Handle tensor parallelism
- Return the LLM instance

### `create_sampling_params(temperature=0.8, top_p=0.95, max_tokens=100, **kwargs)`
Create sampling parameters for generation.
- Return a SamplingParams object

### `generate_offline(llm, prompts, sampling_params)`
Generate completions for a batch of prompts (offline mode).
- Use llm.generate()
- Return list of generated texts

### `measure_throughput(llm, prompts, sampling_params)`
Measure tokens per second throughput.
- Time the generation
- Calculate total tokens generated
- Return throughput metrics

### `start_server(model_name, port=8000, **kwargs)`
Start a vLLM OpenAI-compatible server.
- Return a process handle or server object
- Should be non-blocking

### `query_server(prompt, base_url="http://localhost:8000", **kwargs)`
Query a running vLLM server.
- Use the OpenAI-compatible chat completions API
- Return the generated text

### `calculate_memory_usage(model_name, max_model_len, **kwargs)`
Estimate memory usage for a model.
- Consider model weights, KV-cache, activations
- Return memory breakdown dict

## Testing

```bash
# Run tests (some may skip without GPU)
uv run pytest tests/

# Run only CPU-compatible tests
uv run pytest tests/ -m "not gpu"

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
from vllm_serving import (
    create_llm,
    create_sampling_params,
    generate_offline,
    measure_throughput
)

# Create LLM (requires GPU for best performance)
llm = create_llm("meta-llama/Llama-2-7b-hf")

# Create sampling params
params = create_sampling_params(temperature=0.7, max_tokens=100)

# Generate
prompts = ["The meaning of life is", "In the future, AI will"]
outputs = generate_offline(llm, prompts, params)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")

# Measure throughput
metrics = measure_throughput(llm, prompts * 50, params)  # 100 prompts
print(f"Throughput: {metrics['tokens_per_second']:.1f} tok/s")
```

## Server Example

```python
from vllm_serving import start_server, query_server
import time

# Start server (background)
server = start_server("gpt2", port=8000)
time.sleep(10)  # Wait for server to start

# Query
response = query_server("Hello, how are you?")
print(response)

# Cleanup
server.terminate()
```

## Hints

- vLLM is optimized for GPUs; CPU mode is slow
- Use smaller models (e.g., gpt2) for testing
- Tensor parallelism splits model across multiple GPUs
- gpu_memory_utilization controls KV-cache size

## Key Metrics

- **Throughput**: Tokens generated per second
- **TTFT**: Time to first token
- **TPOT**: Time per output token (after first)
- **Memory**: GPU memory used

## Verification

All tests pass = you can deploy with vLLM!

Milestone: Achieve >100 tokens/second throughput on a 7B model with GPU.
