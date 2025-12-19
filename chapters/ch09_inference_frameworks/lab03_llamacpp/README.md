# Lab 03: llama.cpp

## Objective

Run efficient CPU inference with llama.cpp and understand GGUF quantization.

## What You'll Build

Functions and utilities for:
- Loading GGUF models
- CPU and GPU inference
- Quantization format handling
- Performance benchmarking

## Prerequisites

Read these docs first:
- `../docs/03_llama_cpp.md`

## Instructions

1. Open `src/llamacpp_inference.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

**Note**: Requires `llama-cpp-python` package. Install with:
```bash
pip install llama-cpp-python
```

## Functions to Implement

### `load_model(model_path, n_ctx=2048, n_threads=None, n_gpu_layers=0, **kwargs)`
Load a GGUF model for inference.
- Set context size
- Configure threading
- Optional GPU offloading

### `generate_text(model, prompt, max_tokens=100, temperature=0.7, **kwargs)`
Generate text completion.
- Handle stop sequences
- Return generated text

### `generate_chat(model, messages, max_tokens=100, **kwargs)`
Generate chat completion.
- Format messages properly
- Support system, user, assistant roles

### `get_model_info(model_path)`
Extract GGUF model metadata.
- Model architecture
- Quantization type
- Parameter count
- Context length

### `estimate_memory_usage(model_path, n_ctx)`
Estimate memory requirements.
- Model file size
- KV-cache size
- Total RAM needed

### `benchmark_inference(model, prompts, **kwargs)`
Benchmark inference performance.
- Tokens per second
- Time to first token
- Memory usage

### `count_tokens(model, text)`
Count tokens in text.
- Use model's tokenizer

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
from llamacpp_inference import (
    load_model,
    generate_text,
    generate_chat,
    get_model_info,
    benchmark_inference
)

# Load model
model = load_model(
    "models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8
)

# Simple generation
text = generate_text(model, "The meaning of life is", max_tokens=50)
print(text)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]
response = generate_chat(model, messages, max_tokens=200)
print(response)

# Get model info
info = get_model_info("models/llama-2-7b-chat.Q4_K_M.gguf")
print(f"Quantization: {info['quantization']}")
print(f"Context length: {info['context_length']}")

# Benchmark
metrics = benchmark_inference(model, ["Test prompt"] * 10)
print(f"Throughput: {metrics['tokens_per_second']:.1f} tok/s")
```

## Hints

- `n_threads` should match physical CPU cores
- `n_gpu_layers=-1` offloads entire model to GPU
- Larger `n_ctx` uses more memory
- Q4_K_M is a good balance of size and quality

## GGUF Quantization Types

| Type | Bits | Quality | Speed |
|------|------|---------|-------|
| F16 | 16 | Best | Slow |
| Q8_0 | 8 | Excellent | Good |
| Q5_K_M | 5.5 | Very Good | Fast |
| Q4_K_M | 4.5 | Good | Fast |
| Q4_0 | 4 | Good | Fastest |
| Q3_K_M | 3.5 | OK | Fastest |

## Verification

All tests pass = you can run llama.cpp inference!

Milestone: Run a 7B quantized model on CPU at >10 tokens/second.
