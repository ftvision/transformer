# Lab 01: Memory Bandwidth Analysis

## Objective

Build tools to analyze memory-bound inference and understand the roofline model.

## What You'll Build

Functions to calculate:
- Theoretical maximum tokens/second for any model size and GPU
- Arithmetic intensity of different operations
- KV-cache memory requirements
- Whether an operation is memory-bound or compute-bound

## Prerequisites

Read these docs first:
- `../docs/01_memory_bound_inference.md`

## Instructions

1. Open `src/memory_analysis.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `calculate_model_memory(num_params, dtype_bytes=2)`
Calculate the memory required to store model weights.
- `num_params`: Number of parameters (e.g., 7e9 for 7B)
- `dtype_bytes`: Bytes per parameter (2 for fp16, 1 for int8, 0.5 for int4)
- Returns: Memory in bytes

### `calculate_max_tokens_per_second(model_memory_bytes, bandwidth_bytes_per_sec)`
Calculate the theoretical maximum tokens per second.
- This is the absolute upper bound based on memory bandwidth
- `max_tokens/sec = bandwidth / model_memory`
- Returns: Maximum tokens per second

### `calculate_arithmetic_intensity(flops, bytes_loaded)`
Calculate the arithmetic intensity of an operation.
- `intensity = FLOPS / bytes_loaded`
- Returns: FLOPS per byte

### `is_memory_bound(arithmetic_intensity, compute_flops, bandwidth_bytes_per_sec)`
Determine if an operation is memory-bound or compute-bound.
- Calculate the "ridge point" = compute / bandwidth
- If intensity < ridge point: memory-bound
- Returns: bool (True if memory-bound)

### `calculate_kv_cache_memory(batch_size, seq_len, num_layers, d_model, dtype_bytes=2)`
Calculate KV-cache memory requirements.
- Formula: `batch × seq_len × num_layers × 2 × d_model × dtype_bytes`
- The `2` is for K and V
- Returns: Memory in bytes

### `calculate_attention_flops(batch_size, seq_len_q, seq_len_k, d_model, num_heads)`
Calculate FLOPs for attention computation.
- QK^T: `batch × heads × seq_q × seq_k × (2 × head_dim)` (multiply-add)
- Softmax: approximately `batch × heads × seq_q × seq_k × 5` ops
- Attention × V: `batch × heads × seq_q × seq_k × (2 × head_dim)`
- Returns: Total FLOPs

### `analyze_inference_bottleneck(model_config, gpu_config)`
Comprehensive analysis of inference characteristics.
- Takes model and GPU configuration dicts
- Returns dict with: model_memory, max_tokens_per_sec, kv_cache_memory, is_memory_bound, utilization

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_memory_analysis.py::TestModelMemory

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
from memory_analysis import (
    calculate_model_memory,
    calculate_max_tokens_per_second,
    calculate_kv_cache_memory,
    analyze_inference_bottleneck
)

# 7B model in fp16
model_memory = calculate_model_memory(7e9, dtype_bytes=2)
print(f"7B fp16 model: {model_memory / 1e9:.1f} GB")  # 14.0 GB

# Theoretical max on A100 (2 TB/s bandwidth)
max_tps = calculate_max_tokens_per_second(model_memory, 2e12)
print(f"Max tokens/sec: {max_tps:.0f}")  # ~143

# KV-cache for batch of 8, seq_len 2048
kv_memory = calculate_kv_cache_memory(
    batch_size=8, seq_len=2048,
    num_layers=32, d_model=4096, dtype_bytes=2
)
print(f"KV-cache: {kv_memory / 1e9:.1f} GB")  # ~8 GB

# Full analysis
model_config = {
    'num_params': 7e9,
    'num_layers': 32,
    'd_model': 4096,
    'num_heads': 32,
    'dtype_bytes': 2
}
gpu_config = {
    'bandwidth_bytes_per_sec': 2e12,  # 2 TB/s
    'compute_flops': 312e12,  # 312 TFLOPS
    'memory_bytes': 80e9  # 80 GB
}
analysis = analyze_inference_bottleneck(model_config, gpu_config)
print(f"Memory-bound: {analysis['is_memory_bound']}")  # True
```

## Hints

- All calculations should handle both small and large numbers (use scientific notation)
- Pay attention to units: bytes vs GB, FLOPS vs TFLOPS
- For attention FLOPs, multiply-add counts as 2 operations
- The "ridge point" on the roofline is where compute = bandwidth × intensity

## Verification

All tests pass = you understand memory-bound inference!

Key insight: For single-token inference, you should find that:
- Arithmetic intensity is very low (~1-2 FLOPS/byte)
- All LLMs are memory-bound during decoding
- Batching increases arithmetic intensity
