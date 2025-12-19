# Lab 05: Profiling and Optimization

## Objective

Learn to profile deep learning code, identify bottlenecks, and apply targeted optimizations.

## What You'll Build

1. Profiling utilities for PyTorch and JAX
2. Performance analysis of transformer components
3. Optimization techniques and their impact measurement
4. Benchmarking infrastructure

## Prerequisites

- Complete Labs 01-04
- Read `../docs/06_profiling.md`
- Understanding of GPU memory hierarchy

## Why Profile?

Performance intuition is often wrong. Common misconceptions:
- "Matrix multiply is always the bottleneck" → Often memory-bound ops are
- "Simple kernels are fast" → Launch overhead matters
- "GPU utilization is 100%" → Compute vs memory utilization differs

**Always measure, never guess.**

## Instructions

1. Open `src/profiling.py`
2. Implement the profiling and optimization functions
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `benchmark_function(fn, args, warmup, iterations)`

Accurate benchmarking with proper warmup and synchronization.

```python
# Warmup (includes compilation)
for _ in range(warmup):
    result = fn(*args)
    sync()  # Wait for GPU!

# Timed runs
start = time.perf_counter()
for _ in range(iterations):
    result = fn(*args)
    sync()
end = time.perf_counter()

return (end - start) / iterations
```

### `profile_attention_components(seq_len, d_model, num_heads)`

Profile each component of attention separately.

```python
# Profile:
# 1. QKV projection
# 2. Score computation (Q @ K.T)
# 3. Softmax
# 4. Weighted sum (weights @ V)
# 5. Output projection
```

### `identify_bottleneck(profile_results)`

Analyze profile results to identify the bottleneck.

```python
# Compare times
# Identify which component takes most time
# Determine if memory-bound or compute-bound
```

### `optimize_attention(Q, K, V, optimization_level)`

Apply different optimization strategies.

```python
# Level 0: Baseline (no optimization)
# Level 1: Use torch.compile
# Level 2: Use scaled_dot_product_attention
# Level 3: Use flash attention (if available)
```

### `measure_memory_usage(fn, args)`

Track peak memory usage during execution.

```python
# Reset memory stats
# Run function
# Get peak memory
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Skip GPU tests if no CUDA
uv run pytest tests/ -v

# Run with visible output
uv run pytest tests/ -v -s
```

## Profiling Workflow

```
1. Measure baseline
        ↓
2. Profile to find bottleneck
        ↓
3. Identify root cause
        ↓
4. Apply targeted fix
        ↓
5. Measure improvement
        ↓
   Repeat until fast enough
```

## Key Metrics

### Time Metrics
- **Wall time**: Total elapsed time
- **GPU time**: Time spent on GPU operations
- **CPU time**: Time spent on CPU (launch overhead, data prep)

### Memory Metrics
- **Peak memory**: Maximum memory allocated
- **Current memory**: Memory currently in use
- **Cached memory**: Memory held by allocator

### Utilization Metrics
- **Compute utilization**: How much compute capacity is used
- **Memory bandwidth utilization**: How much bandwidth is used
- **SM occupancy**: What fraction of SMs are active

## Common Bottlenecks

### 1. Memory Bandwidth
- **Symptom**: Low compute utilization, high memory bandwidth
- **Operations**: LayerNorm, softmax, element-wise ops
- **Fix**: Kernel fusion, reduce memory movement

### 2. Kernel Launch Overhead
- **Symptom**: Many small kernels, gaps in timeline
- **Fix**: torch.compile, batch operations, custom kernels

### 3. CPU-GPU Data Transfer
- **Symptom**: Large gaps between GPU kernels
- **Fix**: Pin memory, async transfers, prefetching

### 4. Synchronization
- **Symptom**: GPU idle while waiting for CPU
- **Fix**: Batch logging, avoid .item() calls

## Optimization Techniques

### torch.compile (PyTorch 2.0+)
```python
model = torch.compile(model)
# Automatically fuses operations, reduces memory traffic
```

### Flash Attention
```python
from torch.nn.functional import scaled_dot_product_attention
# Uses memory-efficient algorithm, O(N) memory vs O(N²)
```

### Mixed Precision
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
# 2x faster matmuls on tensor cores
```

### Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint
# Trade compute for memory
```

## Expected Output

Your profiling should produce reports like:

```
Attention Component Profiling:
========================================
Component           Time (ms)    % Total
----------------------------------------
QKV Projection      2.31         15.4%
Score Computation   5.67         37.8%
Softmax             1.23          8.2%
Weighted Sum        4.89         32.6%
Output Projection   0.90          6.0%
----------------------------------------
Total               15.00       100.0%

Bottleneck: Score Computation (37.8%)
Recommendation: Use Flash Attention
```

## Verification

All tests pass = you understand profiling and optimization!

This completes Chapter 12. You now have skills in:
- Triton kernel programming
- Kernel fusion
- JAX transformations
- Performance profiling

## Bonus Challenges (Optional)

1. Profile a full transformer layer
2. Compare PyTorch vs JAX performance
3. Implement a memory-efficient attention variant
4. Create a profiling dashboard with visualization
5. Profile on different hardware (A100 vs V100 vs T4)
