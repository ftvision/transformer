# Lab 01: Complexity Analysis

## Objective

Benchmark standard attention to understand the O(n²) complexity bottleneck firsthand.

## What You'll Build

Functions to:
1. Measure attention computation time for different sequence lengths
2. Measure memory usage for different sequence lengths
3. Fit complexity curves to verify O(n²) scaling
4. Visualize the "complexity wall"

## Prerequisites

Read these docs first:
- `../docs/01_quadratic_bottleneck.md`

## Why This Matters

Before optimizing something, you need to understand the problem. This lab will:
- Give you concrete numbers for attention complexity
- Show where the bottleneck appears
- Motivate why linear attention matters

## Instructions

1. Open `src/complexity.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `standard_attention(Q, K, V)`

Implement basic scaled dot-product attention (no mask) for benchmarking.
- Use the formula: `softmax(QK^T / √d_k) V`
- This is the baseline we're measuring

### `measure_attention_time(seq_len, d_model, num_runs=10)`

Measure the average time to compute attention for a given sequence length.
- Create random Q, K, V tensors
- Time multiple runs and average
- Return mean and std of execution time

### `measure_attention_memory(seq_len, d_model)`

Estimate the memory used by the attention matrix.
- The attention matrix is (seq_len, seq_len)
- Return memory in bytes (assuming float32)

### `fit_complexity_curve(seq_lengths, times)`

Fit a polynomial to the timing data to verify O(n²) scaling.
- Fit times = a * seq_len^b
- Return the exponent b (should be close to 2.0)

### `find_max_seq_length(memory_limit_mb, d_model)`

Find the maximum sequence length that fits in a given memory budget.
- Consider only the attention matrix memory
- Return the maximum sequence length

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_complexity.py::TestStandardAttention

# Run with verbose output
uv run pytest tests/ -v
```

## Expected Results

When you run the benchmarks, you should observe:
- Time grows quadratically with sequence length
- Memory grows quadratically with sequence length
- There's a clear "wall" where computation becomes impractical

Example output:
```
seq_len=512:   time=2.1ms,   memory=1.0 MB
seq_len=1024:  time=8.5ms,   memory=4.0 MB
seq_len=2048:  time=34.0ms,  memory=16.0 MB
seq_len=4096:  time=136.0ms, memory=64.0 MB
```

Notice: doubling seq_len → ~4x time and memory (O(n²) confirmed!)

## Hints

- Use `time.perf_counter()` for timing
- For memory, calculate `seq_len * seq_len * 4` bytes (float32)
- Use `np.polyfit` with log-log transform to fit the exponent
- Warm up with a few runs before timing

## Verification

All tests pass = you understand the O(n²) bottleneck!

This motivates everything in the rest of Chapter 5: making attention linear.
