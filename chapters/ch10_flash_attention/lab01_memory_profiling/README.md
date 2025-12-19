# Lab 01: Memory Profiling

## Objective

Profile standard attention memory usage to understand the O(N²) memory bottleneck.

## What You'll Build

Functions to:
1. Measure memory usage of standard attention
2. Profile memory at different sequence lengths
3. Visualize the quadratic memory scaling

## Prerequisites

Read these docs first:
- `../docs/01_gpu_memory_hierarchy.md`

## Instructions

1. Open `src/profiling.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `standard_attention(Q, K, V, mask=None)`
Implement standard attention that stores the full attention matrix.
- This is intentionally memory-inefficient (for comparison)
- Returns output and the full attention weights matrix

### `measure_attention_memory(seq_len, d_model, batch_size=1)`
Measure the memory required for standard attention.
- Create random Q, K, V matrices
- Compute attention
- Return memory statistics (input size, attention matrix size, total)

### `profile_memory_scaling(seq_lengths, d_model=64)`
Profile how memory scales with sequence length.
- Run `measure_attention_memory` for each sequence length
- Return dict mapping seq_len to memory stats

### `estimate_attention_memory(seq_len, d_model, dtype_bytes=4)`
Theoretically estimate memory usage.
- Q, K, V: 3 × seq_len × d_model × dtype_bytes
- Attention matrix: seq_len × seq_len × dtype_bytes
- Output: seq_len × d_model × dtype_bytes

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_profiling.py::TestStandardAttention

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use `np.float32` for realistic memory measurements
- Memory in bytes = array.nbytes
- The attention matrix is the memory bottleneck!

## Expected Results

For seq_len=4096, d_model=64, batch=1:
- Q, K, V: 3 × 4096 × 64 × 4 bytes = ~3 MB
- Attention matrix: 4096 × 4096 × 4 bytes = ~64 MB
- The attention matrix dominates!

```
Sequence Length vs Memory:
seq_len=512:   ~1 MB (attention matrix)
seq_len=1024:  ~4 MB
seq_len=2048:  ~16 MB
seq_len=4096:  ~64 MB
seq_len=8192:  ~256 MB ← This is where it hurts!
```

## Verification

All tests pass = you understand the memory problem that Flash Attention solves!

This is the "before" picture. Later labs will show the "after" with tiling and Flash Attention.
