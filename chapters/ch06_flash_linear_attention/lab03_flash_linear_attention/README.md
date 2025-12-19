# Lab 03: Flash Linear Attention

## Objective

Implement a memory-efficient version of linear attention that demonstrates the core principles of Flash Linear Attention.

## What You'll Build

A memory-efficient linear attention implementation that:
1. Processes data in tiles/chunks
2. Avoids materializing large intermediate tensors
3. Uses recomputation for memory efficiency

## Prerequisites

- Complete Lab 01 (RNN view)
- Complete Lab 02 (Chunkwise parallel)
- Read `../docs/03_flash_linear_attention.md`

## Why This Matters

The key innovations of Flash Linear Attention:
1. **Tiled computation**: Process chunks in fast SRAM instead of slow HBM
2. **State checkpointing**: Only save states at chunk boundaries
3. **Recomputation**: Trade compute for memory in backward pass

## Instructions

1. Open `src/flash_linear.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `tiled_forward(Q, K, V, chunk_size, feature_map_fn)`

Memory-efficient forward pass:
- Process one chunk at a time
- Keep running state in memory
- Save only boundary states for backward

### `FlashLinearAttention` class

A class implementing:
- `forward()`: Memory-efficient forward pass
- `get_saved_states()`: States at chunk boundaries
- Memory tracking utilities

### `measure_memory(fn, *args)`

Utility to measure memory usage of a function.
(Demonstrates the memory savings)

### `flash_vs_naive_memory(Q, K, V, chunk_size)`

Compare memory usage of flash vs naive implementations.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_flash_linear.py::TestTiledForward

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Think about what needs to be saved for the backward pass
- Only intermediate states at chunk boundaries are needed
- Memory savings come from not materializing the full cumsum tensor

## Memory Analysis

Naive implementation:
```
KV = einsum('bnd,bnv->bndv', K, V)  # O(n × d_k × d_v) memory!
S = cumsum(KV, dim=1)  # Still O(n × d_k × d_v)
```

Flash implementation:
```
For each chunk:
    - Only chunk's KV in memory: O(C × d_k × d_v)
    - Running state: O(d_k × d_v)
Saved for backward:
    - States at boundaries: O(n/C × d_k × d_v)
```

Total: O(n/C × d²) instead of O(n × d²)!

## Expected Memory Savings

For seq_len=1024, d=64, chunk_size=64:
- Naive: ~256 MB for cumsum tensor
- Flash: ~4 MB for boundary states + ~1 MB working memory

That's a ~50x reduction!

## Verification

All tests pass = you understand Flash Linear Attention's memory efficiency!

Key insight: This is the same algorithm used in production implementations like the `fla` library.
