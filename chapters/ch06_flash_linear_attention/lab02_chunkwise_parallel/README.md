# Lab 02: Chunkwise Parallel Linear Attention

## Objective

Implement the chunkwise parallel algorithm that combines the best of parallel and recurrent computation.

## What You'll Build

A function that processes sequences in chunks:
- **Within chunks**: Parallel computation (GPU-friendly)
- **Between chunks**: Sequential state passing (minimal overhead)

## Prerequisites

- Complete Lab 01 (RNN view)
- Read `../docs/02_chunkwise_parallel.md`

## Why This Matters

Pure parallel: O(n²) memory for cumulative sum tensor
Pure recurrent: O(n) time but sequential (slow on GPU)

Chunkwise: O(C²) memory per chunk + O(d²) state = best of both!

## Instructions

1. Open `src/chunkwise.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `chunk_sequence(x, chunk_size)`

Split a sequence into chunks of fixed size:
```
[1, 2, 3, 4, 5, 6, 7, 8] with chunk_size=3
→ [[1, 2, 3], [4, 5, 6], [7, 8, PAD]]
```

### `intra_chunk_attention(Q_chunk, K_chunk, V_chunk, feature_map_fn)`

Compute causal linear attention within a single chunk.
Returns both the output and the chunk's contribution to state.

### `inter_chunk_contribution(Q_chunk, state, feature_map_fn)`

Compute the contribution from previous chunks' accumulated state.
```
output = φ(Q_chunk) @ state  # state from all previous chunks
```

### `chunkwise_linear_attention(Q, K, V, chunk_size, feature_map_fn)`

Full chunkwise algorithm:
```
for each chunk:
    1. Compute intra-chunk attention (parallel within chunk)
    2. Add inter-chunk contribution (from accumulated state)
    3. Update state for next chunk
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_chunkwise.py::TestChunkSequence

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use `np.pad` to pad sequences to multiples of chunk_size
- Intra-chunk attention is just linear attention on a small chunk
- State update: `new_state = old_state + sum(φ(K_chunk)^T @ V_chunk)`
- Final output = intra_chunk_output + inter_chunk_output

## Expected Shapes

```
Input: (batch, seq_len, d)
Chunks: (batch, num_chunks, chunk_size, d)
State: (batch, d_k, d_v)
Output: (batch, seq_len, d)
```

## Complexity Analysis

For sequence length n, chunk size C, dimension d:

| Component | Time | Memory |
|-----------|------|--------|
| Intra-chunk | O(n × C × d) | O(C²) per chunk |
| Inter-chunk | O(n × d²) | O(d²) |
| State passing | O(n/C × d²) | O(d²) |
| **Total** | O(n × (C×d + d²)) | O(C² + d²) |

Choose C ≈ √d for optimal balance.

## Verification

All tests pass = you understand chunkwise parallel computation!

Key insight: This is the foundation for Flash Linear Attention's memory efficiency.
