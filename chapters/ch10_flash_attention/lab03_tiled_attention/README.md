# Lab 03: Tiled Attention

## Objective

Implement block-by-block (tiled) attention computation using online softmax.

## What You'll Build

A `TiledAttention` class that:
1. Processes Q, K, V in blocks that fit in simulated "SRAM"
2. Uses online softmax to compute correct results incrementally
3. Never materializes the full N×N attention matrix

## Prerequisites

- Complete Lab 02 (online softmax)
- Read `../docs/02_tiling_and_blocking.md`

## Why Tiling Matters

Standard attention needs O(N²) memory for the attention matrix.
With tiling:
- We process attention in B_r × B_c blocks
- Each block fits in fast on-chip memory (SRAM)
- We use online softmax to accumulate results correctly
- Total memory: O(N) instead of O(N²)!

## Instructions

1. Open `src/tiled_attention.py`
2. Implement the `TiledAttention` class
3. Run tests: `uv run pytest tests/`

## The Algorithm

```python
# Initialize output and statistics
O = zeros(N, d)        # Output accumulator
l = zeros(N)           # Running softmax denominator
m = full(N, -inf)      # Running max

# Outer loop: K, V blocks
for j in range(0, N, B_c):
    K_j = K[j:j+B_c]
    V_j = V[j:j+B_c]

    # Inner loop: Q blocks
    for i in range(0, N, B_r):
        Q_i = Q[i:i+B_r]

        # Compute block attention scores
        S_ij = Q_i @ K_j.T / sqrt(d_k)

        # Update online softmax stats and accumulate output
        # (This is where online_softmax magic happens)
        m[i:i+B_r], l[i:i+B_r], O[i:i+B_r] = update_block(...)

# Final normalization
O = O / l[:, None]
```

## Class to Implement

### `TiledAttention`

```python
class TiledAttention:
    def __init__(self, block_size_q: int = 32, block_size_kv: int = 32):
        """
        Initialize tiled attention.

        Args:
            block_size_q: Block size for Q (B_r)
            block_size_kv: Block size for K, V (B_c)
        """

    def forward(self, Q, K, V, mask=None):
        """
        Compute attention using tiled algorithm.

        Never stores full N×N attention matrix!

        Returns:
            output: Same as standard attention
        """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_tiled_attention.py::TestTiledCorrectness

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use your online softmax code from Lab 02
- Process Q blocks in the inner loop, K/V blocks in the outer loop
- Track per-row statistics (m, l) as vectors of shape (N,)
- The output accumulator O gets rescaled when max changes

## Expected Results

Tiled attention should match standard attention exactly:

```python
tiled = TiledAttention(block_size_q=32, block_size_kv=32)
standard_out, _ = standard_attention(Q, K, V)
tiled_out = tiled.forward(Q, K, V)

np.testing.assert_allclose(tiled_out, standard_out, atol=1e-5)
```

But with O(N) memory instead of O(N²)!

## Memory Verification

```python
# For seq_len=4096:
# Standard: stores 4096×4096 = 16M attention weights
# Tiled: stores at most block_size_q × block_size_kv per iteration

# Check memory during forward pass
# (The test suite verifies this)
```

## Causal Masking Optimization

For causal attention (decoder), we can skip upper-triangular blocks entirely:

```
Block (i, j) where i * B_r < j * B_c:
  All positions are masked (query can't see future keys)
  Skip this block entirely!

This saves ~50% compute for causal attention.
```

## Verification

All tests pass = you've implemented the core of Flash Attention!

This tiled algorithm is exactly what Flash Attention does, just without the CUDA kernel optimizations.
