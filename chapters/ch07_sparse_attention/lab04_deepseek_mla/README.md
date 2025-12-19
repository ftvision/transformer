# Lab 04: DeepSeek MLA (Multi-head Latent Attention)

## Objective

Implement DeepSeek's Multi-head Latent Attention, which combines low-rank KV compression with multi-head attention for dramatic memory savings.

## What You'll Build

A `MultiHeadLatentAttention` class that:
1. Compresses Key-Value pairs into a compact latent representation
2. Caches only the latent during generation (not full K, V)
3. Decompresses on-the-fly during attention computation
4. Achieves 4-8x KV-cache reduction with minimal quality loss

## Prerequisites

- Complete Lab 03 (KV compression basics)
- Read `../docs/03_deepseek_mla.md`

## Why MLA?

Standard attention caches K and V for all past tokens:
```
Standard: cache = [K1, V1, K2, V2, ..., Kn, Vn]
Size: 2 × seq × heads × head_dim

MLA: cache = [c1, c2, ..., cn]  (compressed latent)
Size: seq × d_latent

Reduction: (2 × heads × head_dim) / d_latent ≈ 4-8x
```

DeepSeek-V2 uses MLA to enable 128K context with manageable memory.

## Instructions

1. Open `src/mla.py`
2. Implement the `MultiHeadLatentAttention` class
3. Run tests: `uv run pytest tests/`

## The MLA Architecture

```
Input X: (seq_len, d_model)
    │
    ├─────────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
Q = X @ W_Q                           c_KV = X @ W_DKV
(Full Q projection)                   (Compress to latent)
    │                                         │
    │                                    ┌────┴────┐
    │                                    │ Cache   │
    │                                    │ c_KV    │
    │                                    └────┬────┘
    │                                         │
    │                                    ┌────┴────┐
    │                                    ▼         ▼
    │                               K = c @ W_UK   V = c @ W_UV
    │                               (Decompress)   (Decompress)
    │                                    │         │
    ▼                                    ▼         ▼
    └──────────► Attention(Q, K, V) ◄────┴─────────┘
                        │
                        ▼
                    Output
```

## Class to Implement

### `MultiHeadLatentAttention`

```python
class MultiHeadLatentAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        head_dim: int = None
    ):
        """
        Initialize MLA.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_latent: Dimension of compressed KV latent
            head_dim: Dimension per head (default: d_model // num_heads)
        """

    def forward(
        self,
        x: np.ndarray,
        kv_cache: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MLA attention.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            kv_cache: Cached latent from previous tokens

        Returns:
            output: Attention output
            new_cache: Updated latent cache
        """
```

## The Math

### Compression

```
c_KV = X @ W_DKV

Where:
- X: (seq_len, d_model)
- W_DKV: (d_model, d_latent)
- c_KV: (seq_len, d_latent)
```

### Decompression

```
K = c_KV @ W_UK
V = c_KV @ W_UV

Where:
- c_KV: (seq_len, d_latent)
- W_UK: (d_latent, num_heads × head_dim)
- W_UV: (d_latent, num_heads × head_dim)
```

### Why This Works

The compression assumes K and V have low-rank structure:
- Information in K/V is correlated across heads
- A smaller latent can capture the essential information
- The model learns optimal compression during training

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_mla.py::TestMLAInit

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- d_latent should be significantly smaller than `num_heads × head_dim`
- Initialize weights with small random values
- For incremental decoding, cache grows by 1 row per token
- The cache stores the LATENT, not the full K/V

## Expected Behavior

```python
# MLA with 8x compression
mla = MultiHeadLatentAttention(
    d_model=512,
    num_heads=8,
    d_latent=128  # vs 512 for full KV
)

# Prefill: process full sequence
x = np.random.randn(2, 100, 512)
output, cache = mla(x)
# cache shape: (2, 100, 128)

# Decode: process one new token, reuse cache
x_new = np.random.randn(2, 1, 512)
output_new, cache_new = mla(x_new, kv_cache=cache)
# cache_new shape: (2, 101, 128)
```

## Memory Analysis

Compare with standard attention:

```python
# Standard MHA
kv_size = 2 × seq_len × num_heads × head_dim × dtype_bytes
       = 2 × 4096 × 8 × 64 × 2 = 8MB per layer

# MLA with d_latent=128
cache_size = seq_len × d_latent × dtype_bytes
          = 4096 × 128 × 2 = 1MB per layer

# 8x reduction!
```

## Milestone

**Chapter 7 Milestone**: Your MLA implementation should:
1. Reduce KV-cache by at least 4x compared to standard MHA
2. Produce valid attention outputs (weights sum to 1, correct shapes)
3. Support incremental decoding with cache accumulation

## Verification

All tests pass = you've implemented DeepSeek MLA!

This technique is at the core of DeepSeek-V2's efficiency, enabling it to handle 128K context with practical memory requirements.

Next: Lab 05 explores Mixture-of-Experts for sparse feed-forward computation.
