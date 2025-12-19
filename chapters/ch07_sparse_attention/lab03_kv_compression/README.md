# Lab 03: KV Compression

## Objective

Implement low-rank Key-Value compression to reduce the KV-cache memory footprint.

## What You'll Build

Functions and classes that:
1. Compress K and V into a lower-dimensional latent space
2. Implement Grouped-Query Attention (GQA)
3. Compare memory usage between approaches

## Prerequisites

- Complete Lab 01 and Lab 02
- Read `../docs/04_kv_compression.md`

## Why KV Compression?

During autoregressive generation, we cache K and V for all past tokens:
- Standard MHA: 2 × layers × seq_len × num_heads × head_dim
- For 70B models with 32K context, this can be **100GB+** just for KV-cache!

Compression techniques reduce this dramatically:
- **GQA**: Share KV heads across multiple query heads (4-8x reduction)
- **Low-rank**: Compress KV to a smaller latent dimension (4-8x reduction)
- **Combined**: 16-64x reduction possible

## Instructions

1. Open `src/kv_compression.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `compute_kv_cache_size(seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes=2)`

Calculate the KV-cache size in bytes.

```python
# Example: Llama-2 7B style
>>> compute_kv_cache_size(
...     seq_len=4096,
...     num_layers=32,
...     num_kv_heads=32,
...     head_dim=128,
...     dtype_bytes=2  # fp16
... )
2147483648  # 2GB
```

### `grouped_query_attention(Q, K, V, num_kv_groups)`

Implement GQA where multiple query heads share the same KV head.

```
Standard MHA:
  Q heads: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]
  K heads: [K1, K2, K3, K4, K5, K6, K7, K8]  (8 KV heads)

GQA with 2 groups (4 Q heads per KV head):
  Q heads: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]
  K heads: [K1, K1, K1, K1, K2, K2, K2, K2]  (2 KV heads)
```

### `compress_kv(K, V, d_latent)`

Compress K and V to a lower-dimensional latent space using learned projections.

### `decompress_kv(kv_latent, W_uk, W_uv)`

Decompress latent back to K and V using up-projection matrices.

## Classes to Implement

### `GroupedQueryAttention`

```python
class GroupedQueryAttention:
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,  # < num_q_heads
    ):
        """
        GQA where num_q_heads query heads share num_kv_heads KV heads.

        Args:
            d_model: Model dimension
            num_q_heads: Number of query heads
            num_kv_heads: Number of KV heads (must divide num_q_heads)
        """
```

### `LowRankKVAttention`

```python
class LowRankKVAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,  # Compressed dimension
    ):
        """
        Attention with low-rank KV compression.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_latent: Dimension of compressed KV representation
        """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_kv_compression.py::TestGQA

# Run with verbose output
uv run pytest tests/ -v
```

## Memory Savings Analysis

Implement and verify memory savings:

```python
# Standard MHA
mha_cache = compute_kv_cache_size(4096, 32, 32, 128, 2)
# = 2 × 32 × 4096 × 32 × 128 × 2 = 2GB

# GQA with 4 KV heads (8x reduction)
gqa_cache = compute_kv_cache_size(4096, 32, 4, 128, 2)
# = 256MB

# Low-rank with d_latent=256 (8x reduction from d_model=2048)
lowrank_cache = compute_kv_cache_size(4096, 32, 1, 256, 2)
# Custom calculation needed
```

## Hints

- For GQA, use `repeat_interleave` to expand KV to match Q heads
- For low-rank, the compression is: `kv_latent = concat(K, V) @ W_down`
- The decompression is: `K, V = split(kv_latent @ W_up)`
- Quality depends on choosing d_latent large enough to preserve information

## Expected Behavior

```python
# GQA: 8 query heads, 2 KV heads
gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)
x = np.random.randn(10, 64)
output = gqa(x)
# Each group of 4 query heads shares the same KV

# Low-rank: compress 512-dim KV to 128-dim latent
lowrank = LowRankKVAttention(d_model=512, num_heads=8, d_latent=128)
x = np.random.randn(10, 512)
output = lowrank(x)
# KV-cache is 4x smaller
```

## Verification

All tests pass = you understand KV compression techniques!

These techniques are used in production models:
- **GQA**: Llama-2 70B, Mistral, Gemma
- **Low-rank KV**: DeepSeek-V2 (combined with GQA as MLA)

Next: Lab 04 combines these ideas into DeepSeek's Multi-head Latent Attention.
