# Lab 02: KV-Cache Implementation

## Objective

Implement KV-cache for efficient autoregressive generation, reducing complexity from O(N²) to O(N).

## What You'll Build

An attention mechanism with KV-cache that:
- Caches Key and Value tensors from previous tokens
- Only computes Q, K, V for new tokens
- Achieves linear complexity for generating N tokens

## Prerequisites

Read these docs first:
- `../docs/02_kv_cache.md`
- Completed Lab 01 (memory bandwidth concepts)

## Instructions

1. Open `src/kv_cache.py`
2. Implement the functions and classes marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Classes and Functions to Implement

### `KVCache` Class

A container for storing cached K and V tensors.

```python
class KVCache:
    def __init__(self, max_seq_len, num_heads, head_dim, dtype=np.float32):
        """Initialize cache buffers."""

    def update(self, k, v, position):
        """Add new K, V at given position."""

    def get(self, end_position):
        """Get cached K, V up to position."""

    @property
    def length(self):
        """Current cache length."""

    def reset(self):
        """Clear the cache."""
```

### `attention_with_kv_cache(q, k, v, cache, position)`

Compute attention using KV-cache.
- `q`: Query for new token(s), shape `(batch, num_heads, seq_len_q, head_dim)`
- `k`, `v`: Key/Value for new token(s), shape `(batch, num_heads, seq_len_new, head_dim)`
- `cache`: KVCache instance (or None for no caching)
- `position`: Starting position for new tokens

Returns:
- `output`: Attention output, shape `(batch, num_heads, seq_len_q, head_dim)`
- `cache`: Updated KVCache

### `generate_with_cache(prompt_tokens, attention_fn, max_new_tokens)`

Generate tokens using KV-cache.
- Prefill: Process all prompt tokens, initialize cache
- Decode: Generate one token at a time using cached K, V

Returns: Generated token sequence

### `generate_without_cache(prompt_tokens, attention_fn, max_new_tokens)`

Generate tokens without KV-cache (for comparison).
- Each step recomputes attention for all tokens
- O(N²) complexity

Returns: Generated token sequence

### `count_attention_operations(seq_len, with_cache)`

Count the number of attention score computations.
- Without cache: `1 + 2 + 3 + ... + N = N(N+1)/2`
- With cache: `1 + 1 + 1 + ... = N`

Returns: Total attention computations

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_kv_cache.py::TestKVCache

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
import numpy as np
from kv_cache import KVCache, attention_with_kv_cache

# Initialize cache
cache = KVCache(max_seq_len=2048, num_heads=32, head_dim=128)

# Prefill: process prompt
prompt_q = np.random.randn(1, 32, 10, 128)  # 10 tokens
prompt_k = np.random.randn(1, 32, 10, 128)
prompt_v = np.random.randn(1, 32, 10, 128)

output, cache = attention_with_kv_cache(
    prompt_q, prompt_k, prompt_v,
    cache=cache, position=0
)
print(f"Cache length after prefill: {cache.length}")  # 10

# Decode: generate new tokens one at a time
for i in range(5):
    new_q = np.random.randn(1, 32, 1, 128)  # 1 new token
    new_k = np.random.randn(1, 32, 1, 128)
    new_v = np.random.randn(1, 32, 1, 128)

    output, cache = attention_with_kv_cache(
        new_q, new_k, new_v,
        cache=cache, position=cache.length
    )
    print(f"Cache length: {cache.length}")  # 11, 12, 13, 14, 15
```

## Hints

- The cache stores K and V, not Q (Q is always computed fresh)
- During prefill, you process all tokens at once and cache their K, V
- During decode, you process ONE new token but attend to ALL cached K, V
- Position tracking is crucial for causal masking
- Pre-allocate cache to `max_seq_len` to avoid dynamic allocation

## Expected Behavior

```
Without cache (generating 100 tokens):
  Step 1: attention on 1 token
  Step 2: attention on 2 tokens (1 redundant)
  ...
  Step 100: attention on 100 tokens (99 redundant)
  Total attention ops: 5050

With cache (generating 100 tokens):
  Step 1: attention on 1 token, cache K1, V1
  Step 2: attention Q2 against [K1,K2], [V1,V2]
  ...
  Step 100: attention Q100 against [K1...K100], [V1...V100]
  Total attention ops: 100
```

## Verification

All tests pass = you've correctly implemented KV-cache!

The key insight: With KV-cache, generating N tokens is O(N), not O(N²).
