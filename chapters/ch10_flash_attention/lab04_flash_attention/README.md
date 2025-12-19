# Lab 04: Flash Attention Integration

## Objective

Learn to use the Flash Attention library in practice.

## What You'll Build

A `FlashAttentionWrapper` class that:
1. Integrates the flash-attn library
2. Compares performance with standard attention
3. Demonstrates practical usage patterns

## Prerequisites

- Complete Labs 01-03
- Read `../docs/04_flash_attention_algorithm.md`
- Read `../docs/05_flash_attention_v2_v3.md`

## Why Flash Attention?

After understanding the algorithm (Labs 01-03), it's time to use the optimized implementation:
- flash-attn: Official CUDA implementation
- 2-4x faster than PyTorch attention
- Sub-linear memory scaling
- Supports causal masking, dropout, multi-head attention

## Instructions

1. Open `src/flash_attention.py`
2. Implement the wrapper functions
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `has_flash_attention()`

Check if flash-attn is available.

```python
def has_flash_attention() -> bool:
    """Check if flash-attn library is available."""
```

### `FlashAttentionWrapper`

```python
class FlashAttentionWrapper:
    def __init__(self, dropout_p: float = 0.0, causal: bool = False):
        """
        Initialize Flash Attention wrapper.

        Args:
            dropout_p: Dropout probability
            causal: Whether to use causal masking
        """

    def forward(self, Q, K, V):
        """
        Compute attention using Flash Attention.

        Args:
            Q: (batch, seq_len, num_heads, d_head)
            K: (batch, seq_len, num_heads, d_head)
            V: (batch, seq_len, num_heads, d_head)

        Returns:
            output: (batch, seq_len, num_heads, d_head)
        """
```

### `benchmark_attention()`

```python
def benchmark_attention(
    seq_len: int,
    d_model: int,
    num_heads: int,
    batch_size: int = 1,
    num_iterations: int = 10
) -> dict:
    """
    Benchmark Flash Attention vs standard attention.

    Returns:
        Dictionary with timing and memory results
    """
```

### `compare_outputs()`

```python
def compare_outputs(
    Q, K, V,
    causal: bool = False
) -> dict:
    """
    Compare Flash Attention output with standard attention.

    Returns:
        Dictionary with max_diff, mean_diff, allclose
    """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_flash_attention.py::TestFlashWrapper

# Run with verbose output
uv run pytest tests/ -v
```

## Notes on GPU Requirements

Flash Attention requires:
- NVIDIA GPU with compute capability >= 7.5 (Turing+)
- CUDA 11.6+
- PyTorch with CUDA support

The tests will skip gracefully if GPU is not available.

## Expected Results

```python
# Flash Attention should be faster
results = benchmark_attention(seq_len=2048, d_model=512, num_heads=8)
print(f"Standard: {results['standard_time_ms']:.2f}ms")
print(f"Flash: {results['flash_time_ms']:.2f}ms")
print(f"Speedup: {results['speedup']:.2f}x")

# Output should match (within tolerance)
comparison = compare_outputs(Q, K, V)
assert comparison['allclose'], "Flash output should match standard"
```

## Integration Patterns

### Multi-Head Attention

```python
# Reshape for flash attention: (B, S, H, D)
Q = Q.reshape(batch, seq_len, num_heads, d_head)
K = K.reshape(batch, seq_len, num_heads, d_head)
V = V.reshape(batch, seq_len, num_heads, d_head)

wrapper = FlashAttentionWrapper(causal=True)
output = wrapper.forward(Q, K, V)

# Reshape back: (B, S, H*D)
output = output.reshape(batch, seq_len, num_heads * d_head)
```

### With Dropout (Training)

```python
wrapper = FlashAttentionWrapper(dropout_p=0.1, causal=True)
# Dropout is only applied during training
output = wrapper.forward(Q, K, V)
```
