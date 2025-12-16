# Lab 03: Multi-Head Attention

## Objective

Implement multi-head attention, the core building block of transformers.

## What You'll Build

A `MultiHeadAttention` class that:
1. Projects input to multiple heads
2. Computes attention in parallel across heads
3. Concatenates and projects back to model dimension

## Prerequisites

- Complete Lab 01 (dot-product attention)
- Read `../docs/03_multihead_attention.md`

## Why Multiple Heads?

Single-head attention can only learn one attention pattern. Multiple heads allow the model to:
- Attend to different positions simultaneously
- Learn different types of relationships (syntactic, semantic, positional)
- Increase model capacity without proportionally increasing computation

## Instructions

1. Open `src/multihead.py`
2. Implement the `MultiHeadAttention` class
3. Run tests: `uv run pytest tests/`

## The Math

For input X of shape `(seq_len, d_model)`:

```
1. Project to Q, K, V for each head:
   Q = X @ W_Q  →  reshape to (num_heads, seq_len, d_k)
   K = X @ W_K  →  reshape to (num_heads, seq_len, d_k)
   V = X @ W_V  →  reshape to (num_heads, seq_len, d_v)

2. Compute attention for each head in parallel:
   head_i = Attention(Q_i, K_i, V_i)

3. Concatenate heads:
   concat = [head_0, head_1, ..., head_h]  shape: (seq_len, num_heads * d_v)

4. Final projection:
   output = concat @ W_O  shape: (seq_len, d_model)
```

## Class to Implement

### `MultiHeadAttention`

```python
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as x
        """
```

## Implementation Steps

1. **Initialize weight matrices:**
   - `W_Q`: shape `(d_model, d_model)`
   - `W_K`: shape `(d_model, d_model)`
   - `W_V`: shape `(d_model, d_model)`
   - `W_O`: shape `(d_model, d_model)`

2. **In forward pass:**
   - Project: `Q = x @ W_Q`, etc.
   - Reshape to separate heads: `(batch, seq_len, num_heads, d_k)` → `(batch, num_heads, seq_len, d_k)`
   - Apply scaled dot-product attention to each head
   - Reshape back: `(batch, num_heads, seq_len, d_k)` → `(batch, seq_len, num_heads * d_k)`
   - Final projection: `output = concat @ W_O`

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_multihead.py::TestMultiHeadInit

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- `d_k = d_v = d_model // num_heads`
- Use `transpose` or `swapaxes` to move the head dimension
- For batched input, remember to handle both 2D and 3D inputs
- Initialize weights with small random values (e.g., `np.random.randn(...) * 0.02`)

## Expected Shapes (batch_size=2, seq_len=4, d_model=8, num_heads=2)

```
Input x:          (2, 4, 8)

After W_Q:        (2, 4, 8)
After reshape:    (2, 4, 2, 4)  # (batch, seq, heads, d_k)
After transpose:  (2, 2, 4, 4)  # (batch, heads, seq, d_k)

After attention:  (2, 2, 4, 4)  # (batch, heads, seq, d_v)
After transpose:  (2, 4, 2, 4)  # (batch, seq, heads, d_v)
After reshape:    (2, 4, 8)     # (batch, seq, d_model)

After W_O:        (2, 4, 8)     # Final output
```

## Verification

All tests pass = you've correctly implemented multi-head attention!

Next up: Lab 04 where you'll verify your implementation matches PyTorch.
