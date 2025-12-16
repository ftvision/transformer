# Lab 01: Dot-Product Attention

## Objective

Implement scaled dot-product attention from scratch using NumPy.

## What You'll Build

A function that computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

## Prerequisites

Read these docs first:
- `../docs/01_attention_intuition.md`
- `../docs/02_scaled_dot_product.md`

## Instructions

1. Open `src/attention.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `softmax(x, axis=-1)`
Compute softmax along the specified axis.
- Remember to subtract max for numerical stability!
- `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

### `scaled_dot_product_attention(Q, K, V, mask=None)`
Compute attention output and weights.
1. Compute scores: `QK^T`
2. Scale by `√d_k`
3. Apply mask if provided (set masked positions to -inf before softmax)
4. Apply softmax to get attention weights
5. Compute output: `attention_weights @ V`

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_attention.py::TestSoftmax

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- `d_k` is the last dimension of Q (or K)
- Use `np.einsum` or `@` for matrix multiplication
- For masking, use `-np.inf` (softmax of -inf = 0)
- Check your shapes at each step!

## Expected Shapes

```
Q: (seq_len, d_k) or (batch, seq_len, d_k)
K: (seq_len, d_k) or (batch, seq_len, d_k)
V: (seq_len, d_v) or (batch, seq_len, d_v)

attention_weights: (seq_len, seq_len) or (batch, seq_len, seq_len)
output: (seq_len, d_v) or (batch, seq_len, d_v)
```

## Verification

All tests pass = you've correctly implemented attention!

If stuck, check `solutions/attention.py` for the reference implementation.
