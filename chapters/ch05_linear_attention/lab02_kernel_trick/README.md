# Lab 02: The Kernel Trick

## Objective

Implement the kernel trick that enables O(n) attention by changing the order of operations.

## What You'll Build

Functions that demonstrate:
1. Standard attention: `(QK^T)V` - O(n²)
2. Linearized attention: `Q(K^T V)` - O(nd²)
3. Verification that both produce equivalent results (with the right feature map)

## Prerequisites

- Complete Lab 01 (complexity analysis)
- Read `../docs/02_kernel_trick.md`

## The Key Insight

Matrix multiplication is associative: `(AB)C = A(BC)`

Standard attention computes:
```
output = softmax(QK^T / √d) @ V
       = (n×n) @ (n×d) → O(n²d)
```

If we could avoid the softmax, we could compute:
```
output = Q @ (K^T @ V)
       = (n×d) @ (d×d) @ → O(nd²)
```

Since typically d << n, this is much faster for long sequences!

## Instructions

1. Open `src/kernel_trick.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `standard_attention_order(Q, K, V)`

Compute attention in the standard order: `(QK^T)V`
- Explicitly form the (n, n) attention matrix
- Return both output and the attention matrix

### `linear_attention_order(Q, K, V, feature_map)`

Compute attention in the linear order: `Q(K^T V)`
- Apply feature map to Q and K first
- Never form the (n, n) matrix explicitly
- Return the output

### `simple_feature_map(x)`

A simple feature map: `φ(x) = x` (identity)
- This won't match softmax attention, but demonstrates the concept

### `elu_feature_map(x)`

The ELU+1 feature map: `φ(x) = ELU(x) + 1`
- Ensures positive values
- Used in the original Linear Transformer paper

### `verify_associativity(Q, K, V)`

Verify that `(QK^T)V = Q(K^T V)` for simple matrix multiplication.
- This shows the math works before we add feature maps

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_kernel_trick.py::TestAssociativity

# Run with verbose output
uv run pytest tests/ -v
```

## The Math in Detail

**Standard attention** (ignoring softmax for now):
```
scores = Q @ K.T          # (n, d) @ (d, n) = (n, n)  ← O(n²d)
output = scores @ V       # (n, n) @ (n, d) = (n, d)  ← O(n²d)
Total: O(n²d)
```

**Linear attention**:
```
KV = K.T @ V              # (d, n) @ (n, d) = (d, d)  ← O(nd²)
output = Q @ KV           # (n, d) @ (d, d) = (n, d)  ← O(nd²)
Total: O(nd²)
```

For d=64 and n=4096:
- Standard: 4096² × 64 = 1.07 billion ops
- Linear: 4096 × 64² = 16.8 million ops
- **Speedup: 64x!**

## Expected Behavior

```python
# These should be mathematically equivalent (no feature map)
Q, K, V = random_tensors(seq_len=100, d_model=64)

# Standard order - forms (n,n) matrix
out1, attn_matrix = standard_attention_order(Q, K, V)
print(attn_matrix.shape)  # (100, 100)

# Linear order - never forms (n,n) matrix
out2 = linear_attention_order(Q, K, V, feature_map=identity)

# With identity feature map and no normalization, these should match
np.testing.assert_allclose(out1, out2, rtol=1e-5)
```

## Hints

- For `linear_attention_order`, apply feature_map to Q and K before computing
- Remember: `K.T @ V` gives you a (d, d) matrix
- Use `np.einsum` for clarity if matrix multiplication gets confusing
- The ELU function: `ELU(x) = x if x > 0 else exp(x) - 1`

## Why Feature Maps?

The trick only works if we can separate the interaction between Q and K:
- Softmax(Q @ K.T) can't be factored
- But if similarity(q, k) = φ(q) · φ(k), then we can use associativity

Different feature maps approximate softmax to different degrees.

## Verification

All tests pass = you understand the kernel trick!

This is the mathematical foundation for all linear attention methods.
