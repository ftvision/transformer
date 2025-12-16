# Lab 04: PyTorch Comparison

## Objective

Verify your multi-head attention implementation by matching PyTorch's `nn.MultiheadAttention` output exactly.

## What You'll Build

A wrapper that:
1. Takes your NumPy MHA implementation from Lab 03
2. Loads weights from PyTorch's `nn.MultiheadAttention`
3. Produces identical outputs (within numerical tolerance)

## Prerequisites

- Complete Lab 03 (multi-head attention)
- Basic PyTorch knowledge (tensors, modules)

## Why This Matters

Matching a reference implementation:
- Validates your understanding of the algorithm
- Catches subtle bugs (wrong transpose, missing scaling, etc.)
- Builds confidence before using your code in larger systems

## Instructions

1. Open `src/comparison.py`
2. Implement the functions to load PyTorch weights and compare outputs
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `load_weights_from_pytorch(numpy_mha, pytorch_mha)`

Load weights from PyTorch MHA into your NumPy implementation.

PyTorch stores weights differently:
- `in_proj_weight`: Combined [W_Q, W_K, W_V] of shape `(3 * d_model, d_model)`
- `out_proj.weight`: W_O of shape `(d_model, d_model)`

You need to:
1. Split `in_proj_weight` into W_Q, W_K, W_V
2. Handle the transpose (PyTorch uses different convention)
3. Copy to your NumPy arrays

### `compare_outputs(numpy_mha, pytorch_mha, x, atol=1e-5)`

Compare outputs of both implementations.

Returns True if outputs match within tolerance.

### `create_matching_mha(d_model, num_heads)`

Create a NumPy MHA with weights copied from a new PyTorch MHA.

Useful for quick testing.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v
```

## Understanding PyTorch's MHA

PyTorch's `nn.MultiheadAttention` has these parameters:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(
    embed_dim=512,     # d_model
    num_heads=8,
    batch_first=True   # IMPORTANT: Use this for (batch, seq, dim) format
)

# Weights:
# mha.in_proj_weight: (3 * embed_dim, embed_dim) = (1536, 512)
# mha.in_proj_bias: (3 * embed_dim,) = (1536,) - we ignore bias
# mha.out_proj.weight: (embed_dim, embed_dim) = (512, 512)
# mha.out_proj.bias: (embed_dim,) = (512,) - we ignore bias
```

The `in_proj_weight` is laid out as:
```
in_proj_weight = [W_Q]  # rows 0 to embed_dim
                 [W_K]  # rows embed_dim to 2*embed_dim
                 [W_V]  # rows 2*embed_dim to 3*embed_dim
```

## Common Pitfalls

1. **Transpose confusion**: PyTorch uses (out_features, in_features), NumPy typically uses (in_features, out_features)

2. **Batch dimension**: Make sure both use `batch_first=True` or handle the transpose

3. **Bias terms**: For simplicity, we ignore biases. Set them to zero in PyTorch if needed.

4. **Numerical precision**: Use `atol=1e-5` for float32 comparisons

## Example Usage

```python
import torch
import torch.nn as nn
import numpy as np

# Create PyTorch MHA
pytorch_mha = nn.MultiheadAttention(
    embed_dim=64,
    num_heads=8,
    batch_first=True,
    bias=False  # No bias for simpler comparison
)

# Create your NumPy MHA
from multihead import MultiHeadAttention
numpy_mha = MultiHeadAttention(d_model=64, num_heads=8)

# Load weights
load_weights_from_pytorch(numpy_mha, pytorch_mha)

# Compare on same input
x_np = np.random.randn(2, 10, 64).astype(np.float32)
x_torch = torch.from_numpy(x_np)

# Your output
out_np = numpy_mha(x_np)

# PyTorch output
with torch.no_grad():
    out_torch, _ = pytorch_mha(x_torch, x_torch, x_torch)

# Should match!
np.testing.assert_allclose(out_np, out_torch.numpy(), atol=1e-5)
```

## Verification

All tests pass = your implementation matches PyTorch exactly!

This is the milestone for Chapter 1:
> Your multi-head attention matches PyTorch's `nn.MultiheadAttention` output within 1e-5 tolerance.

Congratulations on completing Chapter 1!
