# Lab 01: Layer Normalization

## Objective

Implement Layer Normalization and RMSNorm from scratch using NumPy.

## What You'll Build

Functions that compute:
```
LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
```

## Prerequisites

Read these docs first:
- `../docs/01_residuals_and_normalization.md`

## Instructions

1. Open `src/layer_norm.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `layer_norm(x, gamma, beta, eps=1e-5)`
Compute layer normalization along the last dimension.
1. Compute mean along last axis
2. Compute variance along last axis
3. Normalize: `(x - mean) / sqrt(var + eps)`
4. Scale and shift: `gamma * normalized + beta`

### `rms_norm(x, gamma, eps=1e-5)`
Compute RMS (Root Mean Square) normalization.
1. Compute RMS: `sqrt(mean(x^2))`
2. Normalize: `x / (rms + eps)`
3. Scale: `gamma * normalized`

### `class LayerNorm`
A class that stores learnable parameters gamma and beta.

### `class RMSNorm`
A class that stores learnable parameter gamma.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_layer_norm.py::TestLayerNormFunction

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use `np.mean()` and `np.var()` with `axis=-1` and `keepdims=True`
- For variance, use `unbiased=False` (or don't specify, as NumPy defaults to population variance)
- `eps` prevents division by zero - it's added inside the sqrt
- gamma and beta have shape `(d_model,)`, same as the last dimension of x

## Expected Shapes

```
x: (..., d_model)  - any number of leading dimensions
gamma: (d_model,)
beta: (d_model,)

output: (..., d_model) - same shape as input
```

## Key Insight

Layer norm normalizes each token independently:
- Input: (batch=2, seq_len=4, d_model=8)
- For each of the 8 tokens, compute mean and variance over the 8 features
- This is different from batch norm, which normalizes over the batch

## Numerical Stability

The `eps` parameter is crucial:
- Without it: `sqrt(0)` can cause NaN
- With it: `sqrt(0 + 1e-5)` is safe

Always add eps inside the sqrt, not outside!

## Verification

All tests pass = you've correctly implemented layer normalization!

Next up: Lab 02 where you'll implement positional encodings.
