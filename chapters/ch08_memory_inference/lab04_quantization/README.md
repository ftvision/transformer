# Lab 04: Quantization Basics

## Objective

Implement basic quantization (INT8) from scratch to understand how precision reduction speeds up inference.

## What You'll Build

Functions for:
- Symmetric and asymmetric quantization
- Per-tensor and per-channel quantization
- Quantized matrix multiplication
- Quantization error analysis

## Prerequisites

Read these docs first:
- `../docs/04_quantization_basics.md`
- Completed Labs 01-03

## Instructions

1. Open `src/quantization.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `quantize_symmetric(x, num_bits=8)`
Symmetric quantization: map floats to integers with zero at zero.
- Find the maximum absolute value: `scale = max(|x|) / (2^(bits-1) - 1)`
- Quantize: `q = round(x / scale)`
- Clamp to valid range: `[-2^(bits-1), 2^(bits-1) - 1]`

Returns: `(quantized_tensor, scale)`

### `dequantize_symmetric(q, scale)`
Reverse symmetric quantization.
- `x = q * scale`

Returns: Dequantized float tensor

### `quantize_asymmetric(x, num_bits=8)`
Asymmetric quantization: uses zero_point for asymmetric distributions.
- `scale = (max(x) - min(x)) / (2^bits - 1)`
- `zero_point = round(-min(x) / scale)`
- `q = round(x / scale + zero_point)`

Returns: `(quantized_tensor, scale, zero_point)`

### `dequantize_asymmetric(q, scale, zero_point)`
Reverse asymmetric quantization.
- `x = (q - zero_point) * scale`

Returns: Dequantized float tensor

### `quantize_per_channel(x, axis, num_bits=8)`
Per-channel quantization: separate scale per output channel.
- Compute scale for each slice along the specified axis
- Better accuracy than per-tensor for weights with varying ranges

Returns: `(quantized_tensor, scales)` where scales has shape along axis

### `quantization_error(original, quantized, dequantized)`
Calculate quantization error metrics.

Returns dict with:
- `mse`: Mean squared error
- `mae`: Mean absolute error
- `max_error`: Maximum absolute error
- `relative_error`: MSE / variance(original)

### `quantized_matmul(a_q, a_scale, b_q, b_scale, a_zp=0, b_zp=0)`
Perform matrix multiplication on quantized tensors.
- Compute integer matmul: `c_int = a_q @ b_q`
- Apply scaling: `c = c_int * (a_scale * b_scale)`
- Handle zero points if provided

Returns: Float result of the quantized computation

### `compare_quantization_methods(x, num_bits=8)`
Compare symmetric vs asymmetric, per-tensor vs per-channel.

Returns dict with error metrics for each method.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_quantization.py::TestSymmetricQuantization

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
import numpy as np
from quantization import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantized_matmul,
    quantization_error
)

# Create some weights
weights = np.random.randn(256, 256).astype(np.float32)

# Symmetric quantization
w_q, scale = quantize_symmetric(weights, num_bits=8)
w_deq = dequantize_symmetric(w_q, scale)

# Check error
error = quantization_error(weights, w_q, w_deq)
print(f"MSE: {error['mse']:.6f}")
print(f"Max error: {error['max_error']:.6f}")

# Asymmetric quantization (better for ReLU activations)
activations = np.abs(np.random.randn(32, 256)).astype(np.float32)  # All positive
a_q, a_scale, a_zp = quantize_asymmetric(activations, num_bits=8)

# Quantized matrix multiplication
# output = activations @ weights.T
a_q, a_scale, a_zp = quantize_asymmetric(activations)
w_q, w_scale = quantize_symmetric(weights)
output = quantized_matmul(a_q, a_scale, w_q.T, w_scale, a_zp=a_zp)

# Compare with float
output_float = activations @ weights.T
print(f"Matmul error: {np.mean((output - output_float)**2):.6f}")
```

## Hints

- INT8 range: [-128, 127] for signed, [0, 255] for unsigned
- Symmetric uses signed integers centered at 0
- Asymmetric can use the full unsigned range for positive-only values
- Clamping is essential to prevent overflow
- Per-channel quantization is typically along the output dimension of weight matrices

## Key Formulas

**Symmetric Quantization:**
```
scale = max(|x|) / 127
q = clamp(round(x / scale), -128, 127)
x' = q * scale
```

**Asymmetric Quantization:**
```
scale = (max(x) - min(x)) / 255
zero_point = clamp(round(-min(x) / scale), 0, 255)
q = clamp(round(x / scale + zero_point), 0, 255)
x' = (q - zero_point) * scale
```

**Per-Channel (for weights with shape [out, in]):**
```
for each output channel i:
    scale[i] = max(|W[i, :]|) / 127
    W_q[i, :] = round(W[i, :] / scale[i])
```

## Verification

All tests pass = you understand quantization basics!

Key insight: INT8 quantization typically achieves <0.1% relative error on typical weight distributions, enabling 2x memory reduction with minimal accuracy loss.
