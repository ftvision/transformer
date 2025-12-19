# Lab 02: Gradient Visualization

## Objective

Build tools to visualize and analyze gradient flow through transformer layers.

## What You'll Build

Functions to:
1. Compute gradient statistics across layers
2. Detect vanishing and exploding gradients
3. Visualize gradient flow through the network
4. Analyze gradient norms over training

## Prerequisites

- Complete Lab 01 (loss functions)
- Read `../docs/02_gradient_flow.md`

## Why Gradient Visualization Matters

Understanding gradient flow is crucial for:
- Debugging training failures (loss doesn't decrease)
- Diagnosing instability (loss explodes to NaN)
- Tuning hyperparameters (learning rate, warmup)
- Understanding model behavior

## Instructions

1. Open `src/gradients.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `compute_gradient_norm(gradients)`
Compute the L2 norm of gradients.
- Can be per-parameter or global
- Useful for gradient clipping decisions

### `compute_gradient_stats(gradients)`
Compute statistics about gradients:
- Mean, std, min, max
- Percentage of near-zero gradients
- Percentage of very large gradients

### `detect_gradient_issues(gradient_history)`
Analyze gradient history to detect problems:
- Vanishing gradients (norms decreasing to ~0)
- Exploding gradients (norms increasing rapidly)
- Stable training (norms relatively constant)

### `clip_gradients(gradients, max_norm)`
Implement gradient clipping by global norm.
- If total norm > max_norm, scale all gradients proportionally
- Returns clipped gradients and original norm

### `visualize_layer_gradients(layer_gradients)`
Create data structure for visualizing gradients across layers.
- Shows gradient flow from output to input layers
- Helps identify where gradients vanish or explode

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test class
uv run pytest tests/test_gradients.py::TestGradientNorm
```

## Example Usage

```python
import numpy as np
from gradients import (
    compute_gradient_norm,
    compute_gradient_stats,
    detect_gradient_issues,
    clip_gradients
)

# Simulated gradients from a 6-layer network
layer_gradients = {
    'layer_0': np.random.randn(100, 100) * 0.01,
    'layer_1': np.random.randn(100, 100) * 0.01,
    'layer_2': np.random.randn(100, 100) * 0.001,  # Getting smaller
    'layer_3': np.random.randn(100, 100) * 0.0001,
    'layer_4': np.random.randn(100, 100) * 0.00001,
    'layer_5': np.random.randn(100, 100) * 0.000001,  # Very small!
}

# Analyze each layer
for name, grad in layer_gradients.items():
    stats = compute_gradient_stats(grad)
    print(f"{name}: norm={stats['norm']:.6f}, mean={stats['mean']:.6f}")

# Detect issues
all_grads = list(layer_gradients.values())
issues = detect_gradient_issues(all_grads)
print(f"Detected issues: {issues['status']}")
# Output: "Detected issues: vanishing"

# Apply gradient clipping
clipped, original_norm = clip_gradients(all_grads, max_norm=1.0)
print(f"Original norm: {original_norm:.4f}, clipping applied: {original_norm > 1.0}")
```

## Gradient Flow in Transformers

A healthy transformer should show:
```
Layer 6 (output):   |████████████| norm: 0.10
Layer 5:            |████████████| norm: 0.09
Layer 4:            |████████████| norm: 0.11
Layer 3:            |████████████| norm: 0.10
Layer 2:            |████████████| norm: 0.10
Layer 1:            |████████████| norm: 0.09
Layer 0 (input):    |████████████| norm: 0.10
                    Stable gradient flow!
```

Vanishing gradients look like:
```
Layer 6 (output):   |████████████| norm: 0.10
Layer 5:            |████████    | norm: 0.05
Layer 4:            |████        | norm: 0.01
Layer 3:            |██          | norm: 0.001
Layer 2:            |            | norm: 0.0001
Layer 1:            |            | norm: 0.00001
Layer 0 (input):    |            | norm: 0.000001
                    Vanishing gradients!
```

## Hints

- Global norm is `sqrt(sum of squared norms)` across all parameters
- For clipping, scale factor is `max_norm / global_norm` (only if global_norm > max_norm)
- Near-zero threshold: ~1e-7, very large threshold: ~1e3
- Gradient history should be a list of gradient norms over time

## Verification

All tests pass = you've built the gradient analysis toolkit!

These tools will be essential when debugging your training loop in Lab 03.
