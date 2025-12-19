# Lab 03: Feature Maps

## Objective

Implement and compare different feature maps for linear attention, understanding their trade-offs.

## What You'll Build

A collection of feature maps and tools to:
1. Implement various φ functions (ELU+1, ReLU, exp, random features)
2. Compare how well they approximate softmax attention
3. Analyze quality vs speed trade-offs
4. Visualize implicit attention patterns

## Prerequisites

- Complete Lab 02 (kernel trick)
- Read `../docs/03_feature_maps.md`

## Why Feature Maps Matter

The feature map φ is the heart of linear attention. It determines:
- **Approximation quality**: How close to softmax attention?
- **Computational cost**: Feature dimension d_φ
- **Training stability**: Gradient behavior
- **Expressiveness**: What patterns can be learned?

## Instructions

1. Open `src/feature_maps.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### Feature Maps

#### `elu_plus_one(x)`
The ELU+1 feature map from the Linear Transformer paper.
- `φ(x) = ELU(x) + 1`
- Always positive, simple and fast

#### `relu_feature_map(x)`
Simple ReLU feature map.
- `φ(x) = max(0, x)`
- Fast but can have zero attention weights

#### `squared_relu_feature_map(x)`
Squared ReLU for smoother gradients.
- `φ(x) = max(0, x)²`
- Positive outputs, smoother than ReLU

#### `softmax_kernel_feature_map(x, projection_matrix)`
Random Fourier Features to approximate softmax.
- Based on the Performers paper
- Better softmax approximation but more expensive

### Analysis Functions

#### `compute_implicit_attention(Q, K, feature_map)`
Compute the implicit attention matrix from linear attention.
- `A[i,j] = φ(q_i)·φ(k_j) / Σ_l φ(q_i)·φ(k_l)`
- Useful for visualization and comparison

#### `compare_to_softmax(Q, K, feature_map)`
Compare linear attention weights to softmax attention.
- Returns MSE, max difference, correlation

#### `analyze_feature_map_quality(feature_map, Q, K, V)`
Comprehensive analysis of a feature map.
- Output similarity to softmax
- Attention pattern correlation
- Numerical stability metrics

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_feature_maps.py::TestELUPlusOne

# Run with verbose output
uv run pytest tests/ -v
```

## Feature Map Comparison

| Feature Map | Positivity | Softmax Approx | Speed | Stability |
|-------------|------------|----------------|-------|-----------|
| ELU+1 | ✓ Always | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| ReLU | Partial | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Squared ReLU | ✓ Always | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Random Features | ✓ Always | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## Expected Behavior

```python
from feature_maps import *

Q = np.random.randn(100, 64).astype(np.float32)
K = np.random.randn(100, 64).astype(np.float32)
V = np.random.randn(100, 64).astype(np.float32)

# Compare different feature maps
for fm_name, fm in [('ELU+1', elu_plus_one),
                     ('ReLU', relu_feature_map),
                     ('Squared ReLU', squared_relu_feature_map)]:
    metrics = compare_to_softmax(Q, K, fm)
    print(f"{fm_name}: MSE={metrics['mse']:.4f}, corr={metrics['correlation']:.3f}")

# Output:
# ELU+1: MSE=0.0823, corr=0.712
# ReLU: MSE=0.1245, corr=0.534
# Squared ReLU: MSE=0.0912, corr=0.689
```

## The Math: Why Different Feature Maps?

Softmax attention: `A[i,j] = exp(q_i·k_j) / Σ_l exp(q_i·k_l)`

We want: `A[i,j] ≈ φ(q_i)·φ(k_j) / Σ_l φ(q_i)·φ(k_l)`

Different φ functions trade off:
- **Exact approximation** (random features): More computation
- **Efficiency** (ELU+1, ReLU): Less accurate but faster
- **Stability** (squared): Smoother gradients

## Hints

- ELU(x) = x if x > 0, else exp(x) - 1
- For random features, use cos/sin of random projections
- Use `np.corrcoef` for correlation
- Normalize attention matrices before comparing

## Verification

All tests pass = you understand feature map trade-offs!

This knowledge is essential for choosing the right linear attention variant.
