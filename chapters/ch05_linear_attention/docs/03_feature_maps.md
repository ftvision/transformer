# Feature Maps: Different φ Functions and Their Properties

![Feature Maps](vis/feature_maps.svg)

## Why Feature Maps Matter

The feature map φ is the heart of linear attention. It determines:
- **Approximation quality**: How well does `φ(q)^T φ(k)` approximate `exp(q·k)`?
- **Computational cost**: How large is d_φ (feature dimension)?
- **Training stability**: Are gradients well-behaved?
- **Expressiveness**: What patterns can the attention learn?

## The Ideal: Exact Softmax Kernel

In theory, we could approximate softmax exactly using Random Fourier Features:

```
exp(q · k) ≈ φ(q)^T φ(k)

where φ(x) = exp(-||x||²/2) × [cos(w_1·x), sin(w_1·x), ..., cos(w_m·x), sin(w_m·x)]
```

**Problem**: Requires very high d_φ (thousands of features) to get good approximation.

In practice, we use simpler feature maps that don't exactly match softmax but work well enough.

## Feature Map 1: ELU + 1

**Formula**:
```python
def elu_feature_map(x):
    return F.elu(x) + 1
```

**Properties**:
- Simple and fast
- Always positive (required for attention weights to be positive)
- ELU(x) + 1 = x + 1 for x ≥ 0, exp(x) for x < 0

**Visualization**:
```
         │    ELU(x) + 1
         │         ╱
     2.0 ┤        ╱
         │       ╱
     1.0 ┤──────╱
         │     ╱
     0.0 ┼────┼────────
        -2   -1   0   1   2
```

**Code**:
```python
def linear_attention_elu(Q, K, V):
    Q_prime = F.elu(Q) + 1
    K_prime = F.elu(K) + 1

    KV = torch.einsum('bnd,bnv->bdv', K_prime, V)
    Z = K_prime.sum(dim=1)

    numerator = torch.einsum('bnd,bdv->bnv', Q_prime, KV)
    denominator = torch.einsum('bnd,bd->bn', Q_prime, Z)

    return numerator / (denominator.unsqueeze(-1) + 1e-6)
```

**Trade-offs**:
- ✅ Fast, simple
- ✅ No additional parameters
- ❌ Poor approximation of softmax
- ❌ Can have numerical issues (values close to 0)

## Feature Map 2: ReLU

**Formula**:
```python
def relu_feature_map(x):
    return F.relu(x)
```

**Properties**:
- Even simpler than ELU
- Sparse (many zeros)
- Doesn't guarantee positivity after dot product

**Visualization**:
```
         │    ReLU(x)
         │         ╱
     2.0 ┤        ╱
         │       ╱
     1.0 ┤      ╱
         │     ╱
     0.0 ┼────┼────────
        -2   -1   0   1   2
```

**Trade-offs**:
- ✅ Extremely fast
- ✅ Sparsity can help with computation
- ❌ Non-positive values possible
- ❌ "Dead" features (always 0)

## Feature Map 3: Softmax Kernel (Performers)

**Formula** (from the Performers paper):
```python
def softmax_kernel_feature_map(x, projection_matrix):
    """
    Approximates exp(q·k) using random features.
    projection_matrix: (d, m) random matrix
    """
    # Project to m-dimensional space
    projected = x @ projection_matrix  # (n, m)

    # Compute normalization factor
    norm = torch.sum(x ** 2, dim=-1, keepdim=True) / 2

    # Random features
    features = torch.exp(-norm) * torch.cat([
        torch.cos(projected),
        torch.sin(projected)
    ], dim=-1) / math.sqrt(m)

    return features
```

**Properties**:
- Theoretically approximates softmax attention
- Unbiased estimator of exp(q·k)
- Feature dimension: 2m (cosine + sine terms)

**Trade-offs**:
- ✅ Closest to true softmax behavior
- ✅ Theoretical guarantees
- ❌ Requires random projection matrix
- ❌ Higher computational overhead
- ❌ Variance can be high

## Feature Map 4: 1 + elu (Linear Transformer)

The original "Linear Transformer" paper used:

```python
def linear_transformer_feature_map(x):
    return 1 + F.elu(x)  # Same as ELU + 1
```

This ensures φ(x) > 0, which keeps attention weights positive.

## Feature Map 5: Squared ReLU

**Formula**:
```python
def squared_relu_feature_map(x):
    return F.relu(x) ** 2
```

**Properties**:
- Positive (squared values)
- Smoother gradients than ReLU
- Used in some recent architectures

## Feature Map 6: Exponential (exp)

**Formula**:
```python
def exp_feature_map(x):
    return torch.exp(x)
```

**Warning**: This can cause numerical overflow! Use with care.

```python
def exp_feature_map_stable(x, max_val=10):
    return torch.exp(torch.clamp(x, max=max_val))
```

**Trade-offs**:
- ✅ Closest to softmax in spirit
- ❌ Numerical instability (exp grows fast)
- ❌ Requires careful scaling

## Comparison: Quality vs Speed

| Feature Map | Softmax Approx. | Speed | Stability | Used In |
|-------------|----------------|-------|-----------|---------|
| ELU + 1 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Linear Transformer |
| ReLU | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Experiments |
| Random Features | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Performers |
| Squared ReLU | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Recent work |
| Exp | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | With caution |

## The Positivity Requirement

For attention to make sense, weights should be non-negative:

```
Attention weights = φ(q)^T φ(k) / normalizer ≥ 0
```

Feature maps that guarantee positivity:
- ELU + 1 (φ(x) > 0 for all x)
- ReLU (φ(x) ≥ 0)
- Exp (φ(x) > 0)
- Squared ReLU (φ(x) ≥ 0)
- Random Fourier Features (construction ensures positivity)

## Practical Recommendations

**For getting started** (Lab 03):
```python
# Simple and works reasonably well
def feature_map(x):
    return F.elu(x) + 1
```

**For better quality**:
```python
# Performers-style random features
def feature_map(x, omega):
    proj = x @ omega
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(omega.shape[1])
```

**For maximum speed**:
```python
# Just ReLU, but check your use case
def feature_map(x):
    return F.relu(x)
```

## The Bigger Picture

Feature maps alone don't make linear attention competitive with softmax attention. Modern approaches (Chapter 6) add:
- **Gating mechanisms** (GLA, DeltaNet)
- **Data-dependent decay** (forgetting old information)
- **Chunkwise parallelism** (efficient training)

These innovations address the expressiveness gap that simple feature maps leave.

## What's Next

So far we've discussed non-causal attention. For language models, we need **causal** attention where position i can only attend to positions ≤ i. This requires a different formulation—see `04_causal_linear.md`.
