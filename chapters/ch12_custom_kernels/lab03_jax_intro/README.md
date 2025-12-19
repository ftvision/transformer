# Lab 03: Introduction to JAX

## Objective

Learn the fundamentals of JAX by implementing attention from scratch using JAX's functional programming model.

## What You'll Build

1. Basic array operations in JAX
2. Scaled dot-product attention in JAX
3. Multi-head attention using JAX idioms
4. Simple feedforward network

## Prerequisites

- Complete Labs 01-02
- Read `../docs/04_jax_transformations.md`
- Basic understanding of NumPy

## JAX vs NumPy

JAX provides a NumPy-like API that runs on accelerators:

```python
import jax.numpy as jnp

# Almost identical to NumPy!
x = jnp.array([1, 2, 3])
y = jnp.dot(x, x)
z = jnp.sum(x ** 2)
```

Key differences:
- Arrays are immutable (no in-place operations)
- Random numbers require explicit keys
- Functions should be pure (no side effects)

## Instructions

1. Open `src/jax_attention.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `softmax(x, axis=-1)`

Compute numerically stable softmax along an axis.

```python
# Stable softmax: subtract max before exp
x_max = jnp.max(x, axis=axis, keepdims=True)
exp_x = jnp.exp(x - x_max)
return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)
```

### `scaled_dot_product_attention(Q, K, V, mask=None)`

Single-head attention computation.

```python
# Q: (seq_q, d_k), K: (seq_k, d_k), V: (seq_k, d_v)
# scores = Q @ K.T / sqrt(d_k)
# if mask: scores = where(mask, scores, -inf)
# weights = softmax(scores)
# output = weights @ V
```

### `linear(x, weight, bias=None)`

Simple linear transformation.

```python
# output = x @ weight
# if bias: output = output + bias
```

### `multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)`

Multi-head attention with projections.

```python
# 1. Project Q, K, V: Q_proj = Q @ W_q, etc.
# 2. Reshape to (seq, num_heads, d_head)
# 3. Compute attention per head
# 4. Concatenate heads and project: output @ W_o
```

### `feedforward(x, W1, b1, W2, b2)`

Two-layer feedforward network with GELU activation.

```python
# hidden = gelu(x @ W1 + b1)
# output = hidden @ W2 + b2
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_jax_attention.py::TestSoftmax -v

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### Immutability

```python
# NumPy (mutable)
x[0] = 1  # Works

# JAX (immutable)
x = x.at[0].set(1)  # Returns new array
```

### Random Numbers

```python
import jax.random as random

# Create a key
key = random.PRNGKey(42)

# Split for multiple uses
key1, key2 = random.split(key)

# Generate random numbers
x = random.normal(key1, shape=(10, 10))
y = random.uniform(key2, shape=(5,))
```

### Einsum

JAX supports einsum like NumPy:

```python
# Matrix multiply
c = jnp.einsum('ij,jk->ik', a, b)

# Batched matrix multiply
c = jnp.einsum('bij,bjk->bik', a, b)

# Attention scores
scores = jnp.einsum('qd,kd->qk', q, k)
```

## Expected Output

Your JAX implementations should match reference implementations:

```python
# Test softmax
x = jnp.array([1.0, 2.0, 3.0])
result = softmax(x)
# Should sum to 1.0

# Test attention
Q = jnp.ones((4, 8))  # 4 queries, d_k=8
K = jnp.ones((6, 8))  # 6 keys
V = jnp.ones((6, 16)) # 6 values, d_v=16
output = scaled_dot_product_attention(Q, K, V)
# output.shape == (4, 16)
```

## GELU Activation

The GELU (Gaussian Error Linear Unit) activation is commonly used in transformers:

```python
def gelu(x):
    return x * 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))

# Or use JAX's built-in
jax.nn.gelu(x)
```

## Verification

All tests pass = you understand JAX fundamentals!

This prepares you for Lab 04: JAX Transformations.

## Bonus Challenges (Optional)

1. Implement causal masking for autoregressive attention
2. Add dropout using JAX random
3. Implement RoPE (Rotary Position Embeddings) in JAX
4. Compare performance with PyTorch implementations
