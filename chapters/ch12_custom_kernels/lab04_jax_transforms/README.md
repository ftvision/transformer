# Lab 04: JAX Transformations

## Objective

Master JAX's function transformations (`jit`, `vmap`, `grad`) to optimize and parallelize your code.

## What You'll Build

1. JIT-compiled functions for performance
2. Vectorized batch processing with vmap
3. Automatic differentiation with grad
4. Composed transformations

## Prerequisites

- Complete Lab 03 (JAX Introduction)
- Read `../docs/04_jax_transformations.md`
- Understanding of basic calculus (for gradients)

## Key Transformations

### jit: Just-In-Time Compilation

```python
import jax

@jax.jit
def fast_fn(x):
    return x @ x.T + x
```

- Compiles function with XLA
- First call is slow (compilation), subsequent calls are fast
- Functions must be pure (no side effects)

### vmap: Automatic Vectorization

```python
def single_example(x):
    return jnp.sum(x ** 2)

# Automatically batches!
batch_fn = jax.vmap(single_example)
```

- Transforms single-example functions to batch functions
- No manual loop needed
- Efficient parallel execution

### grad: Automatic Differentiation

```python
def loss(x):
    return jnp.sum(x ** 2)

grad_fn = jax.grad(loss)
gradient = grad_fn(jnp.array([1.0, 2.0, 3.0]))
```

- Computes gradients automatically
- Supports higher-order derivatives
- Works with complex nested structures

## Instructions

1. Open `src/transforms.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `jit_attention(Q, K, V)`

JIT-compiled attention function.

```python
@jax.jit
def jit_attention(Q, K, V):
    # Your attention implementation
    pass
```

### `batched_attention(Q, K, V)`

Vectorized attention for batched inputs using vmap.

```python
# Q: (batch, seq_q, d_k) -> output: (batch, seq_q, d_v)
batched_attention = jax.vmap(single_attention)
```

### `attention_gradient(Q, K, V, dL_dout)`

Compute gradients of attention output with respect to Q, K, V.

```python
# Use jax.vjp (vector-Jacobian product) for backprop
```

### `train_step(params, x, y)`

Single training step with gradient computation.

```python
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Update params with gradients
    return new_params, loss
```

### `multi_head_batched(Q, K, V, num_heads)`

Batched multi-head attention using nested vmap.

```python
# Layer vmap for batch dimension
# Layer vmap for head dimension
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_transforms.py::TestJit -v

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### JIT Caching

```python
# First call compiles (slow)
result = jit_fn(x)

# Subsequent calls use cache (fast)
result = jit_fn(x)  # Fast!
result = jit_fn(y)  # Fast if same shape/dtype
```

### vmap Axis Specification

```python
# Default: batch over axis 0
jax.vmap(fn)

# Batch over axis 1
jax.vmap(fn, in_axes=1)

# Different axes for different inputs
jax.vmap(fn, in_axes=(0, None))  # First batched, second not

# Output axis
jax.vmap(fn, out_axes=1)  # Output batch dim is axis 1
```

### Gradient Computation

```python
# Gradient w.r.t. first argument
grad_fn = jax.grad(fn)

# Gradient w.r.t. specific argument
grad_fn = jax.grad(fn, argnums=1)

# Value and gradient together
loss, grads = jax.value_and_grad(fn)(x)

# Gradient w.r.t. multiple arguments
grads = jax.grad(fn, argnums=(0, 1))(x, y)
```

### Composing Transformations

```python
# Order matters! Inner transforms applied first
@jax.jit
@jax.vmap
def fn(x):
    return x ** 2

# Equivalent to:
fn = jax.jit(jax.vmap(lambda x: x ** 2))
```

## Performance Comparison

Your implementations should show significant speedups:

```python
import time

# Without JIT
start = time.time()
for _ in range(100):
    result = slow_fn(x)
print(f"Without JIT: {time.time() - start:.3f}s")

# With JIT (after warmup)
jit_fn = jax.jit(slow_fn)
_ = jit_fn(x)  # Warmup

start = time.time()
for _ in range(100):
    result = jit_fn(x)
result.block_until_ready()  # Wait for GPU!
print(f"With JIT: {time.time() - start:.3f}s")
```

## Expected Output

Your transformations should:

1. **JIT**: Match non-JIT output exactly
2. **vmap**: Process batches correctly
3. **grad**: Compute correct gradients

```python
# Test JIT correctness
result_jit = jit_attention(Q, K, V)
result_ref = attention_reference(Q, K, V)
assert jnp.allclose(result_jit, result_ref)

# Test vmap
Qs = jnp.stack([Q] * 8)  # batch of 8
result = batched_attention(Qs, Ks, Vs)
assert result.shape == (8, seq_q, d_v)

# Test grad
dL_dQ, dL_dK, dL_dV = attention_gradient(Q, K, V, dL_dout)
# Gradients should have same shapes as inputs
```

## Verification

All tests pass = you understand JAX transformations!

This is the final JAX lab. You now have the skills to write efficient JAX code.

## Bonus Challenges (Optional)

1. Implement gradient checkpointing using `jax.checkpoint`
2. Use `jax.pmap` for multi-device training
3. Implement a custom VJP rule with `jax.custom_vjp`
4. Profile and compare performance with PyTorch
