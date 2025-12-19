# Lab 04: Causal Linear Attention

## Objective

Implement causal (autoregressive) linear attention using the cumulative sum formulation.

## What You'll Build

1. Causal linear attention using cumsum (parallel form)
2. Causal linear attention using RNN state updates (recurrent form)
3. Verification that both forms produce identical results
4. Comparison with standard causal softmax attention

## Prerequisites

- Complete Lab 02 (kernel trick) and Lab 03 (feature maps)
- Read `../docs/04_causal_linear.md`

## Why Causal Attention Matters

Language models are autoregressive: they predict token i using only tokens 0..i-1.
This requires causal masking:

```
Position i can only attend to positions j ≤ i

Standard attention mask:
      K: 0  1  2  3  4
    ┌────────────────┐
Q:0 │ ✓              │
  1 │ ✓  ✓           │
  2 │ ✓  ✓  ✓        │
  3 │ ✓  ✓  ✓  ✓     │
  4 │ ✓  ✓  ✓  ✓  ✓  │
    └────────────────┘
```

## The Challenge

Our linear attention formula sums over ALL positions:
```
KV = Σ_j φ(k_j) ⊗ v_j    # All positions!
```

For causal attention, position i should only see positions ≤ i:
```
KV_i = Σ_{j≤i} φ(k_j) ⊗ v_j    # Only past positions
```

This is a **cumulative sum** (cumsum)!

## Instructions

1. Open `src/causal_linear.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `causal_linear_attention_parallel(Q, K, V, feature_map)`

Parallel form using cumulative sums.
- Compute `kv_cumsum[i] = Σ_{j≤i} φ(k_j) ⊗ v_j`
- Use `np.cumsum` for efficiency
- Good for training (parallelizable)

### `causal_linear_attention_recurrent(Q, K, V, feature_map)`

Recurrent form with explicit state updates.
- Maintain state `S` that accumulates key-value products
- Update `S` at each step: `S_i = S_{i-1} + φ(k_i) ⊗ v_i`
- Good for inference (O(1) per token)

### `causal_linear_attention_rnn_step(q, k, v, state, feature_map)`

Single step of the RNN form.
- Takes previous state, current q/k/v
- Returns output and new state
- Used for autoregressive generation

### `compare_parallel_vs_recurrent(Q, K, V, feature_map)`

Verify both forms give identical outputs.
- The two should match exactly (up to float precision)
- This validates your implementations

### `compare_to_causal_softmax(Q, K, V, feature_map)`

Compare linear attention to softmax attention with causal mask.
- Both should respect causality
- Measure output similarity

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_causal_linear.py::TestParallelForm

# Run with verbose output
uv run pytest tests/ -v
```

## The Two Forms

### Parallel Form (Training)

```python
# Compute outer products
kv = einsum('nd,nv->ndv', K_prime, V)    # (n, d_φ, d_v)

# Cumulative sum along sequence
kv_cumsum = cumsum(kv, axis=0)            # (n, d_φ, d_v)
z_cumsum = cumsum(K_prime, axis=0)        # (n, d_φ)

# Query each cumulative state
output = einsum('nd,ndv->nv', Q_prime, kv_cumsum)
output = output / (einsum('nd,nd->n', Q_prime, z_cumsum) + eps)
```

### Recurrent Form (Inference)

```python
S = zeros(d_φ, d_v)  # State
Z = zeros(d_φ)       # Normalizer

for i in range(n):
    # Update state
    S = S + outer(K_prime[i], V[i])
    Z = Z + K_prime[i]

    # Query state
    output[i] = (Q_prime[i] @ S) / (Q_prime[i] @ Z + eps)
```

## Why Both Forms?

| Aspect | Parallel | Recurrent |
|--------|----------|-----------|
| Use case | Training | Inference |
| Per-token cost | O(d²) amortized | O(d²) |
| Parallelization | Full | None |
| Memory | O(n·d²) | O(d²) |
| KV cache | Not needed | State IS the cache |

The recurrent form is the **key advantage** of linear attention:
- Standard attention: O(n) per new token (recompute with full KV cache)
- Linear attention: O(1) per new token (just update state)

## Expected Behavior

```python
Q = np.random.randn(100, 64)
K = np.random.randn(100, 64)
V = np.random.randn(100, 64)

# Both should give same output
out_parallel = causal_linear_attention_parallel(Q, K, V, elu_plus_one)
out_recurrent = causal_linear_attention_recurrent(Q, K, V, elu_plus_one)

np.testing.assert_allclose(out_parallel, out_recurrent, rtol=1e-5)
print("✓ Parallel and recurrent forms match!")
```

## Hints

- For cumsum, use `np.cumsum(x, axis=0)` along the sequence dimension
- For outer product, `np.outer(a, b)` or `a[:, None] * b[None, :]`
- The state shape is `(d_phi, d_v)` - think of it as a compressed KV cache
- Don't forget the normalizer (sum of keys)!

## Verification

All tests pass = you've implemented causal linear attention!

**Chapter 5 Milestone**: Your linear attention is 10x faster than standard attention for seq_len=4096.
