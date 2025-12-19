# Lab 01: RNN View of Linear Attention

## Objective

Implement linear attention in its recurrent form, revealing its connection to RNNs.

## What You'll Build

Functions that compute linear attention in two equivalent ways:
1. **Parallel form**: Using cumulative sums (for training)
2. **Recurrent form**: Step-by-step with hidden state (for inference)

## Prerequisites

Read these docs first:
- `../docs/01_linear_attention_as_rnn.md`
- Chapter 5's linear attention materials

## Why This Matters

Understanding the RNN view of linear attention is crucial because:
- It enables O(1) per-token inference (constant time regardless of sequence length)
- It reveals the hidden state S = K^T V as a "compressed memory"
- It's the foundation for efficient inference in GLA, Mamba, and other models

## Instructions

1. Open `src/rnn_view.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `feature_map(x, method='elu_plus_one')`

Apply a feature map φ(x) to the input. Options:
- `'elu_plus_one'`: ELU(x) + 1 (ensures positivity)
- `'relu'`: ReLU(x)
- `'identity'`: No transformation

### `linear_attention_parallel(Q, K, V, feature_map_fn)`

Compute linear attention using the parallel (cumulative sum) formulation:
```
output_i = φ(q_i) @ S_i
where S_i = cumsum(φ(K)^T @ V)
```

### `linear_attention_recurrent(Q, K, V, feature_map_fn)`

Compute linear attention using the recurrent formulation:
```
For each position i:
    S_i = S_{i-1} + φ(k_i)^T @ v_i
    output_i = φ(q_i) @ S_i
```

### `linear_attention_step(q, k, v, state, feature_map_fn)`

Single-step recurrent computation for inference:
- Takes single token's q, k, v and current state
- Returns output and updated state

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_rnn_view.py::TestFeatureMap

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use `np.einsum` for efficient outer products: `np.einsum('...d,...v->...dv', k, v)`
- For cumulative sum: `np.cumsum(tensor, axis=seq_dim)`
- The state S has shape `(d_k, d_v)` - it's a matrix, not a vector!
- Parallel and recurrent forms should give **identical outputs**

## Expected Shapes

```
Q, K: (batch, seq_len, d_k) or (seq_len, d_k)
V: (batch, seq_len, d_v) or (seq_len, d_v)

State S: (batch, d_k, d_v) or (d_k, d_v)
Output: same shape as V
```

## Verification

All tests pass = you understand linear attention's dual nature!

Key insight: The recurrent form shows that linear attention maintains a **fixed-size state** regardless of sequence length.
