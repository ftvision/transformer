# Lab 04: Gated Linear Attention (GLA)

## Objective

Implement Gated Linear Attention, which adds data-dependent forgetting to linear attention.

## What You'll Build

A `GatedLinearAttention` module that:
1. Computes data-dependent gates from input
2. Uses gates to control state retention
3. Enables selective forgetting of old information

## Prerequisites

- Complete Labs 01-03 (linear attention basics)
- Read `../docs/04_gated_linear_attention.md`

## Why Gating Matters

Vanilla linear attention: `S_t = S_{t-1} + φ(k_t)^T v_t`
- No forgetting: state accumulates forever
- State magnitude can grow unboundedly
- Old information never gets removed

Gated linear attention: `S_t = g_t ⊙ S_{t-1} + (1-g_t) ⊙ φ(k_t)^T v_t`
- Data-dependent gates control retention
- Can selectively forget old information
- Bounded state dynamics

## Instructions

1. Open `src/gated_attention.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `compute_gate(x, W_g, b_g)`

Compute the data-dependent gate:
```
g = sigmoid(x @ W_g + b_g)
```
Output is in (0, 1) for each dimension.

### `gated_state_update(state, k, v, gate, feature_map_fn)`

Update state with gating:
```
new_state = gate * state + (1 - gate) * (φ(k)^T @ v)
```

### `GatedLinearAttention` class

Full GLA implementation with:
- Linear projections for Q, K, V, G
- Gated state updates
- Both recurrent and parallel computation

### `gla_step(q, k, v, gate, state, feature_map_fn)`

Single-step GLA for inference.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_gated_attention.py::TestComputeGate

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Gate should be element-wise: `gate.shape == (d_k, d_v)` or broadcast-compatible
- Sigmoid ensures gate values are in (0, 1)
- High gate value = keep old state; Low gate value = write new value
- Initialize gate bias positive so model starts with high retention

## GLA Architecture

```
Input x ──┬──────────────┬──────────────┬──────────────┐
          ↓              ↓              ↓              ↓
       W_q(x)         W_k(x)         W_v(x)       W_g(x)
          ↓              ↓              ↓              ↓
         q_t           k_t           v_t         g_t=σ(...)
                        │              │              │
                        └──────┬───────┘              │
                               ↓                      │
                         k_t^T @ v_t                  │
                               │                      │
      S_{t-1} ────────> [g_t ⊙ S + (1-g_t) ⊙ kv] ───> S_t
                               │
                               ↓
          q_t ─────────> [q_t @ S_t] ───> o_t
```

## Expected Shapes

```
Input x: (batch, seq_len, d_model)
q, k: (batch, seq_len, d_k)
v: (batch, seq_len, d_v)
gate: (batch, seq_len, d_k) or broadcastable
state S: (batch, d_k, d_v)
output: (batch, seq_len, d_v)
```

## Verification

All tests pass = you've implemented data-dependent gating!

Key insight: Gating allows the model to learn WHAT to remember and forget.
