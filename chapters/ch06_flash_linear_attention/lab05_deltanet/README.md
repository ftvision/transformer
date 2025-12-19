# Lab 05: DeltaNet

## Objective

Implement DeltaNet, which uses the delta rule for implicit forgetting in linear attention.

## What You'll Build

A `DeltaNet` module that:
1. Uses error-correction updates instead of explicit gating
2. Implements associative memory with overwriting
3. Achieves implicit forgetting through the delta rule

## Prerequisites

- Complete Labs 01-04
- Read `../docs/04_gated_linear_attention.md` (DeltaNet section)

## Why DeltaNet?

GLA uses **explicit** gating: `new_state = g * old_state + (1-g) * new_value`

DeltaNet uses **implicit** forgetting via the delta rule:
```
retrieved = φ(k)^T @ S        # Query current memory
error = v - retrieved          # Compute error
S_new = S + φ(k)^T @ error    # Correct the error
```

This is equivalent to:
```
S_new = S + φ(k)^T @ v - φ(k)^T @ (φ(k)^T @ S)
      = S + φ(k)^T @ v - (φ(k)^T @ φ(k)) @ S
```

The second term acts as implicit forgetting: it removes the old value at key k before writing the new one.

## Instructions

1. Open `src/deltanet.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `delta_rule_update(state, k, v, feature_map_fn, beta=1.0)`

Apply the delta rule update:
```
retrieved = state @ φ(k)
error = v - beta * retrieved
new_state = state + φ(k)^T @ error
```
Beta controls the "forgetting strength" (1.0 = full delta rule).

### `deltanet_step(q, k, v, state, feature_map_fn, beta=1.0)`

Single-step DeltaNet for inference.

### `deltanet_recurrent(Q, K, V, feature_map_fn, beta=1.0)`

Full sequence DeltaNet using recurrent computation.

### `DeltaNet` class

Full DeltaNet module with:
- Linear projections for Q, K, V
- Optional beta parameter (learnable or fixed)
- Both step and sequence modes

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_deltanet.py::TestDeltaRuleUpdate

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- The delta rule is: `S += k^T @ (v - S^T @ k) = k^T @ v - (k^T @ k) @ S`
- This removes old value at key k, then writes new value
- Beta=0 reduces to vanilla linear attention (no forgetting)
- Beta=1 is full delta rule (complete overwriting)
- Think of S as an associative memory where keys map to values

## The Delta Rule Connection

From Hopfield networks, the delta rule learns associations:
- **Store**: When you see (key, value), update memory to associate them
- **Retrieve**: Given a key, return the associated value
- **Overwrite**: If key already exists, replace the old value

DeltaNet applies this to attention:
```
Memory S stores key→value associations
Query with key k returns S^T @ φ(k)
Update with (k, v) overwrites the association
```

## Expected Shapes

```
Q, K: (batch, seq_len, d_k) or (seq_len, d_k)
V: (batch, seq_len, d_v) or (seq_len, d_v)
State S: (batch, d_k, d_v) or (d_k, d_v)
Output: same shape as V
```

## DeltaNet vs GLA

| Aspect | GLA | DeltaNet |
|--------|-----|----------|
| Forgetting | Explicit gate g | Implicit delta rule |
| Update | g*S + (1-g)*kv | S + k^T(v - S^Tk) |
| Parameters | Extra gate projection | Beta hyperparameter |
| Intuition | Interpolation | Error correction |
| Memory model | Leaky integrator | Associative memory |

## Verification

All tests pass = you understand the delta rule for linear attention!

Milestone: Working DeltaNet matches expected behavior from the Kimi approach.
