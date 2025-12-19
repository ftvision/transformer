# Lab 02: Online Softmax

## Objective

Implement online (incremental) softmax that processes data in blocks without storing all values.

## What You'll Build

Functions to:
1. Implement numerically stable softmax
2. Implement one-pass online softmax with running statistics
3. Implement online softmax with output accumulation (for attention)

## Prerequisites

- Complete Lab 01 (memory profiling)
- Read `../docs/03_online_softmax.md`

## Why Online Softmax Matters

Standard softmax needs to see all values at once:
```python
# Standard: need ALL values
softmax(x) = exp(x) / sum(exp(x))  # sum over everything!
```

Online softmax computes the same result incrementally:
- Process blocks one at a time
- Track running statistics (max, sum)
- Rescale when new max is found

This is the key insight that enables Flash Attention!

## Instructions

1. Open `src/online_softmax.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `safe_softmax(x, axis=-1)`
Standard softmax with numerical stability (subtract max).
- `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

### `online_softmax_stats(x_blocks)`
Compute softmax statistics (max and sum) in one pass over blocks.
- Track running max `m` and running sum `l = sum(exp(x - m))`
- When max changes, rescale: `l = l * exp(m_old - m_new) + sum(exp(block - m_new))`
- Returns final `(m, l)` that can reconstruct softmax

### `online_softmax(x_blocks)`
Compute full softmax output incrementally.
- Use `online_softmax_stats` to get statistics
- Then compute `softmax(x) = exp(x - m) / l` for each block

### `online_attention_accumulator(Q_row, K_blocks, V_blocks)`
Compute attention output for one query row, incrementally.
- Track running max `m`, sum `l`, and output accumulator `o`
- For each K, V block: compute scores, update statistics, accumulate output
- Return final normalized output

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_online_softmax.py::TestSafeSoftmax

# Run with verbose output
uv run pytest tests/ -v
```

## The Key Insight

When we find a new maximum, we rescale everything:

```
Before block 2: m = 3, l = 1.5
Block 2 has max = 5 (new max!)

Rescale: l = 1.5 * exp(3 - 5) + sum(exp(block - 5))
                  ↑
           This rescales old sum to new max!
```

The output accumulator gets the same rescaling, ensuring correctness.

## Expected Results

Your online softmax should match standard softmax exactly:

```python
x = np.random.randn(100)

# Split into blocks
blocks = [x[i:i+10] for i in range(0, 100, 10)]

# Online version
online_result = online_softmax(blocks)

# Standard version
standard_result = safe_softmax(x)

# Should match!
np.testing.assert_allclose(online_result, standard_result, atol=1e-6)
```

## Verification

All tests pass = you understand the core insight of Flash Attention!

This online softmax + output accumulation is exactly what Flash Attention uses internally.
