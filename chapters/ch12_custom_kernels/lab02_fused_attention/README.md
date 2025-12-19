# Lab 02: Fused Attention in Triton

## Objective

Implement a fused attention kernel in Triton that combines scaling, softmax, and the attention computation into a single kernel.

## What You'll Build

A fused attention kernel that:
1. Computes Q @ K^T
2. Applies scaling by 1/√d_k
3. Applies optional causal mask
4. Computes softmax
5. Computes attention_weights @ V

All in one kernel pass, minimizing memory traffic.

## Prerequisites

- Complete Lab 01 (Triton basics)
- Read `../docs/01_triton_basics.md`
- Read `../docs/02_kernel_fusion.md`

## Why Fusion Matters

Standard attention launches multiple kernels:
```python
# Kernel 1: MatMul
scores = Q @ K.T

# Kernel 2: Scale
scores = scores / sqrt(d_k)

# Kernel 3: Mask (optional)
scores = scores + mask

# Kernel 4: Softmax
weights = softmax(scores)

# Kernel 5: MatMul
output = weights @ V
```

Each kernel:
- Reads data from HBM
- Writes results back to HBM
- Has launch overhead

Fused attention: **one read, one write**.

## Instructions

1. Open `src/fused_attention.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `fused_attention_kernel`

The main Triton kernel that computes attention in a single pass.

Key challenges:
- Tiling: Process the attention matrix in blocks
- Online softmax: Compute softmax incrementally as you process K blocks
- Memory management: Keep intermediate results in SRAM

### `fused_attention(Q, K, V, causal=False)`

Python wrapper that:
1. Validates inputs
2. Sets up the kernel grid
3. Launches the kernel
4. Returns the output

## The Online Softmax Algorithm

Computing softmax requires knowing the maximum value, but we process in blocks.
Solution: **online softmax** (also called "safe softmax with running max").

```
For each block of K:
    1. Compute partial scores: block_scores = Q_block @ K_block.T
    2. Update running max: new_max = max(old_max, block_scores.max())
    3. Rescale old accumulator: acc = acc * exp(old_max - new_max)
    4. Add new contribution: acc += exp(block_scores - new_max) @ V_block
    5. Update normalizer: normalizer = normalizer * exp(old_max - new_max) + exp(block_scores - new_max).sum()

Final: output = acc / normalizer
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_fused_attention.py::TestFusedAttention

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### Block Sizes
```python
BLOCK_M = 64   # Queries per block
BLOCK_N = 64   # Keys per block
BLOCK_K = 64   # Head dimension (usually same as d_k)
```

### Memory Access Patterns
- Q is accessed row by row (one block of queries at a time)
- K and V are accessed multiple times (once per Q block)
- Keep Q block in registers, iterate over K/V blocks

### Causal Masking
For causal attention, mask out positions where key_pos > query_pos:
```python
# In kernel
offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
causal_mask = offs_m[:, None] >= offs_n[None, :]
scores = tl.where(causal_mask, scores, float('-inf'))
```

### Numerical Stability
- Track `max_so_far` and `sum_so_far` for online softmax
- Use `float('-inf')` for masked positions before softmax

## Expected Output

Your fused attention should match standard attention:

```python
# Standard attention (PyTorch)
scores = Q @ K.T / sqrt(d_k)
if causal:
    mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
expected = weights @ V

# Your implementation
result = fused_attention(Q, K, V, causal=causal)

# Should match!
assert torch.allclose(result, expected, atol=1e-4)
```

## Simplifications

For this lab, we make some simplifications compared to Flash Attention:
- Single head only (no multi-head batching)
- No dropout
- Fixed block sizes (no autotuning)
- Forward pass only (no backward)

The goal is to understand the core fusion concept, not production optimization.

## Verification

All tests pass = you've implemented fused attention!

This is a major milestone: the core idea behind Flash Attention.

## Bonus Challenges (Optional)

1. Add multi-head support
2. Implement the backward pass
3. Add autotuning for block sizes
4. Compare performance against `torch.nn.functional.scaled_dot_product_attention`
