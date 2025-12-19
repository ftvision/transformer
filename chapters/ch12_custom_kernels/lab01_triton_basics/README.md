# Lab 01: Triton Basics

## Objective

Learn Triton's block-level programming model by implementing simple GPU kernels.

## What You'll Build

Three Triton kernels:
1. Vector addition (the "hello world" of GPU programming)
2. Softmax (a reduction operation)
3. RMSNorm (combining reduction and element-wise ops)

## Prerequisites

Read these docs first:
- `../docs/01_triton_basics.md`
- `../docs/02_kernel_fusion.md`

## Requirements

- PyTorch 2.0+ (includes Triton)
- NVIDIA GPU with CUDA support

```bash
# Verify Triton is available
python -c "import triton; print(triton.__version__)"
```

## Instructions

1. Open `src/kernels.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `add_kernel` and `vector_add(x, y)`

Implement vector addition in Triton.

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load, compute, store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

### `softmax_kernel` and `softmax(x)`

Implement row-wise softmax in Triton.

Key steps:
1. Load a row
2. Compute max for numerical stability
3. Subtract max and compute exp
4. Sum the exponentials
5. Divide by sum
6. Store result

### `rmsnorm_kernel` and `rmsnorm(x, weight, eps)`

Implement RMSNorm: `output = x / sqrt(mean(x^2) + eps) * weight`

This combines:
- Reduction (computing mean of squares)
- Element-wise operations (division, multiplication)

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_kernels.py::TestVectorAdd

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### Block Size
- Use powers of 2: 64, 128, 256, 512, 1024
- For softmax/rmsnorm, BLOCK_SIZE should be >= row length

### Masking
Always use masks for boundary conditions:
```python
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### Reductions
Use `tl.sum()` for summation:
```python
total = tl.sum(x, axis=0)
```

### Common Patterns
```python
# Program ID
pid = tl.program_id(axis=0)

# Generate offsets for this block
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# Load with mask
data = tl.load(ptr + offsets, mask=offsets < n, other=0.0)

# Store with mask
tl.store(ptr + offsets, data, mask=offsets < n)
```

## Expected Output

Your Triton kernels should match PyTorch's output within numerical tolerance:

```python
# Vector add
assert torch.allclose(vector_add(x, y), x + y)

# Softmax
assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-5)

# RMSNorm
expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
assert torch.allclose(rmsnorm(x, weight, eps), expected, atol=1e-5)
```

## Verification

All tests pass = you've mastered Triton basics!

Next: Lab 02 where you'll implement a fused attention kernel.
