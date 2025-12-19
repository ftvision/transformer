"""
Lab 01: Triton Basics

Implement simple Triton kernels to learn block-level programming.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/

Requirements:
- PyTorch 2.0+ (includes Triton)
- NVIDIA GPU with CUDA support
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# KERNEL 1: Vector Addition
# =============================================================================

@triton.jit
def add_kernel(
    x_ptr,          # Pointer to first input vector
    y_ptr,          # Pointer to second input vector
    output_ptr,     # Pointer to output vector
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per program
):
    """
    Triton kernel for element-wise vector addition.

    Each program instance processes BLOCK_SIZE elements.
    Total programs needed: ceil(n_elements / BLOCK_SIZE)

    Memory layout:
        Program 0: elements [0, BLOCK_SIZE)
        Program 1: elements [BLOCK_SIZE, 2*BLOCK_SIZE)
        ...

    Steps:
        1. Get program ID
        2. Compute element offsets for this program
        3. Create mask for boundary handling
        4. Load x and y values
        5. Compute x + y
        6. Store result
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement add_kernel")


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two vectors using Triton.

    Args:
        x: First input tensor (1D, on CUDA)
        y: Second input tensor (same shape as x, on CUDA)

    Returns:
        Element-wise sum x + y

    Example:
        >>> x = torch.randn(1024, device='cuda')
        >>> y = torch.randn(1024, device='cuda')
        >>> z = vector_add(x, y)
        >>> torch.allclose(z, x + y)
        True
    """
    # Validate inputs
    assert x.is_cuda and y.is_cuda, "Inputs must be on CUDA"
    assert x.shape == y.shape, "Shapes must match"

    # Allocate output
    output = torch.empty_like(x)
    n_elements = x.numel()

    # YOUR CODE HERE
    # 1. Choose BLOCK_SIZE (e.g., 1024)
    # 2. Compute grid size: ceil(n_elements / BLOCK_SIZE)
    # 3. Launch kernel: add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=...)
    raise NotImplementedError("Implement vector_add")

    return output


# =============================================================================
# KERNEL 2: Softmax
# =============================================================================

@triton.jit
def softmax_kernel(
    input_ptr,      # Pointer to input matrix
    output_ptr,     # Pointer to output matrix
    n_rows,         # Number of rows
    n_cols,         # Number of columns (row length)
    stride_row,     # Stride between rows
    BLOCK_SIZE: tl.constexpr,  # Must be >= n_cols
):
    """
    Triton kernel for row-wise softmax.

    Each program processes one row of the matrix.

    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))

    Steps:
        1. Get row index from program ID
        2. Compute row start pointer
        3. Load the entire row (use mask if n_cols < BLOCK_SIZE)
        4. Compute max for numerical stability
        5. Subtract max and compute exp
        6. Compute sum of exponentials
        7. Divide by sum to get softmax
        8. Store result

    Numerical stability note:
        Computing exp(x) directly can overflow for large x.
        Instead, compute exp(x - max(x)) which is mathematically equivalent
        but numerically stable.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement softmax_kernel")


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise softmax using Triton.

    Args:
        x: Input tensor of shape (n_rows, n_cols) on CUDA

    Returns:
        Softmax of x along the last dimension

    Example:
        >>> x = torch.randn(32, 128, device='cuda')
        >>> y = softmax(x)
        >>> torch.allclose(y, torch.softmax(x, dim=-1), atol=1e-5)
        True
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 2, "Input must be 2D (n_rows, n_cols)"

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # YOUR CODE HERE
    # 1. BLOCK_SIZE must be >= n_cols (use next power of 2)
    # 2. Grid size = n_rows (one program per row)
    # 3. Launch kernel
    raise NotImplementedError("Implement softmax")

    return output


# =============================================================================
# KERNEL 3: RMSNorm
# =============================================================================

@triton.jit
def rmsnorm_kernel(
    input_ptr,      # Pointer to input
    weight_ptr,     # Pointer to weight (gamma)
    output_ptr,     # Pointer to output
    n_rows,         # Number of rows
    n_cols,         # Number of columns (hidden dimension)
    stride_row,     # Stride between rows
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMSNorm.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Unlike LayerNorm, RMSNorm doesn't center the data (no mean subtraction).
    This makes it slightly more efficient while maintaining good performance.

    Each program processes one row.

    Steps:
        1. Load the row
        2. Compute sum of squares
        3. Compute mean of squares (divide by n_cols)
        4. Compute RMS = sqrt(mean + eps)
        5. Normalize: x / RMS
        6. Scale by weight
        7. Store result
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement rmsnorm_kernel")


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply RMSNorm using Triton.

    Args:
        x: Input tensor of shape (*, hidden_size) on CUDA
        weight: Learnable scale parameter of shape (hidden_size,) on CUDA
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as x

    Formula:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Example:
        >>> x = torch.randn(32, 128, device='cuda')
        >>> weight = torch.ones(128, device='cuda')
        >>> y = rmsnorm(x, weight)
        >>> # Compare with manual computation
        >>> rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        >>> expected = x / rms * weight
        >>> torch.allclose(y, expected, atol=1e-5)
        True
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert x.shape[-1] == weight.shape[0], "Hidden dims must match"

    # Handle arbitrary input shape by flattening to 2D
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)

    # YOUR CODE HERE
    # 1. BLOCK_SIZE must be >= n_cols
    # 2. Grid size = n_rows
    # 3. Launch kernel with eps parameter
    raise NotImplementedError("Implement rmsnorm")

    return output.view(original_shape)


# =============================================================================
# BONUS: Autotuned Softmax
# =============================================================================

# Uncomment to try autotuning after completing the basic implementations

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 128}),
#         triton.Config({'BLOCK_SIZE': 256}),
#         triton.Config({'BLOCK_SIZE': 512}),
#         triton.Config({'BLOCK_SIZE': 1024}),
#     ],
#     key=['n_cols'],
# )
# @triton.jit
# def softmax_kernel_autotuned(...):
#     ...
