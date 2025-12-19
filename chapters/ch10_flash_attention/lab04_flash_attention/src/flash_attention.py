"""
Lab 04: Flash Attention Integration

Learn to use the flash-attn library in practice.

Your task: Implement the wrapper and utility functions.
Run: uv run pytest tests/
"""

import time
from typing import Optional, Tuple

import numpy as np

# Try to import PyTorch and flash-attn
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


def has_flash_attention() -> bool:
    """
    Check if flash-attn library is available.

    Returns:
        True if flash-attn can be used, False otherwise

    Example:
        >>> if has_flash_attention():
        ...     # Use flash attention
        ... else:
        ...     # Fall back to standard attention
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement has_flash_attention")


def has_cuda() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement has_cuda")


def standard_attention_torch(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    V: "torch.Tensor",
    causal: bool = False,
    dropout_p: float = 0.0
) -> "torch.Tensor":
    """
    Standard scaled dot-product attention in PyTorch.

    Args:
        Q: (batch, seq_len, num_heads, d_head)
        K: (batch, seq_len, num_heads, d_head)
        V: (batch, seq_len, num_heads, d_head)
        causal: Whether to apply causal masking
        dropout_p: Dropout probability (ignored in this implementation)

    Returns:
        output: (batch, seq_len, num_heads, d_head)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Transpose to (batch, num_heads, seq_len, d_head)
    # 2. Compute scores = Q @ K.T / sqrt(d_k)
    # 3. Apply causal mask if needed
    # 4. Softmax
    # 5. Compute output = attn @ V
    # 6. Transpose back
    raise NotImplementedError("Implement standard_attention_torch")


class FlashAttentionWrapper:
    """
    Wrapper for Flash Attention library.

    Provides a consistent interface for using Flash Attention
    with fallback to standard attention when unavailable.

    Attributes:
        dropout_p: Dropout probability
        causal: Whether to use causal masking
        use_flash: Whether Flash Attention is available and will be used
    """

    def __init__(
        self,
        dropout_p: float = 0.0,
        causal: bool = False
    ):
        """
        Initialize Flash Attention wrapper.

        Args:
            dropout_p: Dropout probability (0.0 = no dropout)
            causal: Whether to apply causal masking

        Example:
            >>> wrapper = FlashAttentionWrapper(dropout_p=0.1, causal=True)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        Q: "torch.Tensor",
        K: "torch.Tensor",
        V: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute attention using Flash Attention (or fallback).

        Args:
            Q: Queries (batch, seq_len, num_heads, d_head)
            K: Keys (batch, seq_len, num_heads, d_head)
            V: Values (batch, seq_len, num_heads, d_head)

        Returns:
            output: (batch, seq_len, num_heads, d_head)

        Example:
            >>> wrapper = FlashAttentionWrapper(causal=True)
            >>> output = wrapper.forward(Q, K, V)
        """
        # YOUR CODE HERE
        #
        # If Flash Attention is available and inputs are on CUDA:
        #   Use flash_attn_func(Q, K, V, dropout_p, causal=causal)
        # Otherwise:
        #   Fall back to standard_attention_torch
        raise NotImplementedError("Implement forward")

    def __call__(
        self,
        Q: "torch.Tensor",
        K: "torch.Tensor",
        V: "torch.Tensor"
    ) -> "torch.Tensor":
        """Allow calling instance like a function."""
        return self.forward(Q, K, V)


def benchmark_attention(
    seq_len: int,
    d_model: int,
    num_heads: int,
    batch_size: int = 1,
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> dict:
    """
    Benchmark Flash Attention vs standard attention.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        batch_size: Batch size
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with:
        - standard_time_ms: Average time for standard attention
        - flash_time_ms: Average time for Flash Attention (if available)
        - speedup: Flash speedup over standard
        - seq_len, d_model, num_heads, batch_size

    Example:
        >>> results = benchmark_attention(2048, 512, 8)
        >>> print(f"Speedup: {results['speedup']:.2f}x")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Check if CUDA is available
    # 2. Create random Q, K, V tensors on GPU
    # 3. Warmup both implementations
    # 4. Time standard attention
    # 5. Time flash attention (if available)
    # 6. Return results dictionary
    raise NotImplementedError("Implement benchmark_attention")


def compare_outputs(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    V: "torch.Tensor",
    causal: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> dict:
    """
    Compare Flash Attention output with standard attention.

    Args:
        Q: Queries (batch, seq_len, num_heads, d_head)
        K: Keys (batch, seq_len, num_heads, d_head)
        V: Values (batch, seq_len, num_heads, d_head)
        causal: Whether to use causal masking
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary with:
        - max_diff: Maximum absolute difference
        - mean_diff: Mean absolute difference
        - allclose: Whether outputs are close within tolerance
        - flash_available: Whether Flash Attention was used

    Example:
        >>> result = compare_outputs(Q, K, V, causal=True)
        >>> assert result['allclose'], "Outputs should match"
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute standard attention output
    # 2. Compute flash attention output (if available)
    # 3. Compare outputs
    # 4. Return comparison dictionary
    raise NotImplementedError("Implement compare_outputs")


def measure_memory_usage(
    seq_len: int,
    d_model: int,
    num_heads: int,
    batch_size: int = 1
) -> dict:
    """
    Measure GPU memory usage for standard vs Flash Attention.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        batch_size: Batch size

    Returns:
        Dictionary with:
        - standard_memory_mb: Memory used by standard attention
        - flash_memory_mb: Memory used by Flash Attention
        - memory_saved_mb: Memory saved by using Flash
        - memory_reduction: Ratio of standard / flash

    Example:
        >>> results = measure_memory_usage(4096, 512, 8)
        >>> print(f"Memory saved: {results['memory_saved_mb']:.1f}MB")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Check CUDA availability
    # 2. Clear GPU cache
    # 3. Measure memory before/after standard attention
    # 4. Clear GPU cache
    # 5. Measure memory before/after flash attention
    # 6. Return memory comparison
    raise NotImplementedError("Implement measure_memory_usage")
