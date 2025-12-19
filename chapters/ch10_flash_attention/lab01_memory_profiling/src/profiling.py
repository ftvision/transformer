"""
Lab 01: Memory Profiling

Profile standard attention memory usage to understand the O(N²) bottleneck.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax along the specified axis.

    Provided for convenience.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def standard_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute standard attention that stores the full attention matrix.

    This implementation is intentionally memory-inefficient - it stores
    the entire N×N attention matrix. This is what we're trying to avoid
    with Flash Attention!

    Args:
        Q: Query tensor of shape (seq_len, d_k) or (batch, seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k) or (batch, seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v) or (batch, seq_len, d_v)
        mask: Optional boolean mask. True values are masked (set to -inf)

    Returns:
        output: Attention output of shape (..., seq_len, d_v)
        attention_weights: Full attention matrix of shape (..., seq_len, seq_len)
                          THIS IS THE MEMORY BOTTLENECK!

    Example:
        >>> Q = np.random.randn(1024, 64).astype(np.float32)
        >>> K = np.random.randn(1024, 64).astype(np.float32)
        >>> V = np.random.randn(1024, 64).astype(np.float32)
        >>> output, attn = standard_attention(Q, K, V)
        >>> attn.shape
        (1024, 1024)  # This is 4 MB for float32!
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
    # 2. Apply mask if provided (set masked positions to -inf)
    # 3. Apply softmax to get attention weights (store this!)
    # 4. Compute output: output = attention_weights @ V
    # 5. Return both output and attention_weights
    raise NotImplementedError("Implement standard_attention")


def measure_attention_memory(
    seq_len: int,
    d_model: int,
    batch_size: int = 1,
    dtype: np.dtype = np.float32
) -> Dict[str, int]:
    """
    Measure the memory required for standard attention.

    Creates random Q, K, V matrices and computes attention,
    then measures the memory usage of each component.

    Args:
        seq_len: Sequence length (N)
        d_model: Model dimension (d)
        batch_size: Batch size (B)
        dtype: Data type (default float32)

    Returns:
        Dictionary containing memory usage in bytes:
        {
            'input_memory': bytes for Q, K, V,
            'attention_matrix_memory': bytes for N×N attention matrix,
            'output_memory': bytes for output,
            'total_memory': sum of above,
            'seq_len': input seq_len,
            'd_model': input d_model
        }

    Example:
        >>> stats = measure_attention_memory(1024, 64)
        >>> stats['attention_matrix_memory']
        4194304  # 1024 * 1024 * 4 bytes
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create random Q, K, V of shape (batch_size, seq_len, d_model)
    # 2. Compute attention using standard_attention
    # 3. Measure memory of each component using .nbytes
    # 4. Return the dictionary with all measurements
    raise NotImplementedError("Implement measure_attention_memory")


def profile_memory_scaling(
    seq_lengths: List[int],
    d_model: int = 64,
    batch_size: int = 1
) -> Dict[int, Dict[str, int]]:
    """
    Profile how memory scales with sequence length.

    Args:
        seq_lengths: List of sequence lengths to test
        d_model: Model dimension
        batch_size: Batch size

    Returns:
        Dictionary mapping seq_len to memory stats:
        {
            512: {'input_memory': ..., 'attention_matrix_memory': ..., ...},
            1024: {...},
            ...
        }

    Example:
        >>> results = profile_memory_scaling([512, 1024, 2048])
        >>> results[1024]['attention_matrix_memory']
        4194304
    """
    # YOUR CODE HERE
    #
    # For each seq_len in seq_lengths:
    #   Call measure_attention_memory and store results
    raise NotImplementedError("Implement profile_memory_scaling")


def estimate_attention_memory(
    seq_len: int,
    d_model: int,
    batch_size: int = 1,
    dtype_bytes: int = 4
) -> Dict[str, int]:
    """
    Theoretically estimate memory usage without actually allocating.

    This is useful for predicting memory requirements before running.

    Args:
        seq_len: Sequence length (N)
        d_model: Model dimension (d)
        batch_size: Batch size (B)
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)

    Returns:
        Dictionary with estimated memory:
        {
            'Q_memory': B × N × d × dtype_bytes,
            'K_memory': B × N × d × dtype_bytes,
            'V_memory': B × N × d × dtype_bytes,
            'attention_matrix_memory': B × N × N × dtype_bytes,
            'output_memory': B × N × d × dtype_bytes,
            'total_memory': sum of above
        }

    Example:
        >>> est = estimate_attention_memory(4096, 64)
        >>> est['attention_matrix_memory']
        67108864  # 4096 * 4096 * 4 = 64 MB
    """
    # YOUR CODE HERE
    #
    # Calculate each component's memory requirement
    raise NotImplementedError("Implement estimate_attention_memory")


def analyze_scaling(
    results: Dict[int, Dict[str, int]]
) -> Dict[str, any]:
    """
    Analyze the memory scaling pattern from profiling results.

    Args:
        results: Output from profile_memory_scaling

    Returns:
        Dictionary with analysis:
        {
            'seq_lengths': list of sequence lengths,
            'attention_memory': list of attention matrix memory values,
            'total_memory': list of total memory values,
            'scaling_factor': approximate factor when seq_len doubles
                             (should be ~4 for O(N²) attention matrix)
        }

    Example:
        >>> results = profile_memory_scaling([512, 1024])
        >>> analysis = analyze_scaling(results)
        >>> analysis['scaling_factor']
        4.0  # Memory 4x when seq_len 2x (quadratic!)
    """
    # YOUR CODE HERE
    #
    # Extract data from results and compute scaling factor
    raise NotImplementedError("Implement analyze_scaling")
