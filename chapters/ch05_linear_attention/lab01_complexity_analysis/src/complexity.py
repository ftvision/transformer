"""
Lab 01: Complexity Analysis

Benchmark standard attention to understand O(n²) complexity.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
import time
from typing import Tuple, List


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
    V: np.ndarray
) -> np.ndarray:
    """
    Compute standard scaled dot-product attention.

    This is the O(n²) attention we're benchmarking.

    Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        Q: Query tensor of shape (seq_len, d_k) or (batch, seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k) or (batch, seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v) or (batch, seq_len, d_v)

    Returns:
        Output tensor of shape (seq_len, d_v) or (batch, seq_len, d_v)

    Example:
        >>> Q = np.random.randn(100, 64)
        >>> K = np.random.randn(100, 64)
        >>> V = np.random.randn(100, 64)
        >>> output = standard_attention(Q, K, V)
        >>> output.shape
        (100, 64)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get d_k from Q's last dimension
    # 2. Compute attention scores: Q @ K^T
    # 3. Scale by sqrt(d_k)
    # 4. Apply softmax
    # 5. Multiply by V
    raise NotImplementedError("Implement standard_attention")


def measure_attention_time(
    seq_len: int,
    d_model: int,
    num_runs: int = 10
) -> Tuple[float, float]:
    """
    Measure average time to compute attention for a given sequence length.

    Creates random Q, K, V tensors and times the attention computation.

    Args:
        seq_len: Sequence length to test
        d_model: Model dimension (d_k = d_v = d_model)
        num_runs: Number of runs to average over

    Returns:
        Tuple of (mean_time_ms, std_time_ms) in milliseconds

    Example:
        >>> mean_time, std_time = measure_attention_time(512, 64)
        >>> print(f"Time: {mean_time:.2f} ± {std_time:.2f} ms")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create random Q, K, V tensors of shape (seq_len, d_model)
    # 2. Warm up with 2-3 runs (don't time these)
    # 3. Time num_runs executions using time.perf_counter()
    # 4. Return mean and std in milliseconds
    raise NotImplementedError("Implement measure_attention_time")


def measure_attention_memory(seq_len: int, d_model: int) -> int:
    """
    Calculate the memory required for the attention matrix.

    The attention matrix has shape (seq_len, seq_len) and stores float32 values.

    Args:
        seq_len: Sequence length
        d_model: Model dimension (not used for attention matrix, but
                 included for consistency)

    Returns:
        Memory in bytes for the attention matrix

    Note:
        This is a theoretical calculation, not actual memory measurement.
        In practice, intermediate tensors may require more memory.

    Example:
        >>> memory = measure_attention_memory(1024, 64)
        >>> print(f"Memory: {memory / 1e6:.1f} MB")
        Memory: 4.2 MB
    """
    # YOUR CODE HERE
    #
    # The attention matrix is (seq_len, seq_len)
    # Each element is float32 (4 bytes)
    raise NotImplementedError("Implement measure_attention_memory")


def fit_complexity_curve(
    seq_lengths: List[int],
    times: List[float]
) -> float:
    """
    Fit a power law to timing data: time = a * seq_len^b

    Uses log-log linear regression to find the exponent b.

    If times follow O(n²), then b should be close to 2.0.
    If times follow O(n), then b should be close to 1.0.

    Args:
        seq_lengths: List of sequence lengths tested
        times: List of corresponding execution times (same units)

    Returns:
        The fitted exponent b

    Example:
        >>> seq_lengths = [256, 512, 1024, 2048]
        >>> times = [1.0, 4.0, 16.0, 64.0]  # Perfect O(n²)
        >>> exponent = fit_complexity_curve(seq_lengths, times)
        >>> abs(exponent - 2.0) < 0.1
        True
    """
    # YOUR CODE HERE
    #
    # To fit time = a * n^b:
    # 1. Take log of both sides: log(time) = log(a) + b * log(n)
    # 2. This is a linear equation: y = c + b * x
    #    where y = log(time), x = log(n), c = log(a)
    # 3. Use np.polyfit(x, y, 1) to get [b, c]
    # 4. Return b (the exponent)
    raise NotImplementedError("Implement fit_complexity_curve")


def find_max_seq_length(memory_limit_mb: float, d_model: int) -> int:
    """
    Find maximum sequence length that fits in memory budget.

    Only considers the attention matrix (seq_len × seq_len × 4 bytes).

    Args:
        memory_limit_mb: Memory limit in megabytes
        d_model: Model dimension (not used for attention matrix calculation)

    Returns:
        Maximum sequence length (integer)

    Example:
        >>> max_len = find_max_seq_length(64.0, 64)  # 64 MB limit
        >>> max_len
        4096  # 4096² × 4 bytes = 64 MB
    """
    # YOUR CODE HERE
    #
    # memory = seq_len² × 4 bytes
    # seq_len² = memory / 4
    # seq_len = sqrt(memory / 4)
    #
    # Don't forget to convert MB to bytes (1 MB = 1e6 bytes)
    raise NotImplementedError("Implement find_max_seq_length")


def benchmark_attention_scaling(
    seq_lengths: List[int],
    d_model: int = 64,
    num_runs: int = 5
) -> dict:
    """
    Run a full benchmark across multiple sequence lengths.

    This is a convenience function that combines all measurements.

    Args:
        seq_lengths: List of sequence lengths to test
        d_model: Model dimension
        num_runs: Number of runs per sequence length

    Returns:
        Dictionary with:
        - 'seq_lengths': The input sequence lengths
        - 'times_ms': Mean execution times in milliseconds
        - 'times_std': Standard deviations of times
        - 'memory_mb': Memory usage in megabytes
        - 'fitted_exponent': The complexity exponent

    Example:
        >>> results = benchmark_attention_scaling([256, 512, 1024])
        >>> print(f"Complexity exponent: {results['fitted_exponent']:.2f}")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. For each seq_length, call measure_attention_time and measure_attention_memory
    # 2. Collect all results
    # 3. Fit the complexity curve
    # 4. Return the dictionary
    raise NotImplementedError("Implement benchmark_attention_scaling")
