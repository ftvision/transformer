"""
Lab 05: Profiling and Optimization

Learn to profile deep learning code and apply targeted optimizations.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/

Requirements:
- PyTorch 2.0+
- CUDA (for GPU profiling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Callable, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def sync_if_cuda():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_function(
    fn: Callable,
    args: Tuple,
    warmup: int = 5,
    iterations: int = 100,
    sync: bool = True
) -> float:
    """
    Benchmark a function with proper warmup and synchronization.

    Args:
        fn: Function to benchmark
        args: Arguments to pass to the function
        warmup: Number of warmup iterations (includes compilation)
        iterations: Number of timed iterations
        sync: Whether to synchronize CUDA after each call

    Returns:
        Average time per iteration in seconds

    IMPORTANT: GPU operations are asynchronous!
    Without synchronization, you'll measure launch time, not execution time.

    Example:
        >>> def my_fn(x):
        ...     return x @ x.T
        >>> x = torch.randn(1000, 1000, device='cuda')
        >>> avg_time = benchmark_function(my_fn, (x,))
        >>> print(f"Average time: {avg_time * 1000:.2f} ms")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Warmup phase
    #    for _ in range(warmup):
    #        result = fn(*args)
    #        if sync:
    #            sync_if_cuda()
    #
    # 2. Timed phase
    #    start = time.perf_counter()
    #    for _ in range(iterations):
    #        result = fn(*args)
    #        if sync:
    #            sync_if_cuda()
    #    end = time.perf_counter()
    #
    # 3. Return average time
    #    return (end - start) / iterations

    raise NotImplementedError("Implement benchmark_function")


@dataclass
class ProfileResult:
    """Result from profiling a component."""
    name: str
    time_ms: float
    memory_mb: float = 0.0


def profile_attention_components(
    seq_len: int,
    d_model: int,
    num_heads: int,
    device: str = 'cuda'
) -> List[ProfileResult]:
    """
    Profile each component of multi-head attention separately.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        List of ProfileResult for each component:
        - qkv_projection: Linear projection of Q, K, V
        - score_computation: Q @ K.T / sqrt(d_k)
        - softmax: Softmax over scores
        - weighted_sum: weights @ V
        - output_projection: Final linear projection

    Example:
        >>> results = profile_attention_components(512, 768, 12)
        >>> for r in results:
        ...     print(f"{r.name}: {r.time_ms:.2f} ms")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create input tensor
    #    x = torch.randn(seq_len, d_model, device=device)
    #
    # 2. Create projection weights
    #    W_q = torch.randn(d_model, d_model, device=device)
    #    W_k = torch.randn(d_model, d_model, device=device)
    #    W_v = torch.randn(d_model, d_model, device=device)
    #    W_o = torch.randn(d_model, d_model, device=device)
    #
    # 3. Profile QKV projection
    #    def qkv_proj():
    #        return x @ W_q, x @ W_k, x @ W_v
    #    qkv_time = benchmark_function(qkv_proj, (), warmup=3, iterations=50) * 1000
    #
    # 4. Profile score computation
    #    d_k = d_model // num_heads
    #    Q = torch.randn(num_heads, seq_len, d_k, device=device)
    #    K = torch.randn(num_heads, seq_len, d_k, device=device)
    #    def score_fn():
    #        return torch.bmm(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    #    score_time = benchmark_function(score_fn, (), warmup=3, iterations=50) * 1000
    #
    # 5. Profile softmax
    #    scores = torch.randn(num_heads, seq_len, seq_len, device=device)
    #    def softmax_fn():
    #        return F.softmax(scores, dim=-1)
    #    softmax_time = benchmark_function(softmax_fn, (), warmup=3, iterations=50) * 1000
    #
    # 6. Profile weighted sum
    #    weights = torch.randn(num_heads, seq_len, seq_len, device=device)
    #    V = torch.randn(num_heads, seq_len, d_k, device=device)
    #    def weighted_sum_fn():
    #        return torch.bmm(weights, V)
    #    weighted_sum_time = benchmark_function(weighted_sum_fn, (), warmup=3, iterations=50) * 1000
    #
    # 7. Profile output projection
    #    concat = torch.randn(seq_len, d_model, device=device)
    #    def output_proj_fn():
    #        return concat @ W_o
    #    output_time = benchmark_function(output_proj_fn, (), warmup=3, iterations=50) * 1000
    #
    # 8. Return results
    #    return [
    #        ProfileResult("qkv_projection", qkv_time),
    #        ProfileResult("score_computation", score_time),
    #        ProfileResult("softmax", softmax_time),
    #        ProfileResult("weighted_sum", weighted_sum_time),
    #        ProfileResult("output_projection", output_time),
    #    ]

    raise NotImplementedError("Implement profile_attention_components")


def identify_bottleneck(results: List[ProfileResult]) -> Tuple[str, float]:
    """
    Identify the bottleneck from profile results.

    Args:
        results: List of ProfileResult from profiling

    Returns:
        Tuple of (bottleneck_name, percentage_of_total)

    Example:
        >>> results = [
        ...     ProfileResult("qkv_projection", 2.0),
        ...     ProfileResult("score_computation", 5.0),
        ...     ProfileResult("softmax", 1.0),
        ... ]
        >>> name, pct = identify_bottleneck(results)
        >>> print(f"Bottleneck: {name} ({pct:.1f}%)")
        Bottleneck: score_computation (62.5%)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Calculate total time
    #    total_time = sum(r.time_ms for r in results)
    #
    # 2. Find the component with maximum time
    #    bottleneck = max(results, key=lambda r: r.time_ms)
    #
    # 3. Calculate percentage
    #    percentage = (bottleneck.time_ms / total_time) * 100
    #
    # 4. Return (name, percentage)
    #    return bottleneck.name, percentage

    raise NotImplementedError("Implement identify_bottleneck")


# =============================================================================
# Memory Profiling
# =============================================================================

def measure_memory_usage(
    fn: Callable,
    args: Tuple,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure memory usage of a function.

    Args:
        fn: Function to profile
        args: Arguments to pass to the function
        device: Device to measure on

    Returns:
        Dictionary with:
        - peak_memory_mb: Peak memory allocated during execution
        - current_memory_mb: Memory allocated after execution
        - cached_memory_mb: Memory held by the caching allocator

    Example:
        >>> def my_fn(x):
        ...     return x @ x.T
        >>> x = torch.randn(1000, 1000, device='cuda')
        >>> mem = measure_memory_usage(my_fn, (x,))
        >>> print(f"Peak: {mem['peak_memory_mb']:.2f} MB")
    """
    if not torch.cuda.is_available():
        return {
            'peak_memory_mb': 0.0,
            'current_memory_mb': 0.0,
            'cached_memory_mb': 0.0,
        }

    # YOUR CODE HERE
    #
    # Steps:
    # 1. Reset peak memory stats
    #    torch.cuda.reset_peak_memory_stats()
    #
    # 2. Clear cache for accurate measurement
    #    torch.cuda.empty_cache()
    #
    # 3. Run the function
    #    result = fn(*args)
    #    torch.cuda.synchronize()
    #
    # 4. Get memory stats
    #    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    #    current = torch.cuda.memory_allocated() / (1024 ** 2)
    #    cached = torch.cuda.memory_reserved() / (1024 ** 2)
    #
    # 5. Return dictionary
    #    return {
    #        'peak_memory_mb': peak,
    #        'current_memory_mb': current,
    #        'cached_memory_mb': cached,
    #    }

    raise NotImplementedError("Implement measure_memory_usage")


# =============================================================================
# Optimization Techniques
# =============================================================================

def attention_baseline(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Baseline attention implementation (no optimization).

    Args:
        Q: Query tensor (batch, heads, seq_q, d_k)
        K: Key tensor (batch, heads, seq_k, d_k)
        V: Value tensor (batch, heads, seq_k, d_v)
        mask: Optional attention mask

    Returns:
        Output tensor (batch, heads, seq_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


def attention_optimized(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    optimization_level: int = 1
) -> torch.Tensor:
    """
    Optimized attention with different optimization levels.

    Args:
        Q: Query tensor (batch, heads, seq_q, d_k)
        K: Key tensor (batch, heads, seq_k, d_k)
        V: Value tensor (batch, heads, seq_k, d_v)
        mask: Optional attention mask
        optimization_level:
            0 = Baseline (no optimization)
            1 = Use scaled_dot_product_attention (Flash Attention when available)
            2 = Use torch.compile on baseline

    Returns:
        Output tensor (batch, heads, seq_q, d_v)

    Example:
        >>> Q = torch.randn(2, 8, 128, 64, device='cuda')
        >>> K = torch.randn(2, 8, 128, 64, device='cuda')
        >>> V = torch.randn(2, 8, 128, 64, device='cuda')
        >>> output = attention_optimized(Q, K, V, optimization_level=1)
    """
    # YOUR CODE HERE
    #
    # if optimization_level == 0:
    #     return attention_baseline(Q, K, V, mask)
    #
    # elif optimization_level == 1:
    #     # Use PyTorch's scaled_dot_product_attention
    #     # This automatically uses Flash Attention when possible
    #     attn_mask = None
    #     if mask is not None:
    #         # Convert boolean mask to float mask for SDPA
    #         attn_mask = mask.float().masked_fill(mask, float('-inf'))
    #     return F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    #
    # elif optimization_level == 2:
    #     # Use torch.compile
    #     compiled_attention = torch.compile(attention_baseline)
    #     return compiled_attention(Q, K, V, mask)
    #
    # else:
    #     raise ValueError(f"Unknown optimization level: {optimization_level}")

    raise NotImplementedError("Implement attention_optimized")


def compare_optimizations(
    batch: int,
    num_heads: int,
    seq_len: int,
    d_k: int,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compare different attention optimization levels.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        d_k: Head dimension
        device: Device to run on

    Returns:
        Dictionary mapping optimization level name to time in ms

    Example:
        >>> results = compare_optimizations(4, 8, 512, 64)
        >>> for name, time_ms in results.items():
        ...     print(f"{name}: {time_ms:.2f} ms")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create test tensors
    #    Q = torch.randn(batch, num_heads, seq_len, d_k, device=device)
    #    K = torch.randn(batch, num_heads, seq_len, d_k, device=device)
    #    V = torch.randn(batch, num_heads, seq_len, d_k, device=device)
    #
    # 2. Benchmark each optimization level
    #    results = {}
    #
    #    # Baseline
    #    baseline_time = benchmark_function(
    #        lambda: attention_optimized(Q, K, V, optimization_level=0),
    #        (), warmup=3, iterations=20
    #    ) * 1000
    #    results['baseline'] = baseline_time
    #
    #    # SDPA (Flash Attention)
    #    sdpa_time = benchmark_function(
    #        lambda: attention_optimized(Q, K, V, optimization_level=1),
    #        (), warmup=3, iterations=20
    #    ) * 1000
    #    results['sdpa'] = sdpa_time
    #
    #    # torch.compile (skip if it causes issues)
    #    try:
    #        compile_time = benchmark_function(
    #            lambda: attention_optimized(Q, K, V, optimization_level=2),
    #            (), warmup=3, iterations=20
    #        ) * 1000
    #        results['compile'] = compile_time
    #    except Exception:
    #        results['compile'] = float('nan')
    #
    # 3. Return results
    #    return results

    raise NotImplementedError("Implement compare_optimizations")


# =============================================================================
# Profiling Report
# =============================================================================

def generate_profile_report(
    seq_len: int,
    d_model: int,
    num_heads: int,
    device: str = 'cuda'
) -> str:
    """
    Generate a formatted profiling report.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        device: Device to run on

    Returns:
        Formatted string report

    Example output:
        Attention Component Profiling:
        ========================================
        Component           Time (ms)    % Total
        ----------------------------------------
        qkv_projection      2.31         15.4%
        score_computation   5.67         37.8%
        ...
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Run profiling
    #    results = profile_attention_components(seq_len, d_model, num_heads, device)
    #
    # 2. Calculate total time
    #    total_time = sum(r.time_ms for r in results)
    #
    # 3. Identify bottleneck
    #    bottleneck_name, bottleneck_pct = identify_bottleneck(results)
    #
    # 4. Format report
    #    lines = [
    #        "Attention Component Profiling:",
    #        "=" * 50,
    #        f"{'Component':<25} {'Time (ms)':<12} {'% Total':<10}",
    #        "-" * 50,
    #    ]
    #
    #    for r in results:
    #        pct = (r.time_ms / total_time) * 100 if total_time > 0 else 0
    #        lines.append(f"{r.name:<25} {r.time_ms:<12.2f} {pct:<10.1f}%")
    #
    #    lines.extend([
    #        "-" * 50,
    #        f"{'Total':<25} {total_time:<12.2f} {'100.0':<10}%",
    #        "",
    #        f"Bottleneck: {bottleneck_name} ({bottleneck_pct:.1f}%)",
    #    ])
    #
    #    return "\n".join(lines)

    raise NotImplementedError("Implement generate_profile_report")


# =============================================================================
# Arithmetic Intensity Analysis
# =============================================================================

def compute_arithmetic_intensity(
    operation: str,
    shapes: Dict[str, Tuple[int, ...]]
) -> float:
    """
    Compute arithmetic intensity (FLOPS / bytes) for an operation.

    Args:
        operation: One of 'matmul', 'softmax', 'layernorm', 'gelu'
        shapes: Dictionary of tensor shapes

    Returns:
        Arithmetic intensity in FLOPS/byte

    Higher AI = more compute-bound
    Lower AI = more memory-bound

    Example:
        >>> ai = compute_arithmetic_intensity('matmul', {'A': (1024, 1024), 'B': (1024, 1024)})
        >>> print(f"Arithmetic intensity: {ai:.2f} FLOPS/byte")
    """
    # YOUR CODE HERE
    #
    # if operation == 'matmul':
    #     # C = A @ B
    #     # FLOPS: 2 * M * N * K (multiply-add)
    #     # Bytes: (M*K + K*N + M*N) * 4 (read A, read B, write C, float32)
    #     M, K = shapes['A']
    #     _, N = shapes['B']
    #     flops = 2 * M * N * K
    #     bytes_moved = (M * K + K * N + M * N) * 4
    #     return flops / bytes_moved
    #
    # elif operation == 'softmax':
    #     # ~5 ops per element: max, sub, exp, sum, div
    #     # 2 passes over data
    #     batch, seq = shapes['x']
    #     flops = 5 * batch * seq
    #     bytes_moved = 2 * batch * seq * 4  # read + write
    #     return flops / bytes_moved
    #
    # elif operation == 'layernorm':
    #     # ~10 ops per element: mean, var, sub, div, scale, shift
    #     # 2 passes over data
    #     batch, hidden = shapes['x']
    #     flops = 10 * batch * hidden
    #     bytes_moved = 2 * batch * hidden * 4
    #     return flops / bytes_moved
    #
    # elif operation == 'gelu':
    #     # ~15 ops per element (approximation)
    #     # 1 pass over data
    #     size = 1
    #     for dim in shapes['x']:
    #         size *= dim
    #     flops = 15 * size
    #     bytes_moved = 2 * size * 4
    #     return flops / bytes_moved
    #
    # else:
    #     raise ValueError(f"Unknown operation: {operation}")

    raise NotImplementedError("Implement compute_arithmetic_intensity")


# =============================================================================
# BONUS: Mixed Precision Profiling
# =============================================================================

def profile_mixed_precision(
    fn: Callable,
    args: Tuple,
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """
    BONUS: Profile function with mixed precision.

    Args:
        fn: Function to profile
        args: Arguments to pass
        dtype: Precision to use (float16 or bfloat16)

    Returns:
        Dictionary with timing and memory for both fp32 and mixed precision
    """
    # YOUR CODE HERE (optional bonus challenge)
    raise NotImplementedError("Bonus: Implement profile_mixed_precision")
