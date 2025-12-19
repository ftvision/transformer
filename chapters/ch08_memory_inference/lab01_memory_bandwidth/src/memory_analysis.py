"""
Lab 01: Memory Bandwidth Analysis

Analyze memory-bound inference and understand the roofline model.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

from typing import Dict, Any


def calculate_model_memory(num_params: float, dtype_bytes: float = 2) -> float:
    """
    Calculate the memory required to store model weights.

    Args:
        num_params: Number of parameters in the model (e.g., 7e9 for 7B)
        dtype_bytes: Bytes per parameter
                     - 4 for fp32
                     - 2 for fp16/bf16
                     - 1 for int8
                     - 0.5 for int4

    Returns:
        Memory in bytes required to store the model weights

    Examples:
        >>> calculate_model_memory(7e9, dtype_bytes=2)
        14000000000.0  # 14 GB

        >>> calculate_model_memory(7e9, dtype_bytes=1)
        7000000000.0  # 7 GB (int8)

        >>> calculate_model_memory(70e9, dtype_bytes=2)
        140000000000.0  # 140 GB
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_model_memory")


def calculate_max_tokens_per_second(
    model_memory_bytes: float,
    bandwidth_bytes_per_sec: float
) -> float:
    """
    Calculate the theoretical maximum tokens per second.

    This is the absolute upper bound for single-token generation, where each
    token requires loading all model weights once from memory.

    Formula: max_tokens/sec = bandwidth / model_memory

    Args:
        model_memory_bytes: Memory required for model weights (bytes)
        bandwidth_bytes_per_sec: GPU memory bandwidth (bytes/second)

    Returns:
        Maximum theoretical tokens per second

    Examples:
        >>> # 7B model (14 GB in fp16) on A100 (2 TB/s)
        >>> calculate_max_tokens_per_second(14e9, 2e12)
        142.857...  # ~143 tokens/sec

        >>> # Same model on H100 (3.35 TB/s)
        >>> calculate_max_tokens_per_second(14e9, 3.35e12)
        239.285...  # ~239 tokens/sec

    Note:
        This is a theoretical maximum assuming:
        - Perfect memory bandwidth utilization
        - No other memory accesses (activations, KV-cache, etc.)
        - No compute overhead
        Actual throughput will be lower.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_max_tokens_per_second")


def calculate_arithmetic_intensity(flops: float, bytes_loaded: float) -> float:
    """
    Calculate the arithmetic intensity of an operation.

    Arithmetic intensity = FLOPS / bytes_loaded

    This metric determines whether an operation is memory-bound or compute-bound:
    - Low intensity: Memory-bound (waiting for data)
    - High intensity: Compute-bound (limited by compute)

    Args:
        flops: Number of floating-point operations
        bytes_loaded: Number of bytes loaded from memory

    Returns:
        Arithmetic intensity (FLOPS per byte)

    Examples:
        >>> # Matrix multiply: lots of compute, load matrix once
        >>> calculate_arithmetic_intensity(1e12, 1e9)
        1000.0  # High intensity, compute-bound

        >>> # Single token inference: load all weights for few ops
        >>> calculate_arithmetic_intensity(14e9, 14e9)
        1.0  # Low intensity, memory-bound

    Note:
        The "ridge point" on the roofline model is where:
        intensity = compute_capability / memory_bandwidth

        For A100: 312 TFLOPS / 2 TB/s = 156 FLOPS/byte
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_arithmetic_intensity")


def is_memory_bound(
    arithmetic_intensity: float,
    compute_flops: float,
    bandwidth_bytes_per_sec: float
) -> bool:
    """
    Determine if an operation is memory-bound or compute-bound.

    Uses the roofline model:
    - Ridge point = compute_capability / bandwidth
    - If arithmetic_intensity < ridge_point: memory-bound
    - If arithmetic_intensity >= ridge_point: compute-bound

    Args:
        arithmetic_intensity: FLOPS per byte for the operation
        compute_flops: GPU compute capability (FLOPS)
        bandwidth_bytes_per_sec: GPU memory bandwidth (bytes/second)

    Returns:
        True if memory-bound, False if compute-bound

    Examples:
        >>> # A100: 312 TFLOPS, 2 TB/s → ridge = 156
        >>> is_memory_bound(1.0, 312e12, 2e12)
        True  # intensity (1) < ridge (156)

        >>> is_memory_bound(200.0, 312e12, 2e12)
        False  # intensity (200) >= ridge (156)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement is_memory_bound")


def calculate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    d_model: int,
    dtype_bytes: float = 2
) -> float:
    """
    Calculate KV-cache memory requirements.

    The KV-cache stores Key and Value tensors for all previous tokens
    across all layers to avoid recomputation during autoregressive generation.

    Formula: batch × seq_len × num_layers × 2 × d_model × dtype_bytes
             (the 2 is for K and V)

    Args:
        batch_size: Number of sequences in the batch
        seq_len: Sequence length (number of tokens cached)
        num_layers: Number of transformer layers
        d_model: Model dimension (hidden size)
        dtype_bytes: Bytes per value (2 for fp16)

    Returns:
        KV-cache memory in bytes

    Examples:
        >>> # Llama-2-7B: 32 layers, d_model=4096
        >>> calculate_kv_cache_memory(1, 2048, 32, 4096, 2)
        1073741824.0  # 1 GB

        >>> calculate_kv_cache_memory(8, 2048, 32, 4096, 2)
        8589934592.0  # 8 GB

    Note:
        KV-cache memory can easily dominate GPU memory for large batches
        or long sequences. This is why techniques like PagedAttention,
        GQA (grouped-query attention), and KV-cache quantization are important.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_kv_cache_memory")


def calculate_attention_flops(
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int,
    d_model: int,
    num_heads: int
) -> float:
    """
    Calculate approximate FLOPs for multi-head attention computation.

    Attention consists of:
    1. QK^T: (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_k)
       → multiply-add = 2 × batch × heads × seq_q × seq_k × head_dim
    2. Softmax: approximately 5 ops per element (exp, sum, div, etc.)
       → 5 × batch × heads × seq_q × seq_k
    3. Attention @ V: (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, head_dim)
       → 2 × batch × heads × seq_q × seq_k × head_dim

    Total ≈ 4 × batch × heads × seq_q × seq_k × head_dim + 5 × batch × heads × seq_q × seq_k

    Simplified: batch × heads × seq_q × seq_k × (4 × head_dim + 5)

    Args:
        batch_size: Number of sequences in the batch
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        d_model: Model dimension
        num_heads: Number of attention heads

    Returns:
        Approximate total FLOPs for attention

    Examples:
        >>> # Single query attending to 1024 keys, d_model=4096, 32 heads
        >>> calculate_attention_flops(1, 1, 1024, 4096, 32)
        4259840.0  # ~4.3M FLOPs

        >>> # Prefill: 512 queries attending to 512 keys
        >>> calculate_attention_flops(1, 512, 512, 4096, 32)
        1113587712.0  # ~1.1B FLOPs
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_attention_flops")


def analyze_inference_bottleneck(
    model_config: Dict[str, Any],
    gpu_config: Dict[str, Any],
    batch_size: int = 1,
    seq_len: int = 1
) -> Dict[str, Any]:
    """
    Comprehensive analysis of inference characteristics.

    Analyzes whether inference is memory-bound or compute-bound and
    calculates key metrics for understanding performance.

    Args:
        model_config: Dictionary with model parameters:
            - num_params: Total parameters (e.g., 7e9)
            - num_layers: Number of transformer layers
            - d_model: Model dimension
            - num_heads: Number of attention heads
            - dtype_bytes: Bytes per parameter (default: 2)
        gpu_config: Dictionary with GPU specifications:
            - bandwidth_bytes_per_sec: Memory bandwidth (e.g., 2e12)
            - compute_flops: Compute capability (e.g., 312e12)
            - memory_bytes: Total GPU memory (e.g., 80e9)
        batch_size: Number of sequences being processed
        seq_len: Sequence length for KV-cache calculation

    Returns:
        Dictionary with analysis results:
            - model_memory_bytes: Memory for model weights
            - max_tokens_per_sec: Theoretical maximum throughput
            - kv_cache_memory_bytes: KV-cache memory for given batch/seq
            - total_memory_bytes: Model + KV-cache memory
            - fits_in_memory: Whether total fits in GPU memory
            - arithmetic_intensity: FLOPS per byte for single-token decode
            - is_memory_bound: Whether decode is memory-bound
            - ridge_point: The compute/bandwidth ratio

    Examples:
        >>> model = {'num_params': 7e9, 'num_layers': 32,
        ...          'd_model': 4096, 'num_heads': 32, 'dtype_bytes': 2}
        >>> gpu = {'bandwidth_bytes_per_sec': 2e12,
        ...        'compute_flops': 312e12, 'memory_bytes': 80e9}
        >>> result = analyze_inference_bottleneck(model, gpu)
        >>> result['is_memory_bound']
        True
        >>> result['max_tokens_per_sec']
        142.857...
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement analyze_inference_bottleneck")
