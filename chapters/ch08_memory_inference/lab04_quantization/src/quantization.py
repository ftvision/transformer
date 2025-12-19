"""
Lab 04: Quantization Basics

Implement INT8 quantization from scratch.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple, Dict, Any


def quantize_symmetric(
    x: np.ndarray,
    num_bits: int = 8
) -> Tuple[np.ndarray, float]:
    """
    Symmetric quantization: map floats to signed integers.

    Symmetric quantization maps 0 to 0 and uses the same scale for positive
    and negative values. Good for distributions centered around zero (like weights).

    Formula:
        scale = max(|x|) / (2^(bits-1) - 1)
        q = clamp(round(x / scale), -2^(bits-1), 2^(bits-1) - 1)

    For 8-bit: scale = max(|x|) / 127, q ∈ [-128, 127]

    Args:
        x: Float tensor to quantize
        num_bits: Number of bits for quantization (default: 8)

    Returns:
        q: Quantized integer tensor (same shape as x)
        scale: Scale factor for dequantization

    Examples:
        >>> x = np.array([-1.0, 0.0, 0.5, 1.0])
        >>> q, scale = quantize_symmetric(x, num_bits=8)
        >>> scale
        0.007874...  # 1.0 / 127
        >>> q
        array([-127,    0,   64,  127])

        >>> # For uniform data in [-1, 1], scale ≈ 1/127
        >>> x = np.linspace(-1, 1, 100)
        >>> q, scale = quantize_symmetric(x)
        >>> np.abs(scale - 1/127) < 0.001
        True

    Note:
        - Returns int8 dtype for 8-bit quantization
        - Handle edge case where x is all zeros (scale = 1.0)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement quantize_symmetric")


def dequantize_symmetric(
    q: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Reverse symmetric quantization.

    Formula: x = q * scale

    Args:
        q: Quantized integer tensor
        scale: Scale factor from quantization

    Returns:
        Dequantized float tensor

    Examples:
        >>> q = np.array([-127, 0, 64, 127], dtype=np.int8)
        >>> scale = 1.0 / 127
        >>> x = dequantize_symmetric(q, scale)
        >>> x
        array([-1.        ,  0.        ,  0.503937..,  1.        ])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement dequantize_symmetric")


def quantize_asymmetric(
    x: np.ndarray,
    num_bits: int = 8
) -> Tuple[np.ndarray, float, int]:
    """
    Asymmetric quantization: uses zero_point for better range utilization.

    Asymmetric quantization maps the min value to 0 and max to 2^bits - 1.
    Better for asymmetric distributions (like ReLU activations that are all positive).

    Formula:
        scale = (max(x) - min(x)) / (2^bits - 1)
        zero_point = clamp(round(-min(x) / scale), 0, 2^bits - 1)
        q = clamp(round(x / scale + zero_point), 0, 2^bits - 1)

    For 8-bit: q ∈ [0, 255]

    Args:
        x: Float tensor to quantize
        num_bits: Number of bits for quantization (default: 8)

    Returns:
        q: Quantized unsigned integer tensor (same shape as x)
        scale: Scale factor for dequantization
        zero_point: Zero point offset

    Examples:
        >>> x = np.array([0.0, 0.5, 1.0, 1.5])
        >>> q, scale, zp = quantize_asymmetric(x, num_bits=8)
        >>> scale
        0.00588...  # 1.5 / 255
        >>> zp
        0  # min(x) = 0 maps to 0
        >>> q
        array([  0,  85, 170, 255])

        >>> # For data in [0, 1], full range is used
        >>> x = np.array([0.0, 1.0])
        >>> q, scale, zp = quantize_asymmetric(x)
        >>> q
        array([  0, 255])

    Note:
        - Returns uint8 dtype for 8-bit quantization
        - Handle edge case where x is constant (scale = 1.0, zp = 0)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement quantize_asymmetric")


def dequantize_asymmetric(
    q: np.ndarray,
    scale: float,
    zero_point: int
) -> np.ndarray:
    """
    Reverse asymmetric quantization.

    Formula: x = (q - zero_point) * scale

    Args:
        q: Quantized unsigned integer tensor
        scale: Scale factor from quantization
        zero_point: Zero point offset

    Returns:
        Dequantized float tensor

    Examples:
        >>> q = np.array([0, 85, 170, 255], dtype=np.uint8)
        >>> scale = 1.5 / 255
        >>> zero_point = 0
        >>> x = dequantize_asymmetric(q, scale, zero_point)
        >>> x
        array([0. , 0.5, 1. , 1.5])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement dequantize_asymmetric")


def quantize_per_channel(
    x: np.ndarray,
    axis: int = 0,
    num_bits: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel symmetric quantization.

    Instead of one scale for the entire tensor, compute a separate scale
    for each slice along the specified axis. This provides better accuracy
    when different channels have different value ranges.

    For weight matrices with shape (out_features, in_features), typically
    quantize per output channel (axis=0).

    Args:
        x: Float tensor to quantize
        axis: Axis along which to compute separate scales
        num_bits: Number of bits for quantization

    Returns:
        q: Quantized integer tensor (same shape as x)
        scales: Scale factors, one per slice along axis

    Examples:
        >>> # Weight matrix with different ranges per output channel
        >>> W = np.array([[1.0, 2.0, 3.0],    # max = 3
        ...               [0.1, 0.2, 0.3]])   # max = 0.3
        >>> W_q, scales = quantize_per_channel(W, axis=0)
        >>> scales.shape
        (2,)
        >>> scales[0] > scales[1]  # First row needs larger scale
        True

        >>> # Each row is quantized independently
        >>> W_q[0]  # Should use range [-127, 127] for values in [-3, 3]
        >>> W_q[1]  # Should use range [-127, 127] for values in [-0.3, 0.3]

    Note:
        - Uses symmetric quantization for each channel
        - scales has shape (x.shape[axis],)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement quantize_per_channel")


def dequantize_per_channel(
    q: np.ndarray,
    scales: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Reverse per-channel symmetric quantization.

    Args:
        q: Quantized integer tensor
        scales: Per-channel scale factors
        axis: Axis along which scales were computed

    Returns:
        Dequantized float tensor

    Examples:
        >>> q = np.array([[127, 127, 127],
        ...               [127, 127, 127]], dtype=np.int8)
        >>> scales = np.array([3.0/127, 0.3/127])
        >>> x = dequantize_per_channel(q, scales, axis=0)
        >>> x[0]  # ≈ [3, 3, 3]
        >>> x[1]  # ≈ [0.3, 0.3, 0.3]
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement dequantize_per_channel")


def quantization_error(
    original: np.ndarray,
    quantized: np.ndarray,
    dequantized: np.ndarray
) -> Dict[str, float]:
    """
    Calculate quantization error metrics.

    Args:
        original: Original float tensor
        quantized: Quantized integer tensor (for info, not used in error calc)
        dequantized: Dequantized float tensor

    Returns:
        Dictionary with:
            - mse: Mean squared error
            - mae: Mean absolute error
            - max_error: Maximum absolute error
            - relative_error: MSE / variance(original)
            - snr: Signal-to-noise ratio in dB (10 * log10(var(x) / mse))

    Examples:
        >>> x = np.random.randn(1000)
        >>> q, scale = quantize_symmetric(x)
        >>> x_deq = dequantize_symmetric(q, scale)
        >>> error = quantization_error(x, q, x_deq)
        >>> error['mse'] < 0.001  # Very small for normal distribution
        True
        >>> error['snr'] > 30  # High SNR (good)
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement quantization_error")


def quantized_matmul(
    a_q: np.ndarray,
    a_scale: float,
    b_q: np.ndarray,
    b_scale: float,
    a_zp: int = 0,
    b_zp: int = 0
) -> np.ndarray:
    """
    Perform matrix multiplication on quantized tensors.

    For symmetric quantization (zp=0):
        C = (A_q * a_scale) @ (B_q * b_scale)
          = A_q @ B_q * (a_scale * b_scale)

    For asymmetric quantization:
        C = ((A_q - a_zp) * a_scale) @ ((B_q - b_zp) * b_scale)

    The integer matmul can be done efficiently with INT8 tensor cores,
    then we rescale the result.

    Args:
        a_q: Quantized matrix A
        a_scale: Scale factor for A
        b_q: Quantized matrix B
        b_scale: Scale factor for B
        a_zp: Zero point for A (default: 0 for symmetric)
        b_zp: Zero point for B (default: 0 for symmetric)

    Returns:
        Float result of the quantized multiplication

    Examples:
        >>> A = np.random.randn(32, 64).astype(np.float32)
        >>> B = np.random.randn(64, 128).astype(np.float32)
        >>> A_q, a_scale = quantize_symmetric(A)
        >>> B_q, b_scale = quantize_symmetric(B)
        >>> C_quant = quantized_matmul(A_q, a_scale, B_q, b_scale)
        >>> C_float = A @ B
        >>> np.allclose(C_quant, C_float, rtol=0.01)  # ~1% error
        True

    Note:
        - Convert to int32 before matmul to prevent overflow
        - For asymmetric quantization, the formula expands to include
          correction terms for zero points
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement quantized_matmul")


def compare_quantization_methods(
    x: np.ndarray,
    num_bits: int = 8
) -> Dict[str, Dict[str, float]]:
    """
    Compare different quantization methods on the same data.

    Methods compared:
    - symmetric_per_tensor: Single scale for entire tensor
    - asymmetric_per_tensor: Scale + zero_point for entire tensor
    - symmetric_per_channel: Per-channel scales (axis=0)

    Args:
        x: Float tensor to quantize (2D array for per-channel comparison)
        num_bits: Number of bits for quantization

    Returns:
        Dictionary mapping method name to error metrics

    Examples:
        >>> W = np.random.randn(256, 256).astype(np.float32)
        >>> results = compare_quantization_methods(W)
        >>> results.keys()
        dict_keys(['symmetric_per_tensor', 'asymmetric_per_tensor', 'symmetric_per_channel'])
        >>> results['symmetric_per_channel']['mse'] < results['symmetric_per_tensor']['mse']
        True  # Per-channel is usually more accurate
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_quantization_methods")


def fake_quantize(
    x: np.ndarray,
    num_bits: int = 8,
    symmetric: bool = True
) -> np.ndarray:
    """
    Fake quantization: quantize then immediately dequantize.

    This simulates the effect of quantization during training (QAT).
    The output is float but has the same precision loss as true quantization.

    Args:
        x: Float tensor
        num_bits: Number of bits
        symmetric: Use symmetric (True) or asymmetric (False) quantization

    Returns:
        Float tensor with quantization effects applied

    Examples:
        >>> x = np.array([0.1, 0.2, 0.3])
        >>> x_fq = fake_quantize(x)
        >>> x_fq  # Similar but not exactly equal to x
        array([0.0984..., 0.2047..., 0.2992...])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement fake_quantize")


def calculate_memory_savings(
    original_dtype: np.dtype,
    quantized_bits: int,
    include_scale_overhead: bool = True,
    elements_per_scale: int = 1
) -> Dict[str, float]:
    """
    Calculate memory savings from quantization.

    Args:
        original_dtype: Original data type (e.g., np.float32, np.float16)
        quantized_bits: Bits per element after quantization
        include_scale_overhead: Include memory for scale factors
        elements_per_scale: How many elements share one scale
                           (1 for per-tensor, n for per-channel with n channels)

    Returns:
        Dictionary with:
            - compression_ratio: Original / quantized size
            - memory_reduction: Percentage reduction
            - effective_bits: Bits per element including overhead

    Examples:
        >>> # FP32 to INT8 per-tensor
        >>> savings = calculate_memory_savings(np.float32, 8, elements_per_scale=1000000)
        >>> savings['compression_ratio']
        4.0  # 32-bit / 8-bit

        >>> # FP16 to INT4 per-channel (256 channels, 1024 elements each)
        >>> savings = calculate_memory_savings(np.float16, 4, elements_per_scale=1024)
        >>> savings['compression_ratio'] > 3.5  # Close to 4x
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_memory_savings")
