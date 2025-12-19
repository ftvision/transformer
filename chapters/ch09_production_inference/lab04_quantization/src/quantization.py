"""
Lab 04: Quantization Effects

Explore how quantization affects model weights and outputs.
This lab demonstrates the memory-quality tradeoffs of quantization.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    bits: int              # Number of bits (e.g., 4, 8)
    block_size: int = 32   # Number of values per block
    symmetric: bool = True # Whether to use symmetric quantization


def quantize_symmetric(
    weights: np.ndarray,
    bits: int
) -> Tuple[np.ndarray, float]:
    """
    Quantize weights using symmetric quantization.

    Symmetric quantization maps weights to integers symmetrically around 0.
    The formula is:
        scale = max(abs(weights)) / (2^(bits-1) - 1)
        quantized = round(weights / scale)

    Args:
        weights: Float weights to quantize, any shape
        bits: Number of bits for quantization (e.g., 8, 4)

    Returns:
        quantized: Integer quantized weights (stored as int8/int32)
        scale: Scale factor for dequantization

    Example:
        >>> w = np.array([0.5, -0.3, 0.8, -0.6])
        >>> q, scale = quantize_symmetric(w, bits=8)
        >>> # q will be integer values, scale allows reconstruction
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Find the maximum absolute value
    # 2. Calculate scale = max_abs / (2^(bits-1) - 1)
    # 3. Quantize: q = round(w / scale)
    # 4. Clip to valid range [-2^(bits-1), 2^(bits-1) - 1]
    # 5. Return quantized values and scale
    raise NotImplementedError("Implement quantize_symmetric")


def dequantize_symmetric(
    quantized: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Dequantize weights back to floating point.

    This reverses symmetric quantization: weights = quantized * scale

    Args:
        quantized: Integer quantized weights
        scale: Scale factor from quantization

    Returns:
        Reconstructed float weights

    Example:
        >>> q = np.array([127, -76, 101, -51])
        >>> scale = 0.0063
        >>> w_reconstructed = dequantize_symmetric(q, scale)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement dequantize_symmetric")


def quantize_block(
    weights: np.ndarray,
    bits: int,
    block_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize weights using per-block quantization.

    Block quantization quantizes each block of `block_size` values
    separately, allowing different scales for different parts of the
    weight matrix. This improves accuracy at the cost of storing
    more scale factors.

    Args:
        weights: Float weights, shape should be divisible by block_size
        bits: Number of bits for quantization
        block_size: Number of values per block

    Returns:
        quantized: Integer quantized weights
        scales: Scale factor per block

    Example:
        >>> w = np.random.randn(128)  # 4 blocks of 32
        >>> q, scales = quantize_block(w, bits=4, block_size=32)
        >>> scales.shape
        (4,)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Reshape weights into blocks: (num_blocks, block_size)
    # 2. For each block, compute scale = max(abs(block)) / max_int
    # 3. Quantize each block using its scale
    # 4. Return quantized values and array of scales
    raise NotImplementedError("Implement quantize_block")


def dequantize_block(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 32
) -> np.ndarray:
    """
    Dequantize block-quantized weights.

    Args:
        quantized: Integer quantized weights
        scales: Scale factor per block
        block_size: Number of values per block

    Returns:
        Reconstructed float weights
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement dequantize_block")


def compute_quantization_error(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> Dict[str, float]:
    """
    Compute various error metrics between original and reconstructed weights.

    Args:
        original: Original float weights
        reconstructed: Dequantized weights

    Returns:
        Dictionary with:
        - 'mse': Mean squared error
        - 'max_error': Maximum absolute error
        - 'relative_error': Mean |error| / mean |original|
        - 'snr_db': Signal-to-noise ratio in dB
    """
    # YOUR CODE HERE
    #
    # MSE = mean((original - reconstructed)^2)
    # Max error = max(|original - reconstructed|)
    # Relative error = mean(|error|) / mean(|original|)
    # SNR (dB) = 10 * log10(signal_power / noise_power)
    raise NotImplementedError("Implement compute_quantization_error")


def simulate_quantized_matmul(
    weights: np.ndarray,
    inputs: np.ndarray,
    bits: int,
    block_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate matrix multiplication with quantized weights.

    This demonstrates how quantized inference works:
    1. Weights are stored in quantized form (saves memory)
    2. For computation, we dequantize and multiply
    (In optimized implementations, this can be fused)

    Args:
        weights: Weight matrix, shape (out_features, in_features)
        inputs: Input matrix, shape (batch, in_features)
        bits: Quantization bits
        block_size: Block size for quantization

    Returns:
        output_fp32: Output using full precision weights
        output_quant: Output using quantized weights

    Example:
        >>> W = np.random.randn(512, 256)
        >>> X = np.random.randn(8, 256)
        >>> out_fp32, out_quant = simulate_quantized_matmul(W, X, bits=4)
        >>> # out_quant should be close to out_fp32
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute output_fp32 = inputs @ weights.T
    # 2. Quantize weights using block quantization
    # 3. Dequantize weights
    # 4. Compute output_quant = inputs @ dequantized_weights.T
    # 5. Return both outputs
    raise NotImplementedError("Implement simulate_quantized_matmul")


def analyze_layer_sensitivity(
    weights: Dict[str, np.ndarray],
    bits_options: list = [8, 4, 3, 2]
) -> Dict[str, Dict[int, float]]:
    """
    Analyze how different layers respond to quantization.

    Some layers (like attention projections) are more sensitive
    to quantization than others (like FFN layers).

    Args:
        weights: Dictionary mapping layer names to weight matrices
        bits_options: List of bit widths to test

    Returns:
        Dictionary mapping layer names to {bits: relative_error}

    Example:
        >>> weights = {
        ...     'attention.q_proj': np.random.randn(512, 512),
        ...     'ffn.up_proj': np.random.randn(2048, 512),
        ... }
        >>> sensitivity = analyze_layer_sensitivity(weights)
        >>> sensitivity['attention.q_proj'][4]  # Error at 4 bits
    """
    # YOUR CODE HERE
    #
    # For each layer and each bit width:
    # 1. Quantize and dequantize the weights
    # 2. Compute relative error
    # 3. Store in result dictionary
    raise NotImplementedError("Implement analyze_layer_sensitivity")


def compute_memory_savings(
    model_params: int,
    original_bits: int = 16,
    quantized_bits: int = 4,
    block_size: int = 32
) -> Dict[str, Any]:
    """
    Calculate memory savings from quantization.

    Args:
        model_params: Number of model parameters
        original_bits: Original precision (default: FP16 = 16 bits)
        quantized_bits: Target quantization bits
        block_size: Block size for quantization

    Returns:
        Dictionary with:
        - 'original_size_gb': Size in GB at original precision
        - 'quantized_size_gb': Size in GB after quantization
        - 'compression_ratio': original / quantized
        - 'scale_overhead_gb': Additional memory for scale factors

    Example:
        >>> result = compute_memory_savings(7_000_000_000, quantized_bits=4)
        >>> print(f"Compression: {result['compression_ratio']:.1f}x")
    """
    # YOUR CODE HERE
    #
    # Original size = model_params * original_bits / 8 bytes
    # Quantized size = model_params * quantized_bits / 8 bytes
    # Scale overhead = (model_params / block_size) * 2 bytes (FP16 scales)
    raise NotImplementedError("Implement compute_memory_savings")


def simulate_mixed_precision(
    weights: Dict[str, np.ndarray],
    layer_bits: Dict[str, int]
) -> Dict[str, Any]:
    """
    Simulate mixed-precision quantization (like K-quants in llama.cpp).

    Different layers can have different bit widths based on their
    sensitivity to quantization.

    Args:
        weights: Dictionary mapping layer names to weight matrices
        layer_bits: Dictionary mapping layer names to bit widths

    Returns:
        Dictionary with:
        - 'total_original_bytes': Total size at FP16
        - 'total_quantized_bytes': Total size with mixed precision
        - 'per_layer_errors': {layer: relative_error}
        - 'effective_bits': Average bits per parameter

    Example:
        >>> weights = {'attn': np.random.randn(512, 512), 'ffn': np.random.randn(2048, 512)}
        >>> bits = {'attn': 6, 'ffn': 4}  # More bits for attention
        >>> result = simulate_mixed_precision(weights, bits)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement simulate_mixed_precision")


class QuantizedLinear:
    """
    A linear layer with quantized weights.

    This simulates how quantized layers work in inference frameworks:
    - Weights are stored quantized (low memory)
    - On forward pass, dequantize and compute
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 32
    ):
        """
        Initialize a quantized linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bits: Quantization bits
            block_size: Block size for quantization
        """
        # YOUR CODE HERE
        #
        # Initialize random weights and quantize them
        # Store: quantized weights, scales, config
        raise NotImplementedError("Implement __init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through quantized linear layer.

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Output tensor of shape (batch, out_features)
        """
        # YOUR CODE HERE
        #
        # 1. Dequantize weights
        # 2. Compute x @ weights.T
        raise NotImplementedError("Implement forward")

    def memory_bytes(self) -> int:
        """Return memory used by this layer in bytes."""
        # YOUR CODE HERE
        #
        # Quantized weights + scale factors
        raise NotImplementedError("Implement memory_bytes")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling layer like a function."""
        return self.forward(x)
