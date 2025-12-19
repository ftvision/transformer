"""Tests for Lab 04: Quantization Effects."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantization import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_block,
    dequantize_block,
    compute_quantization_error,
    simulate_quantized_matmul,
    analyze_layer_sensitivity,
    compute_memory_savings,
    simulate_mixed_precision,
    QuantizedLinear,
)


class TestSymmetricQuantization:
    """Tests for symmetric quantization."""

    def test_quantize_8bit(self):
        """8-bit quantization should work correctly."""
        weights = np.array([0.5, -0.3, 0.8, -0.6, 1.0, -1.0])
        quantized, scale = quantize_symmetric(weights, bits=8)

        # Quantized values should be integers
        assert quantized.dtype in [np.int8, np.int16, np.int32]

        # Max quantized value should be close to 127 (for 8-bit signed)
        assert abs(quantized).max() == 127

    def test_quantize_4bit(self):
        """4-bit quantization should work correctly."""
        weights = np.array([0.5, -0.3, 0.8, -0.6])
        quantized, scale = quantize_symmetric(weights, bits=4)

        # 4-bit signed: range is [-8, 7]
        assert quantized.min() >= -8
        assert quantized.max() <= 7

    def test_roundtrip_preserves_range(self):
        """Dequantized values should be in similar range to originals."""
        np.random.seed(42)
        weights = np.random.randn(100)

        quantized, scale = quantize_symmetric(weights, bits=8)
        reconstructed = dequantize_symmetric(quantized, scale)

        # Max absolute values should be close
        assert abs(reconstructed.max() - weights.max()) < 0.1
        assert abs(reconstructed.min() - weights.min()) < 0.1

    def test_zero_preserved(self):
        """Zero should quantize to zero."""
        weights = np.array([0.0, 1.0, -1.0])
        quantized, scale = quantize_symmetric(weights, bits=8)

        assert quantized[0] == 0

    def test_symmetric_around_zero(self):
        """Opposite values should have opposite quantized values."""
        weights = np.array([0.5, -0.5, 1.0, -1.0])
        quantized, scale = quantize_symmetric(weights, bits=8)

        assert quantized[0] == -quantized[1]
        assert quantized[2] == -quantized[3]


class TestBlockQuantization:
    """Tests for block quantization."""

    def test_block_quantization_shape(self):
        """Block quantization should produce correct shapes."""
        weights = np.random.randn(128)  # 4 blocks of 32
        quantized, scales = quantize_block(weights, bits=4, block_size=32)

        assert quantized.shape == weights.shape
        assert scales.shape == (4,)  # 4 blocks

    def test_block_roundtrip(self):
        """Block quantization roundtrip should preserve approximate values."""
        np.random.seed(42)
        weights = np.random.randn(256)

        quantized, scales = quantize_block(weights, bits=8, block_size=32)
        reconstructed = dequantize_block(quantized, scales, block_size=32)

        # 8-bit should have small error
        mse = ((weights - reconstructed) ** 2).mean()
        assert mse < 0.01  # Small MSE for 8-bit

    def test_different_block_sizes(self):
        """Should work with different block sizes."""
        weights = np.random.randn(256)

        q16, s16 = quantize_block(weights, bits=4, block_size=16)
        q64, s64 = quantize_block(weights, bits=4, block_size=64)

        assert s16.shape == (16,)  # 256/16 = 16 blocks
        assert s64.shape == (4,)   # 256/64 = 4 blocks

    def test_smaller_blocks_more_accurate(self):
        """Smaller blocks should generally be more accurate."""
        np.random.seed(42)
        # Create weights with varying magnitudes (hard case)
        weights = np.concatenate([
            np.random.randn(64) * 0.1,  # Small values
            np.random.randn(64) * 10.0  # Large values
        ])

        q_small, s_small = quantize_block(weights, bits=4, block_size=32)
        q_large, s_large = quantize_block(weights, bits=4, block_size=128)

        r_small = dequantize_block(q_small, s_small, block_size=32)
        r_large = dequantize_block(q_large, s_large, block_size=128)

        mse_small = ((weights - r_small) ** 2).mean()
        mse_large = ((weights - r_large) ** 2).mean()

        assert mse_small < mse_large


class TestQuantizationError:
    """Tests for error computation."""

    def test_error_metrics_computed(self):
        """Should compute all error metrics."""
        original = np.random.randn(100)
        reconstructed = original + np.random.randn(100) * 0.1

        errors = compute_quantization_error(original, reconstructed)

        assert 'mse' in errors
        assert 'max_error' in errors
        assert 'relative_error' in errors
        assert 'snr_db' in errors

    def test_perfect_reconstruction_zero_error(self):
        """Perfect reconstruction should have zero error."""
        original = np.random.randn(100)

        errors = compute_quantization_error(original, original)

        assert errors['mse'] == 0
        assert errors['max_error'] == 0

    def test_mse_increases_with_noise(self):
        """MSE should increase with more noise."""
        original = np.random.randn(100)
        small_noise = original + np.random.randn(100) * 0.01
        large_noise = original + np.random.randn(100) * 0.1

        errors_small = compute_quantization_error(original, small_noise)
        errors_large = compute_quantization_error(original, large_noise)

        assert errors_small['mse'] < errors_large['mse']

    def test_snr_positive_for_good_reconstruction(self):
        """SNR should be positive when signal > noise."""
        original = np.random.randn(100)
        reconstructed = original + np.random.randn(100) * 0.01

        errors = compute_quantization_error(original, reconstructed)

        assert errors['snr_db'] > 0


class TestQuantizedMatmul:
    """Tests for quantized matrix multiplication."""

    def test_matmul_shapes(self):
        """Output shapes should match regardless of quantization."""
        W = np.random.randn(64, 32)
        X = np.random.randn(8, 32)

        out_fp32, out_quant = simulate_quantized_matmul(W, X, bits=4)

        assert out_fp32.shape == (8, 64)
        assert out_quant.shape == (8, 64)

    def test_8bit_close_to_fp32(self):
        """8-bit quantization should be very close to FP32."""
        np.random.seed(42)
        W = np.random.randn(128, 64).astype(np.float32)
        X = np.random.randn(16, 64).astype(np.float32)

        out_fp32, out_quant = simulate_quantized_matmul(W, X, bits=8)

        # 8-bit should be very close
        relative_error = np.abs(out_fp32 - out_quant).mean() / np.abs(out_fp32).mean()
        assert relative_error < 0.01  # Less than 1% error

    def test_4bit_reasonable_error(self):
        """4-bit quantization should have reasonable error."""
        np.random.seed(42)
        W = np.random.randn(128, 64).astype(np.float32)
        X = np.random.randn(16, 64).astype(np.float32)

        out_fp32, out_quant = simulate_quantized_matmul(W, X, bits=4)

        # 4-bit has more error but should still be reasonable
        relative_error = np.abs(out_fp32 - out_quant).mean() / np.abs(out_fp32).mean()
        assert relative_error < 0.1  # Less than 10% error


class TestLayerSensitivity:
    """Tests for layer sensitivity analysis."""

    def test_returns_all_layers(self):
        """Should return sensitivity for all layers."""
        weights = {
            'layer1': np.random.randn(64, 64),
            'layer2': np.random.randn(128, 64),
        }

        sensitivity = analyze_layer_sensitivity(weights, bits_options=[8, 4])

        assert 'layer1' in sensitivity
        assert 'layer2' in sensitivity

    def test_returns_all_bit_widths(self):
        """Should return error for all bit widths."""
        weights = {'layer': np.random.randn(64, 64)}

        sensitivity = analyze_layer_sensitivity(weights, bits_options=[8, 4, 2])

        assert 8 in sensitivity['layer']
        assert 4 in sensitivity['layer']
        assert 2 in sensitivity['layer']

    def test_lower_bits_higher_error(self):
        """Lower bit widths should generally have higher error."""
        weights = {'layer': np.random.randn(256, 256)}

        sensitivity = analyze_layer_sensitivity(weights, bits_options=[8, 4, 2])

        assert sensitivity['layer'][8] < sensitivity['layer'][4]
        assert sensitivity['layer'][4] < sensitivity['layer'][2]


class TestMemorySavings:
    """Tests for memory savings calculation."""

    def test_4bit_compression_ratio(self):
        """4-bit should give ~4x compression over FP16."""
        result = compute_memory_savings(
            model_params=1_000_000,
            original_bits=16,
            quantized_bits=4
        )

        # Should be close to 4x (minus scale overhead)
        assert 3.5 < result['compression_ratio'] < 4.5

    def test_sizes_in_gb(self):
        """Should return sizes in GB."""
        result = compute_memory_savings(
            model_params=7_000_000_000,  # 7B
            quantized_bits=4
        )

        # 7B params at FP16 = 14 GB
        assert 13.5 < result['original_size_gb'] < 14.5

        # 7B params at 4-bit = ~3.5 GB + overhead
        assert 3 < result['quantized_size_gb'] < 5

    def test_scale_overhead_included(self):
        """Should account for scale factor storage overhead."""
        result = compute_memory_savings(
            model_params=1_000_000,
            quantized_bits=4,
            block_size=32
        )

        assert result['scale_overhead_gb'] > 0


class TestMixedPrecision:
    """Tests for mixed precision quantization."""

    def test_mixed_precision_total_size(self):
        """Should calculate total size with mixed precision."""
        weights = {
            'attn': np.random.randn(512, 512),  # 262144 params
            'ffn': np.random.randn(2048, 512),  # 1048576 params
        }
        layer_bits = {'attn': 6, 'ffn': 4}

        result = simulate_mixed_precision(weights, layer_bits)

        assert 'total_original_bytes' in result
        assert 'total_quantized_bytes' in result
        assert result['total_quantized_bytes'] < result['total_original_bytes']

    def test_effective_bits_calculated(self):
        """Should calculate effective average bits."""
        weights = {
            'layer1': np.random.randn(100, 100),  # 10000 params
            'layer2': np.random.randn(100, 100),  # 10000 params
        }
        layer_bits = {'layer1': 8, 'layer2': 4}  # Average should be 6

        result = simulate_mixed_precision(weights, layer_bits)

        assert 5.5 < result['effective_bits'] < 6.5  # Close to 6

    def test_per_layer_errors_returned(self):
        """Should return error for each layer."""
        weights = {
            'layer1': np.random.randn(64, 64),
            'layer2': np.random.randn(64, 64),
        }
        layer_bits = {'layer1': 4, 'layer2': 8}

        result = simulate_mixed_precision(weights, layer_bits)

        assert 'layer1' in result['per_layer_errors']
        assert 'layer2' in result['per_layer_errors']


class TestQuantizedLinear:
    """Tests for the QuantizedLinear layer."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=4)

        # Should be able to get memory usage
        assert layer.memory_bytes() > 0

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=4)

        x = np.random.randn(8, 64)
        output = layer.forward(x)

        assert output.shape == (8, 128)

    def test_callable(self):
        """Layer should be callable."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=4)

        x = np.random.randn(8, 64)
        output = layer(x)  # Call like a function

        assert output.shape == (8, 128)

    def test_memory_savings(self):
        """Quantized layer should use less memory than FP32."""
        in_feat, out_feat = 512, 512

        layer = QuantizedLinear(in_feat, out_feat, bits=4, block_size=32)

        # FP32 would use: 512 * 512 * 4 bytes = 1MB
        fp32_bytes = in_feat * out_feat * 4

        # 4-bit should use roughly 1/8 of that (plus scale overhead)
        assert layer.memory_bytes() < fp32_bytes / 4

    def test_different_bit_widths(self):
        """Different bit widths should affect memory usage."""
        layer_4bit = QuantizedLinear(256, 256, bits=4)
        layer_8bit = QuantizedLinear(256, 256, bits=8)

        # 8-bit should use roughly 2x memory of 4-bit
        ratio = layer_8bit.memory_bytes() / layer_4bit.memory_bytes()
        assert 1.5 < ratio < 2.5
