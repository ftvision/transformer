"""Tests for Lab 04: Quantization Basics."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantization import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    dequantize_asymmetric,
    quantize_per_channel,
    dequantize_per_channel,
    quantization_error,
    quantized_matmul,
    compare_quantization_methods,
    fake_quantize,
    calculate_memory_savings,
)


class TestSymmetricQuantization:
    """Tests for symmetric quantization."""

    def test_basic_quantization(self):
        """Basic symmetric quantization."""
        x = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        q, scale = quantize_symmetric(x, num_bits=8)

        # Scale should be max(|x|) / 127
        assert abs(scale - 1.0 / 127) < 1e-6
        # Zero should quantize to zero
        assert q[1] == 0
        # Max should quantize to 127
        assert q[3] == 127
        # Min should quantize to -127
        assert q[0] == -127

    def test_output_dtype(self):
        """Output should be int8 for 8-bit quantization."""
        x = np.random.randn(100).astype(np.float32)
        q, _ = quantize_symmetric(x, num_bits=8)
        assert q.dtype == np.int8

    def test_range_clamping(self):
        """Values should be clamped to valid range."""
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        q, scale = quantize_symmetric(x, num_bits=8)

        # All values should be in [-128, 127]
        assert np.all(q >= -128)
        assert np.all(q <= 127)

    def test_zero_handling(self):
        """Should handle all-zero input."""
        x = np.zeros(100, dtype=np.float32)
        q, scale = quantize_symmetric(x)

        assert np.all(q == 0)
        assert scale > 0  # Should be some default positive scale

    def test_roundtrip_accuracy(self):
        """Quantize then dequantize should be close to original."""
        x = np.random.randn(1000).astype(np.float32)
        q, scale = quantize_symmetric(x)
        x_deq = dequantize_symmetric(q, scale)

        # Should have low error for normal distribution
        mse = np.mean((x - x_deq) ** 2)
        assert mse < 0.001


class TestDequantizeSymmetric:
    """Tests for symmetric dequantization."""

    def test_basic_dequantization(self):
        """Basic symmetric dequantization."""
        q = np.array([-127, 0, 64, 127], dtype=np.int8)
        scale = 1.0 / 127

        x = dequantize_symmetric(q, scale)

        np.testing.assert_allclose(x, [-1.0, 0.0, 64/127, 1.0], rtol=1e-5)

    def test_output_dtype(self):
        """Output should be float."""
        q = np.array([0, 50, 100], dtype=np.int8)
        x = dequantize_symmetric(q, 0.01)

        assert x.dtype in [np.float32, np.float64]


class TestAsymmetricQuantization:
    """Tests for asymmetric quantization."""

    def test_positive_only_data(self):
        """Asymmetric should efficiently handle positive-only data."""
        x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
        q, scale, zp = quantize_asymmetric(x, num_bits=8)

        # Min (0) should map to 0
        assert q[0] == 0
        # Max (1.5) should map to 255
        assert q[3] == 255
        # Zero point should be 0 for [0, max] range
        assert zp == 0

    def test_output_dtype(self):
        """Output should be uint8 for 8-bit asymmetric."""
        x = np.random.rand(100).astype(np.float32)  # Positive values
        q, _, _ = quantize_asymmetric(x, num_bits=8)
        assert q.dtype == np.uint8

    def test_shifted_data(self):
        """Should handle data not centered at zero."""
        x = np.array([10.0, 10.5, 11.0, 11.5], dtype=np.float32)
        q, scale, zp = quantize_asymmetric(x)

        # Should use full range [0, 255]
        assert q.min() == 0
        assert q.max() == 255

    def test_roundtrip_accuracy(self):
        """Quantize then dequantize should be close to original."""
        x = np.random.rand(1000).astype(np.float32) * 10  # [0, 10]
        q, scale, zp = quantize_asymmetric(x)
        x_deq = dequantize_asymmetric(q, scale, zp)

        mse = np.mean((x - x_deq) ** 2)
        assert mse < 0.01


class TestDequantizeAsymmetric:
    """Tests for asymmetric dequantization."""

    def test_basic_dequantization(self):
        """Basic asymmetric dequantization."""
        q = np.array([0, 85, 170, 255], dtype=np.uint8)
        scale = 1.5 / 255
        zero_point = 0

        x = dequantize_asymmetric(q, scale, zero_point)

        np.testing.assert_allclose(x, [0.0, 0.5, 1.0, 1.5], rtol=0.01)

    def test_with_zero_point(self):
        """Dequantization with non-zero zero point."""
        q = np.array([128, 191, 255], dtype=np.uint8)
        scale = 0.01
        zero_point = 128

        x = dequantize_asymmetric(q, scale, zero_point)

        # q - zp = [0, 63, 127], then multiply by scale
        np.testing.assert_allclose(x, [0.0, 0.63, 1.27], rtol=0.01)


class TestPerChannelQuantization:
    """Tests for per-channel quantization."""

    def test_different_scales_per_channel(self):
        """Each channel should have its own scale."""
        W = np.array([
            [1.0, 2.0, 3.0],    # max = 3
            [0.1, 0.2, 0.3]     # max = 0.3
        ], dtype=np.float32)

        W_q, scales = quantize_per_channel(W, axis=0)

        assert len(scales) == 2
        assert scales[0] > scales[1]  # First row needs larger scale
        # Both rows should use full range
        assert np.abs(W_q[0]).max() == 127
        assert np.abs(W_q[1]).max() == 127

    def test_output_shape(self):
        """Output should have same shape as input."""
        W = np.random.randn(64, 128).astype(np.float32)
        W_q, scales = quantize_per_channel(W, axis=0)

        assert W_q.shape == W.shape
        assert scales.shape == (64,)

    def test_axis_1(self):
        """Should work with axis=1."""
        W = np.random.randn(64, 128).astype(np.float32)
        W_q, scales = quantize_per_channel(W, axis=1)

        assert W_q.shape == W.shape
        assert scales.shape == (128,)

    def test_roundtrip_accuracy(self):
        """Per-channel should have good accuracy."""
        W = np.random.randn(256, 256).astype(np.float32)
        W_q, scales = quantize_per_channel(W, axis=0)
        W_deq = dequantize_per_channel(W_q, scales, axis=0)

        mse = np.mean((W - W_deq) ** 2)
        assert mse < 0.001


class TestQuantizationError:
    """Tests for quantization_error."""

    def test_returns_all_metrics(self):
        """Should return all expected error metrics."""
        x = np.random.randn(100).astype(np.float32)
        q, scale = quantize_symmetric(x)
        x_deq = dequantize_symmetric(q, scale)

        error = quantization_error(x, q, x_deq)

        assert 'mse' in error
        assert 'mae' in error
        assert 'max_error' in error
        assert 'relative_error' in error
        assert 'snr' in error

    def test_mse_calculation(self):
        """MSE should be correct."""
        x = np.array([1.0, 2.0, 3.0])
        x_deq = np.array([1.1, 2.0, 2.9])

        error = quantization_error(x, None, x_deq)

        expected_mse = np.mean([0.1**2, 0.0, 0.1**2])
        np.testing.assert_allclose(error['mse'], expected_mse)

    def test_snr_calculation(self):
        """SNR should be high for good quantization."""
        x = np.random.randn(10000).astype(np.float32)
        q, scale = quantize_symmetric(x)
        x_deq = dequantize_symmetric(q, scale)

        error = quantization_error(x, q, x_deq)

        # For 8-bit quantization, SNR should be > 30 dB
        assert error['snr'] > 30


class TestQuantizedMatmul:
    """Tests for quantized matrix multiplication."""

    def test_symmetric_matmul(self):
        """Quantized matmul with symmetric quantization."""
        np.random.seed(42)
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 128).astype(np.float32)

        A_q, a_scale = quantize_symmetric(A)
        B_q, b_scale = quantize_symmetric(B)

        C_quant = quantized_matmul(A_q, a_scale, B_q, b_scale)
        C_float = A @ B

        # Should be close (within ~5% relative error)
        relative_error = np.abs(C_quant - C_float).mean() / np.abs(C_float).mean()
        assert relative_error < 0.05

    def test_output_shape(self):
        """Output should have correct shape."""
        A_q = np.random.randint(-127, 128, (32, 64), dtype=np.int8)
        B_q = np.random.randint(-127, 128, (64, 128), dtype=np.int8)

        C = quantized_matmul(A_q, 0.01, B_q, 0.01)

        assert C.shape == (32, 128)

    def test_asymmetric_matmul(self):
        """Quantized matmul with asymmetric quantization (activations)."""
        np.random.seed(42)
        # Activations are often positive (after ReLU)
        A = np.abs(np.random.randn(32, 64)).astype(np.float32)
        B = np.random.randn(64, 128).astype(np.float32)

        A_q, a_scale, a_zp = quantize_asymmetric(A)
        B_q, b_scale = quantize_symmetric(B)

        C_quant = quantized_matmul(A_q, a_scale, B_q, b_scale, a_zp=a_zp)
        C_float = A @ B

        relative_error = np.abs(C_quant - C_float).mean() / np.abs(C_float).mean()
        assert relative_error < 0.1


class TestCompareQuantizationMethods:
    """Tests for compare_quantization_methods."""

    def test_returns_all_methods(self):
        """Should return results for all methods."""
        W = np.random.randn(64, 64).astype(np.float32)
        results = compare_quantization_methods(W)

        assert 'symmetric_per_tensor' in results
        assert 'asymmetric_per_tensor' in results
        assert 'symmetric_per_channel' in results

    def test_per_channel_better_or_equal(self):
        """Per-channel should generally be better than per-tensor."""
        # Create weights with varying scales per channel
        W = np.random.randn(64, 64).astype(np.float32)
        W[0, :] *= 10  # Make first row much larger

        results = compare_quantization_methods(W)

        # Per-channel should have lower error
        assert results['symmetric_per_channel']['mse'] <= results['symmetric_per_tensor']['mse'] * 1.1


class TestFakeQuantize:
    """Tests for fake_quantize."""

    def test_output_is_float(self):
        """Output should be float, not int."""
        x = np.random.randn(100).astype(np.float32)
        x_fq = fake_quantize(x)

        assert x_fq.dtype in [np.float32, np.float64]

    def test_output_shape(self):
        """Output should have same shape as input."""
        x = np.random.randn(32, 64).astype(np.float32)
        x_fq = fake_quantize(x)

        assert x_fq.shape == x.shape

    def test_introduces_quantization_noise(self):
        """Output should differ from input due to quantization."""
        x = np.linspace(-1, 1, 1000).astype(np.float32)
        x_fq = fake_quantize(x)

        # Should not be exactly equal
        assert not np.allclose(x, x_fq)
        # But should be close
        assert np.allclose(x, x_fq, atol=0.01)


class TestCalculateMemorySavings:
    """Tests for calculate_memory_savings."""

    def test_fp32_to_int8(self):
        """FP32 to INT8 should give 4x compression."""
        savings = calculate_memory_savings(
            np.float32, 8,
            include_scale_overhead=False
        )

        assert abs(savings['compression_ratio'] - 4.0) < 0.1
        assert abs(savings['memory_reduction'] - 75.0) < 1

    def test_fp16_to_int4(self):
        """FP16 to INT4 should give 4x compression."""
        savings = calculate_memory_savings(
            np.float16, 4,
            include_scale_overhead=False
        )

        assert abs(savings['compression_ratio'] - 4.0) < 0.1

    def test_scale_overhead(self):
        """Scale overhead should reduce effective compression."""
        # With many scales, overhead is significant
        savings_no_overhead = calculate_memory_savings(
            np.float32, 8,
            include_scale_overhead=False
        )
        savings_with_overhead = calculate_memory_savings(
            np.float32, 8,
            include_scale_overhead=True,
            elements_per_scale=10  # Very fine-grained
        )

        assert savings_with_overhead['compression_ratio'] < savings_no_overhead['compression_ratio']


class TestMilestone:
    """Integration tests for quantization."""

    def test_int8_quantization_accuracy(self):
        """INT8 quantization should achieve <0.1% relative error on typical weights."""
        np.random.seed(42)
        W = np.random.randn(256, 256).astype(np.float32)

        # Symmetric quantization
        W_q, scale = quantize_symmetric(W)
        W_deq = dequantize_symmetric(W_q, scale)

        error = quantization_error(W, W_q, W_deq)

        print("\n✅ Milestone Test - INT8 Quantization Accuracy")
        print(f"   MSE: {error['mse']:.6f}")
        print(f"   Relative error: {error['relative_error']*100:.4f}%")
        print(f"   SNR: {error['snr']:.1f} dB")
        print(f"   Max error: {error['max_error']:.4f}")

        # Should have very low relative error
        assert error['relative_error'] < 0.001  # < 0.1%
        assert error['snr'] > 35  # > 35 dB SNR

    def test_quantized_matmul_accuracy(self):
        """Quantized matmul should be close to float matmul."""
        np.random.seed(42)
        A = np.random.randn(64, 128).astype(np.float32)
        B = np.random.randn(128, 64).astype(np.float32)

        # Quantize
        A_q, a_scale = quantize_symmetric(A)
        B_q, b_scale = quantize_symmetric(B)

        # Quantized matmul
        C_quant = quantized_matmul(A_q, a_scale, B_q, b_scale)

        # Float matmul
        C_float = A @ B

        # Calculate error
        mse = np.mean((C_quant - C_float) ** 2)
        relative_error = mse / np.var(C_float)

        print("\n✅ Milestone Test - Quantized Matmul Accuracy")
        print(f"   Output MSE: {mse:.4f}")
        print(f"   Relative error: {relative_error*100:.2f}%")

        # Should be reasonably accurate
        assert relative_error < 0.01  # < 1% relative error

    def test_memory_savings_summary(self):
        """Summarize memory savings from different quantization levels."""
        print("\n✅ Milestone Test - Memory Savings Summary")

        configurations = [
            ("FP32 → INT8", np.float32, 8),
            ("FP32 → INT4", np.float32, 4),
            ("FP16 → INT8", np.float16, 8),
            ("FP16 → INT4", np.float16, 4),
        ]

        for name, dtype, bits in configurations:
            savings = calculate_memory_savings(dtype, bits, include_scale_overhead=False)
            print(f"   {name}: {savings['compression_ratio']:.1f}x compression, "
                  f"{savings['memory_reduction']:.0f}% reduction")

        # Basic sanity checks
        fp32_int8 = calculate_memory_savings(np.float32, 8, include_scale_overhead=False)
        assert fp32_int8['compression_ratio'] >= 3.9
