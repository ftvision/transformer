"""Tests for Lab 01: Triton Basics."""

import numpy as np
import pytest
import sys
from pathlib import Path

import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernels import vector_add, softmax, rmsnorm


class TestVectorAdd:
    """Tests for vector addition kernel."""

    def test_basic_addition(self):
        """Basic vector addition should work."""
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')

        result = vector_add(x, y)
        expected = x + y

        torch.testing.assert_close(result, expected)

    def test_small_vector(self):
        """Should work with small vectors."""
        x = torch.randn(10, device='cuda')
        y = torch.randn(10, device='cuda')

        result = vector_add(x, y)
        expected = x + y

        torch.testing.assert_close(result, expected)

    def test_large_vector(self):
        """Should work with large vectors."""
        x = torch.randn(100000, device='cuda')
        y = torch.randn(100000, device='cuda')

        result = vector_add(x, y)
        expected = x + y

        torch.testing.assert_close(result, expected)

    def test_non_power_of_two(self):
        """Should handle non-power-of-2 sizes."""
        x = torch.randn(1234, device='cuda')
        y = torch.randn(1234, device='cuda')

        result = vector_add(x, y)
        expected = x + y

        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(512, device='cuda')
        y = torch.randn(512, device='cuda')

        result = vector_add(x, y)

        assert result.shape == x.shape

    def test_preserves_dtype(self):
        """Should preserve input dtype."""
        x = torch.randn(1024, device='cuda', dtype=torch.float32)
        y = torch.randn(1024, device='cuda', dtype=torch.float32)

        result = vector_add(x, y)

        assert result.dtype == x.dtype


class TestSoftmax:
    """Tests for softmax kernel."""

    def test_basic_softmax(self):
        """Basic softmax should match PyTorch."""
        x = torch.randn(32, 128, device='cuda')

        result = softmax(x)
        expected = torch.softmax(x, dim=-1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_sums_to_one(self):
        """Each row should sum to 1."""
        x = torch.randn(32, 64, device='cuda')

        result = softmax(x)
        row_sums = result.sum(dim=-1)

        torch.testing.assert_close(
            row_sums,
            torch.ones(32, device='cuda'),
            atol=1e-5, rtol=1e-5
        )

    def test_positive_outputs(self):
        """All outputs should be positive."""
        x = torch.randn(32, 64, device='cuda')

        result = softmax(x)

        assert torch.all(result > 0)

    def test_numerical_stability(self):
        """Should handle large values without overflow."""
        x = torch.randn(32, 64, device='cuda') * 100  # Large values

        result = softmax(x)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_preserves_order(self):
        """Higher inputs should give higher outputs."""
        x = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')

        result = softmax(x)

        assert result[0, 2] > result[0, 1] > result[0, 0]

    def test_different_sizes(self):
        """Should work with various row lengths."""
        for n_cols in [32, 64, 128, 256]:
            x = torch.randn(16, n_cols, device='cuda')
            result = softmax(x)
            expected = torch.softmax(x, dim=-1)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_single_row(self):
        """Should work with single row."""
        x = torch.randn(1, 64, device='cuda')

        result = softmax(x)
        expected = torch.softmax(x, dim=-1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestRMSNorm:
    """Tests for RMSNorm kernel."""

    def test_basic_rmsnorm(self):
        """Basic RMSNorm should match manual computation."""
        x = torch.randn(32, 128, device='cuda')
        weight = torch.ones(128, device='cuda')
        eps = 1e-6

        result = rmsnorm(x, weight, eps)

        # Manual computation
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        expected = x / rms * weight

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_with_learned_weights(self):
        """Should correctly apply learned weights."""
        x = torch.randn(32, 128, device='cuda')
        weight = torch.randn(128, device='cuda')  # Non-unit weights
        eps = 1e-6

        result = rmsnorm(x, weight, eps)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        expected = x / rms * weight

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(32, 128, device='cuda')
        weight = torch.ones(128, device='cuda')

        result = rmsnorm(x, weight)

        assert result.shape == x.shape

    def test_different_hidden_sizes(self):
        """Should work with various hidden dimensions."""
        for hidden_size in [64, 128, 256, 512]:
            x = torch.randn(16, hidden_size, device='cuda')
            weight = torch.ones(hidden_size, device='cuda')

            result = rmsnorm(x, weight)

            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            expected = x / rms * weight

            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_3d_input(self):
        """Should handle 3D input (batch, seq, hidden)."""
        x = torch.randn(4, 32, 128, device='cuda')
        weight = torch.ones(128, device='cuda')
        eps = 1e-6

        result = rmsnorm(x, weight, eps)

        # Manual computation
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        expected = x / rms * weight

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_numerical_stability(self):
        """Should be numerically stable for small values."""
        x = torch.randn(32, 128, device='cuda') * 0.001  # Small values
        weight = torch.ones(128, device='cuda')

        result = rmsnorm(x, weight)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_eps_prevents_division_by_zero(self):
        """Epsilon should prevent division by zero."""
        x = torch.zeros(32, 128, device='cuda')  # All zeros
        weight = torch.ones(128, device='cuda')

        result = rmsnorm(x, weight, eps=1e-6)

        assert not torch.any(torch.isnan(result))
        # Result should be zero (0 / sqrt(eps) * weight = 0)
        torch.testing.assert_close(
            result,
            torch.zeros_like(result),
            atol=1e-5, rtol=1e-5
        )


class TestPerformance:
    """Performance sanity checks."""

    def test_vector_add_faster_than_pytorch(self):
        """Triton vector add should be competitive with PyTorch."""
        # This test just ensures the implementation runs, not that it's faster
        # Triton should be similar speed for simple ops
        x = torch.randn(1000000, device='cuda')
        y = torch.randn(1000000, device='cuda')

        # Warmup
        _ = vector_add(x, y)
        _ = x + y

        # Just verify it works for large inputs
        result = vector_add(x, y)
        expected = x + y

        torch.testing.assert_close(result, expected)

    def test_softmax_correct_for_transformer_sizes(self):
        """Softmax should work for typical transformer sizes."""
        # Typical attention: (batch * heads, seq_len, seq_len)
        x = torch.randn(32, 512, device='cuda')  # 32 heads, 512 seq len

        result = softmax(x)
        expected = torch.softmax(x, dim=-1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
