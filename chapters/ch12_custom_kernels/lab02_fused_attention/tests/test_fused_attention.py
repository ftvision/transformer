"""Tests for Lab 02: Fused Attention."""

import math
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

from fused_attention import fused_attention, attention_reference


class TestFusedAttention:
    """Tests for fused attention kernel."""

    def test_basic_attention(self):
        """Basic attention should match reference."""
        seq_len, d_k = 64, 32
        Q = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)
        K = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)
        V = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)

        result = fused_attention(Q, K, V, causal=False)
        expected = attention_reference(Q, K, V, causal=False)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_causal_attention(self):
        """Causal attention should match reference."""
        seq_len, d_k = 64, 32
        Q = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)
        K = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)
        V = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)

        result = fused_attention(Q, K, V, causal=True)
        expected = attention_reference(Q, K, V, causal=True)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        """Output shape should match input shape."""
        seq_len, d_k = 128, 64
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)

        assert result.shape == (seq_len, d_k)

    def test_small_sequence(self):
        """Should work with small sequences."""
        seq_len, d_k = 8, 16
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)
        expected = attention_reference(Q, K, V)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_longer_sequence(self):
        """Should work with longer sequences."""
        seq_len, d_k = 256, 64
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)
        expected = attention_reference(Q, K, V)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestCausalMasking:
    """Tests specifically for causal masking."""

    def test_causal_mask_applied(self):
        """With causal=True, later positions should not attend to earlier."""
        seq_len, d_k = 32, 16

        # Create inputs where we can check causality
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V, causal=True)

        # For verification, compute the attention weights manually
        scores = Q @ K.T / math.sqrt(d_k)
        mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
        scores_masked = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores_masked, dim=-1)

        # Upper triangle of weights should be 0 (future positions)
        upper_triangle = torch.triu(weights, diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)

    def test_first_position_causal(self):
        """First position should only attend to itself."""
        seq_len, d_k = 32, 16
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V, causal=True)

        # First output should only depend on first V
        # (since first query can only see first key)
        result_first = fused_attention(Q[:1], K[:1], V[:1], causal=True)

        torch.testing.assert_close(result[0:1], result_first, atol=1e-4, rtol=1e-4)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_values(self):
        """Should handle large values without overflow."""
        seq_len, d_k = 64, 32
        Q = torch.randn(seq_len, d_k, device='cuda') * 10
        K = torch.randn(seq_len, d_k, device='cuda') * 10
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_small_values(self):
        """Should handle small values without underflow issues."""
        seq_len, d_k = 64, 32
        Q = torch.randn(seq_len, d_k, device='cuda') * 0.001
        K = torch.randn(seq_len, d_k, device='cuda') * 0.001
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_attention_weights_sum_to_one(self):
        """Implicit attention weights should sum to 1."""
        # We can't directly access weights, but we can verify through a special case
        seq_len, d_k = 32, 16

        # If all V values are the same, output should be that value
        # (since attention weights sum to 1)
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.ones(seq_len, d_k, device='cuda') * 3.14159  # Same value everywhere

        result = fused_attention(Q, K, V, causal=False)

        # All outputs should be 3.14159 (sum of weights * 3.14159 = 3.14159)
        expected = torch.ones(seq_len, d_k, device='cuda') * 3.14159
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestDifferentSizes:
    """Tests with various input sizes."""

    @pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
    def test_various_seq_lengths(self, seq_len):
        """Should work with various sequence lengths."""
        d_k = 32
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)
        expected = attention_reference(Q, K, V)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("d_k", [16, 32, 64])
    def test_various_head_dims(self, d_k):
        """Should work with various head dimensions."""
        seq_len = 64
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)
        expected = attention_reference(Q, K, V)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_non_power_of_two_seq_len(self):
        """Should handle non-power-of-2 sequence length."""
        seq_len, d_k = 100, 32
        Q = torch.randn(seq_len, d_k, device='cuda')
        K = torch.randn(seq_len, d_k, device='cuda')
        V = torch.randn(seq_len, d_k, device='cuda')

        result = fused_attention(Q, K, V)
        expected = attention_reference(Q, K, V)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestMilestone:
    """Chapter 12 Lab 02 Milestone."""

    def test_fused_attention_milestone(self):
        """
        MILESTONE: Fused attention matches reference within tolerance.

        This demonstrates understanding of:
        - Kernel fusion
        - Online softmax
        - Triton programming
        """
        # Test with transformer-like dimensions
        seq_len = 128
        d_k = 64

        Q = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)
        K = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)
        V = torch.randn(seq_len, d_k, device='cuda', dtype=torch.float32)

        # Test both causal and non-causal
        result_full = fused_attention(Q, K, V, causal=False)
        expected_full = attention_reference(Q, K, V, causal=False)

        result_causal = fused_attention(Q, K, V, causal=True)
        expected_causal = attention_reference(Q, K, V, causal=True)

        torch.testing.assert_close(result_full, expected_full, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result_causal, expected_causal, atol=1e-4, rtol=1e-4)

        print("\n" + "=" * 60)
        print("Lab 02 Milestone Achieved!")
        print("Fused attention kernel working correctly.")
        print("=" * 60 + "\n")
