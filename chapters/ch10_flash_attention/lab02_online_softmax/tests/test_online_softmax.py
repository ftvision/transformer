"""Tests for Lab 02: Online Softmax."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from online_softmax import (
    safe_softmax,
    online_softmax_stats,
    online_softmax,
    online_attention_accumulator,
    compare_online_vs_standard,
)


class TestSafeSoftmax:
    """Tests for numerically stable softmax."""

    def test_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = np.array([1.0, 2.0, 3.0])
        result = safe_softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_preserves_order(self):
        """Higher inputs should give higher outputs."""
        x = np.array([1.0, 2.0, 3.0])
        result = safe_softmax(x)
        assert result[2] > result[1] > result[0]

    def test_numerical_stability(self):
        """Should handle large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = safe_softmax(x)

        assert not np.any(np.isnan(result)), "Should not produce NaN"
        assert not np.any(np.isinf(result)), "Should not produce Inf"
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_2d_along_axis(self):
        """Should work along specified axis for 2D."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = safe_softmax(x, axis=-1)

        # Each row should sum to 1
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)

    def test_uniform_input(self):
        """Equal inputs should give uniform output."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = safe_softmax(x)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestOnlineSoftmaxStats:
    """Tests for online softmax statistics computation."""

    def test_single_block(self):
        """Single block should give correct max and sum."""
        blocks = [np.array([1.0, 2.0, 3.0])]
        m, l = online_softmax_stats(blocks)

        assert m == 3.0, "Max should be 3.0"
        # l = exp(1-3) + exp(2-3) + exp(3-3) = exp(-2) + exp(-1) + 1
        expected_l = np.exp(-2) + np.exp(-1) + 1.0
        np.testing.assert_allclose(l, expected_l, rtol=1e-5)

    def test_multiple_blocks_same_max(self):
        """Multiple blocks with max in first block."""
        blocks = [np.array([5.0, 4.0]), np.array([3.0, 2.0])]
        m, l = online_softmax_stats(blocks)

        assert m == 5.0, "Max should be 5.0"

        # All values relative to max=5
        all_values = np.concatenate(blocks)
        expected_l = np.sum(np.exp(all_values - 5.0))
        np.testing.assert_allclose(l, expected_l, rtol=1e-5)

    def test_multiple_blocks_new_max(self):
        """Multiple blocks where later block has new max."""
        blocks = [np.array([1.0, 2.0]), np.array([5.0, 3.0])]
        m, l = online_softmax_stats(blocks)

        assert m == 5.0, "Max should be 5.0 (from second block)"

        # Verify l is correct
        all_values = np.concatenate(blocks)
        expected_l = np.sum(np.exp(all_values - 5.0))
        np.testing.assert_allclose(l, expected_l, rtol=1e-5)

    def test_matches_standard(self):
        """Online stats should enable correct softmax computation."""
        np.random.seed(42)
        x = np.random.randn(100)
        blocks = [x[i:i+10] for i in range(0, 100, 10)]

        m, l = online_softmax_stats(blocks)

        # Compute softmax using online stats
        online_softmax_result = np.exp(x - m) / l

        # Compare with standard
        standard_result = safe_softmax(x)

        np.testing.assert_allclose(online_softmax_result, standard_result, rtol=1e-5)


class TestOnlineSoftmax:
    """Tests for full online softmax computation."""

    def test_matches_standard_simple(self):
        """Online softmax should match standard softmax."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        blocks = [x[:2], x[2:4], x[4:]]

        online_result = online_softmax(blocks)
        standard_result = safe_softmax(x)

        np.testing.assert_allclose(online_result, standard_result, rtol=1e-5)

    def test_matches_standard_random(self):
        """Online softmax should match for random input."""
        np.random.seed(123)
        x = np.random.randn(100)
        blocks = [x[i:i+10] for i in range(0, 100, 10)]

        online_result = online_softmax(blocks)
        standard_result = safe_softmax(x)

        np.testing.assert_allclose(online_result, standard_result, rtol=1e-5)

    def test_single_block(self):
        """Should work with single block."""
        x = np.array([1.0, 2.0, 3.0])
        blocks = [x]

        online_result = online_softmax(blocks)
        standard_result = safe_softmax(x)

        np.testing.assert_allclose(online_result, standard_result, rtol=1e-5)

    def test_many_small_blocks(self):
        """Should work with many small blocks."""
        np.random.seed(456)
        x = np.random.randn(100)
        blocks = [x[i:i+5] for i in range(0, 100, 5)]

        online_result = online_softmax(blocks)
        standard_result = safe_softmax(x)

        np.testing.assert_allclose(online_result, standard_result, rtol=1e-5)

    def test_sums_to_one(self):
        """Online softmax output should sum to 1."""
        np.random.seed(789)
        x = np.random.randn(50)
        blocks = [x[i:i+10] for i in range(0, 50, 10)]

        result = online_softmax(blocks)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)


class TestOnlineAttentionAccumulator:
    """Tests for online attention output accumulation."""

    def test_matches_standard_attention(self):
        """Online attention should match standard attention."""
        np.random.seed(42)
        d_k, d_v = 16, 16
        seq_len = 32
        block_size = 8

        Q_row = np.random.randn(d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_v).astype(np.float32)

        # Split into blocks
        K_blocks = [K[i:i+block_size] for i in range(0, seq_len, block_size)]
        V_blocks = [V[i:i+block_size] for i in range(0, seq_len, block_size)]

        # Online attention
        online_output = online_attention_accumulator(Q_row, K_blocks, V_blocks, d_k)

        # Standard attention
        scores = Q_row @ K.T / np.sqrt(d_k)
        attn_weights = safe_softmax(scores)
        standard_output = attn_weights @ V

        np.testing.assert_allclose(online_output, standard_output, rtol=1e-4)

    def test_output_shape(self):
        """Output should have shape (d_v,)."""
        d_k, d_v = 8, 16
        Q_row = np.random.randn(d_k)
        K_blocks = [np.random.randn(4, d_k) for _ in range(3)]
        V_blocks = [np.random.randn(4, d_v) for _ in range(3)]

        output = online_attention_accumulator(Q_row, K_blocks, V_blocks, d_k)

        assert output.shape == (d_v,)

    def test_single_block(self):
        """Should work with single block."""
        d_k, d_v = 8, 8
        Q_row = np.random.randn(d_k)
        K_blocks = [np.random.randn(4, d_k)]
        V_blocks = [np.random.randn(4, d_v)]

        output = online_attention_accumulator(Q_row, K_blocks, V_blocks, d_k)

        # Standard
        scores = Q_row @ K_blocks[0].T / np.sqrt(d_k)
        attn_weights = safe_softmax(scores)
        expected = attn_weights @ V_blocks[0]

        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_many_blocks(self):
        """Should work with many blocks."""
        np.random.seed(123)
        d_k, d_v = 16, 16
        seq_len = 128
        block_size = 8

        Q_row = np.random.randn(d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_v).astype(np.float32)

        K_blocks = [K[i:i+block_size] for i in range(0, seq_len, block_size)]
        V_blocks = [V[i:i+block_size] for i in range(0, seq_len, block_size)]

        online_output = online_attention_accumulator(Q_row, K_blocks, V_blocks, d_k)

        # Standard
        scores = Q_row @ K.T / np.sqrt(d_k)
        attn_weights = safe_softmax(scores)
        standard_output = attn_weights @ V

        np.testing.assert_allclose(online_output, standard_output, rtol=1e-4)


class TestCompareOnlineVsStandard:
    """Tests for comparison function."""

    def test_outputs_match(self):
        """Online and standard outputs should match."""
        np.random.seed(42)
        seq_len, d_k = 64, 16

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        online_out, standard_out, max_diff = compare_online_vs_standard(Q, K, V)

        assert max_diff < 1e-4, f"Max diff too large: {max_diff}"
        np.testing.assert_allclose(online_out, standard_out, rtol=1e-4)

    def test_returns_correct_types(self):
        """Should return correct types."""
        Q = np.random.randn(16, 8).astype(np.float32)
        K = np.random.randn(16, 8).astype(np.float32)
        V = np.random.randn(16, 8).astype(np.float32)

        online_out, standard_out, max_diff = compare_online_vs_standard(Q, K, V)

        assert isinstance(online_out, np.ndarray)
        assert isinstance(standard_out, np.ndarray)
        assert isinstance(max_diff, float)


class TestNumericalStability:
    """Tests for numerical stability of online softmax."""

    def test_large_values(self):
        """Should handle large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0, 1003.0])
        blocks = [x[:2], x[2:]]

        result = online_softmax(blocks)

        assert not np.any(np.isnan(result)), "Should not produce NaN"
        assert not np.any(np.isinf(result)), "Should not produce Inf"
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_very_different_scales(self):
        """Should handle values with very different scales."""
        x = np.array([-100.0, 0.0, 100.0])
        blocks = [x[:1], x[1:2], x[2:]]

        result = online_softmax(blocks)

        # exp(-100) is tiny, exp(100) dominates
        # So result should be approximately [0, 0, 1]
        assert result[2] > 0.99, "Largest value should dominate"
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_attention_stability(self):
        """Online attention should be stable with large score values."""
        np.random.seed(42)
        d_k = 64  # Large d_k means larger dot products

        Q_row = np.random.randn(d_k).astype(np.float32) * 10  # Scale up
        K = np.random.randn(32, d_k).astype(np.float32) * 10
        V = np.random.randn(32, d_k).astype(np.float32)

        K_blocks = [K[:16], K[16:]]
        V_blocks = [V[:16], V[16:]]

        output = online_attention_accumulator(Q_row, K_blocks, V_blocks, d_k)

        assert not np.any(np.isnan(output)), "Should not produce NaN"
        assert not np.any(np.isinf(output)), "Should not produce Inf"
