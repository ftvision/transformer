"""Tests for Lab 01: Dot-Product Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention import softmax, scaled_dot_product_attention


class TestSoftmax:
    """Tests for the softmax function."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1 along the specified axis."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0), "Softmax should sum to 1"

    def test_softmax_positive_outputs(self):
        """All softmax outputs should be positive."""
        x = np.array([-5.0, 0.0, 5.0])
        result = softmax(x)
        assert np.all(result > 0), "All softmax outputs should be positive"

    def test_softmax_preserves_order(self):
        """Higher inputs should give higher outputs."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result[2] > result[1] > result[0], "Softmax should preserve order"

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)

        assert not np.any(np.isnan(result)), "Should not produce NaN"
        assert not np.any(np.isinf(result)), "Should not produce Inf"
        assert np.isclose(result.sum(), 1.0), "Should still sum to 1"

    def test_softmax_2d_axis(self):
        """Softmax should work along specified axis for 2D input."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=-1)

        # Each row should sum to 1
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)

    def test_softmax_3d(self):
        """Softmax should work for 3D inputs (batched)."""
        x = np.random.randn(2, 3, 4)
        result = softmax(x, axis=-1)

        # Last dimension should sum to 1
        assert result.shape == x.shape
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((2, 3)), rtol=1e-5)

    def test_softmax_uniform_input(self):
        """Equal inputs should give uniform distribution."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = softmax(x)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_attention_output_shape(self):
        """Output shape should be (seq_len_q, d_v)."""
        Q = np.random.randn(4, 8)  # 4 queries, dim 8
        K = np.random.randn(6, 8)  # 6 keys, dim 8
        V = np.random.randn(6, 10)  # 6 values, dim 10

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
        assert weights.shape == (4, 6), f"Expected (4, 6), got {weights.shape}"

    def test_attention_weights_sum_to_one(self):
        """Each query's attention weights should sum to 1."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(6, 8)
        V = np.random.randn(6, 10)

        _, weights = scaled_dot_product_attention(Q, K, V)

        # Each row (query) should sum to 1
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), rtol=1e-5)

    def test_attention_weights_positive(self):
        """All attention weights should be positive."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(6, 8)
        V = np.random.randn(6, 10)

        _, weights = scaled_dot_product_attention(Q, K, V)

        assert np.all(weights >= 0), "Attention weights should be non-negative"

    def test_self_attention(self):
        """Self-attention: Q, K, V from same source, same seq_len."""
        seq_len, d_model = 5, 16
        X = np.random.randn(seq_len, d_model)

        # In self-attention, Q=K=V=X (before projection)
        output, weights = scaled_dot_product_attention(X, X, X)

        assert output.shape == (seq_len, d_model)
        assert weights.shape == (seq_len, seq_len)

    def test_scaling_effect(self):
        """Verify that scaling by sqrt(d_k) is applied."""
        # With high dimension, unscaled attention would have high variance scores
        # leading to sharper (more peaked) attention distributions
        np.random.seed(42)

        d_k = 64
        Q = np.random.randn(4, d_k)
        K = np.random.randn(6, d_k)
        V = np.random.randn(6, 10)

        _, weights = scaled_dot_product_attention(Q, K, V)

        # With proper scaling, weights shouldn't be too peaked
        # (not all mass on one position)
        max_weight = weights.max(axis=-1)
        assert np.all(max_weight < 0.99), \
            "Weights too peaked - scaling may not be applied correctly"

    def test_batched_attention(self):
        """Attention should work with batched inputs."""
        batch_size, seq_len, d_k, d_v = 2, 4, 8, 10
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_v)

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch_size, seq_len, d_v)
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_masking(self):
        """Masked positions should have zero attention weight."""
        Q = np.random.randn(3, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 10)

        # Mask out position 2 for all queries
        mask = np.zeros((3, 4), dtype=bool)
        mask[:, 2] = True  # Mask column 2

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Masked positions should have ~0 weight
        np.testing.assert_allclose(weights[:, 2], 0.0, atol=1e-6)

    def test_causal_mask(self):
        """Causal (triangular) mask for autoregressive attention."""
        seq_len, d_k = 4, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)

        # Causal mask: position i can only attend to positions <= i
        # True = masked out
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Upper triangle (future positions) should be ~0
        upper_triangle = np.triu(weights, k=1)
        np.testing.assert_allclose(upper_triangle, 0.0, atol=1e-6)

        # Rows should still sum to 1
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(seq_len), rtol=1e-5)


class TestAttentionCorrectness:
    """Tests comparing against known correct values."""

    def test_simple_attention(self):
        """Test with simple values where we can compute by hand."""
        # Simple case: 2 positions, 2 dimensions
        Q = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        K = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        V = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        output, weights = scaled_dot_product_attention(Q, K, V)

        # With identical Q, K, V and orthogonal vectors:
        # Q[0] @ K.T = [1/sqrt(2), 0] after scaling
        # Softmax gives higher weight to matching position

        # Check that position 0 attends more to position 0
        assert weights[0, 0] > weights[0, 1], \
            "Query 0 should attend more to Key 0"
        # Check that position 1 attends more to position 1
        assert weights[1, 1] > weights[1, 0], \
            "Query 1 should attend more to Key 1"

    def test_identity_attention(self):
        """When Q=K, each position should attend most to itself."""
        np.random.seed(123)
        seq_len, d_k = 5, 16

        X = np.random.randn(seq_len, d_k)
        # Make rows distinct
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)

        _, weights = scaled_dot_product_attention(X, X, X)

        # Diagonal should have highest values (self-attention peaks)
        for i in range(seq_len):
            assert weights[i, i] == weights[i].max(), \
                f"Position {i} should attend most to itself"
