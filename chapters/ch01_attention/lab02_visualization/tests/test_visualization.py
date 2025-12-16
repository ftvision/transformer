"""Tests for Lab 02: Attention Visualization."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization import (
    create_attention_heatmap,
    compute_attention_entropy,
    find_top_k_attended,
    compute_attention_sparsity,
    compare_attention_patterns,
)


class TestAttentionHeatmap:
    """Tests for create_attention_heatmap function."""

    def test_basic_heatmap(self):
        """Basic heatmap creation with matching dimensions."""
        weights = np.array([[0.7, 0.3], [0.4, 0.6]])
        query_labels = ["Q1", "Q2"]
        key_labels = ["K1", "K2"]

        heatmap = create_attention_heatmap(weights, query_labels, key_labels)

        assert "weights" in heatmap
        assert "query_labels" in heatmap
        assert "key_labels" in heatmap
        assert "shape" in heatmap

    def test_heatmap_preserves_weights(self):
        """Heatmap should preserve the original weights."""
        weights = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        heatmap = create_attention_heatmap(
            weights,
            ["Q1", "Q2"],
            ["K1", "K2", "K3"]
        )

        np.testing.assert_array_equal(heatmap["weights"], weights)

    def test_heatmap_preserves_labels(self):
        """Heatmap should preserve the labels."""
        weights = np.array([[0.5, 0.5], [0.3, 0.7]])
        query_labels = ["The", "cat"]
        key_labels = ["The", "cat"]

        heatmap = create_attention_heatmap(weights, query_labels, key_labels)

        assert heatmap["query_labels"] == query_labels
        assert heatmap["key_labels"] == key_labels

    def test_heatmap_shape(self):
        """Heatmap should record correct shape."""
        weights = np.random.rand(4, 6)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        heatmap = create_attention_heatmap(
            weights,
            ["Q" + str(i) for i in range(4)],
            ["K" + str(i) for i in range(6)]
        )

        assert heatmap["shape"] == (4, 6)

    def test_heatmap_dimension_mismatch_query(self):
        """Should raise error when query labels don't match."""
        weights = np.array([[0.5, 0.5], [0.3, 0.7]])

        with pytest.raises(ValueError):
            create_attention_heatmap(
                weights,
                ["Q1"],  # Wrong: should be 2 labels
                ["K1", "K2"]
            )

    def test_heatmap_dimension_mismatch_key(self):
        """Should raise error when key labels don't match."""
        weights = np.array([[0.5, 0.5], [0.3, 0.7]])

        with pytest.raises(ValueError):
            create_attention_heatmap(
                weights,
                ["Q1", "Q2"],
                ["K1"]  # Wrong: should be 2 labels
            )


class TestAttentionEntropy:
    """Tests for compute_attention_entropy function."""

    def test_peaked_attention_low_entropy(self):
        """Peaked attention (all weight on one position) should have entropy 0."""
        peaked = np.array([[1.0, 0.0, 0.0, 0.0]])
        entropy = compute_attention_entropy(peaked)

        np.testing.assert_allclose(entropy, [0.0], atol=1e-6)

    def test_uniform_attention_high_entropy(self):
        """Uniform attention should have maximum entropy."""
        n = 4
        uniform = np.array([[1/n] * n])
        entropy = compute_attention_entropy(uniform)

        # Max entropy for n categories is log(n)
        expected_entropy = np.log(n)
        np.testing.assert_allclose(entropy, [expected_entropy], rtol=1e-5)

    def test_entropy_shape_2d(self):
        """Entropy should return one value per query (row)."""
        weights = np.random.rand(5, 8)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        entropy = compute_attention_entropy(weights)

        assert entropy.shape == (5,)

    def test_entropy_shape_3d(self):
        """Entropy should handle batched input."""
        weights = np.random.rand(2, 5, 8)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        entropy = compute_attention_entropy(weights)

        assert entropy.shape == (2, 5)

    def test_entropy_comparison(self):
        """More diffuse attention should have higher entropy."""
        peaked = np.array([[0.9, 0.05, 0.05]])
        diffuse = np.array([[0.4, 0.3, 0.3]])

        entropy_peaked = compute_attention_entropy(peaked)[0]
        entropy_diffuse = compute_attention_entropy(diffuse)[0]

        assert entropy_diffuse > entropy_peaked

    def test_entropy_non_negative(self):
        """Entropy should always be non-negative."""
        weights = np.random.rand(10, 20)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        entropy = compute_attention_entropy(weights)

        assert np.all(entropy >= 0)


class TestTopKAttended:
    """Tests for find_top_k_attended function."""

    def test_top_k_basic(self):
        """Find top attended positions."""
        weights = np.array([
            [0.1, 0.7, 0.1, 0.1],
            [0.4, 0.3, 0.2, 0.1]
        ])

        top_k = find_top_k_attended(weights, query_idx=0, k=1)

        assert len(top_k) == 1
        assert top_k[0] == 1  # Position 1 has highest weight (0.7)

    def test_top_k_order(self):
        """Results should be sorted by weight (highest first)."""
        weights = np.array([[0.1, 0.3, 0.5, 0.1]])

        top_k = find_top_k_attended(weights, query_idx=0, k=3)

        assert len(top_k) == 3
        assert top_k[0] == 2  # 0.5 (highest)
        assert top_k[1] == 1  # 0.3 (second)
        # Third could be 0 or 3 (both 0.1)

    def test_top_k_returns_array(self):
        """Should return numpy array."""
        weights = np.array([[0.5, 0.3, 0.2]])

        top_k = find_top_k_attended(weights, query_idx=0, k=2)

        assert isinstance(top_k, np.ndarray)

    def test_top_k_different_queries(self):
        """Different queries should have different top positions."""
        weights = np.array([
            [0.8, 0.1, 0.1],  # Query 0 attends to position 0
            [0.1, 0.8, 0.1],  # Query 1 attends to position 1
            [0.1, 0.1, 0.8],  # Query 2 attends to position 2
        ])

        assert find_top_k_attended(weights, query_idx=0, k=1)[0] == 0
        assert find_top_k_attended(weights, query_idx=1, k=1)[0] == 1
        assert find_top_k_attended(weights, query_idx=2, k=1)[0] == 2

    def test_top_k_with_k_equals_seq_len(self):
        """k equal to sequence length should return all positions."""
        weights = np.array([[0.4, 0.3, 0.2, 0.1]])

        top_k = find_top_k_attended(weights, query_idx=0, k=4)

        assert len(top_k) == 4
        assert set(top_k) == {0, 1, 2, 3}


class TestAttentionSparsity:
    """Tests for compute_attention_sparsity function."""

    def test_fully_sparse(self):
        """All weights below threshold should give sparsity 1.0."""
        weights = np.array([[0.01, 0.01, 0.98]])  # But we measure < threshold

        # If threshold is 0.5, then 2/3 weights are below
        sparsity = compute_attention_sparsity(weights, threshold=0.5)

        np.testing.assert_allclose(sparsity, 2/3, rtol=1e-5)

    def test_no_sparsity(self):
        """All weights above threshold should give sparsity 0.0."""
        weights = np.array([[0.5, 0.5]])

        sparsity = compute_attention_sparsity(weights, threshold=0.1)

        assert sparsity == 0.0

    def test_sparsity_range(self):
        """Sparsity should be between 0 and 1."""
        weights = np.random.rand(10, 20)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        sparsity = compute_attention_sparsity(weights, threshold=0.1)

        assert 0.0 <= sparsity <= 1.0

    def test_peaked_attention_high_sparsity(self):
        """Peaked attention should have high sparsity."""
        peaked = np.array([
            [0.95, 0.02, 0.02, 0.01],
            [0.01, 0.96, 0.02, 0.01]
        ])

        sparsity = compute_attention_sparsity(peaked, threshold=0.1)

        # 6 out of 8 weights are below 0.1
        np.testing.assert_allclose(sparsity, 6/8, rtol=1e-5)

    def test_uniform_attention_low_sparsity(self):
        """Uniform attention should have low sparsity for low threshold."""
        uniform = np.ones((3, 4)) / 4  # All weights = 0.25

        sparsity = compute_attention_sparsity(uniform, threshold=0.1)

        assert sparsity == 0.0  # No weights below 0.1


class TestCompareAttentionPatterns:
    """Tests for compare_attention_patterns function."""

    def test_identical_patterns(self):
        """Identical patterns should have perfect similarity."""
        w = np.array([[0.5, 0.5], [0.3, 0.7]])

        result = compare_attention_patterns(w, w)

        np.testing.assert_allclose(result["cosine_similarity"], 1.0, rtol=1e-5)
        np.testing.assert_allclose(result["mse"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["max_diff"], 0.0, atol=1e-10)

    def test_different_patterns(self):
        """Different patterns should have lower similarity."""
        w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        w2 = np.array([[0.0, 1.0], [1.0, 0.0]])

        result = compare_attention_patterns(w1, w2)

        # These are orthogonal patterns
        assert result["cosine_similarity"] < 0.5
        assert result["mse"] > 0
        assert result["max_diff"] > 0

    def test_comparison_returns_all_metrics(self):
        """Result should contain all required metrics."""
        w1 = np.random.rand(4, 4)
        w1 = w1 / w1.sum(axis=-1, keepdims=True)
        w2 = np.random.rand(4, 4)
        w2 = w2 / w2.sum(axis=-1, keepdims=True)

        result = compare_attention_patterns(w1, w2)

        assert "cosine_similarity" in result
        assert "mse" in result
        assert "max_diff" in result

    def test_mse_calculation(self):
        """MSE should be correctly calculated."""
        w1 = np.array([[0.6, 0.4]])
        w2 = np.array([[0.4, 0.6]])

        result = compare_attention_patterns(w1, w2)

        # MSE = mean((0.6-0.4)^2 + (0.4-0.6)^2) = mean(0.04 + 0.04) = 0.04
        expected_mse = 0.04
        np.testing.assert_allclose(result["mse"], expected_mse, rtol=1e-5)

    def test_max_diff_calculation(self):
        """Max diff should be the largest absolute difference."""
        w1 = np.array([[0.7, 0.3], [0.5, 0.5]])
        w2 = np.array([[0.4, 0.6], [0.5, 0.5]])

        result = compare_attention_patterns(w1, w2)

        # Max diff is |0.7 - 0.4| = 0.3 or |0.3 - 0.6| = 0.3
        np.testing.assert_allclose(result["max_diff"], 0.3, rtol=1e-5)
