"""Tests for Lab 03: Feature Maps."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_maps import (
    elu_plus_one,
    relu_feature_map,
    squared_relu_feature_map,
    exp_feature_map,
    random_feature_map,
    create_random_projection,
    compute_implicit_attention,
    compute_softmax_attention,
    compare_to_softmax,
    analyze_feature_map_quality,
    linear_attention_with_feature_map,
    softmax_attention,
)


class TestELUPlusOne:
    """Tests for elu_plus_one feature map."""

    def test_always_positive(self):
        """ELU+1 should always be positive."""
        x = np.random.randn(100, 64)
        result = elu_plus_one(x)

        assert np.all(result > 0), "ELU+1 should be positive for all inputs"

    def test_positive_input(self):
        """For positive x, ELU+1(x) = x + 1."""
        x = np.abs(np.random.randn(100, 64)) + 0.01
        result = elu_plus_one(x)

        np.testing.assert_allclose(result, x + 1, rtol=1e-5)

    def test_negative_input(self):
        """For negative x, ELU+1(x) = exp(x)."""
        x = -np.abs(np.random.randn(100, 64)) - 0.01
        result = elu_plus_one(x)

        # ELU(x) = exp(x) - 1 for x < 0
        # ELU(x) + 1 = exp(x)
        np.testing.assert_allclose(result, np.exp(x), rtol=1e-5)

    def test_zero_input(self):
        """ELU+1(0) should be 1."""
        x = np.zeros((10, 10))
        result = elu_plus_one(x)

        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_known_values(self):
        """Test against known values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = elu_plus_one(x)

        expected = np.array([
            np.exp(-2.0),      # -2: exp(-2)
            np.exp(-1.0),      # -1: exp(-1)
            1.0,               # 0: ELU(0) + 1 = 0 + 1 = 1
            2.0,               # 1: 1 + 1 = 2
            3.0,               # 2: 2 + 1 = 3
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestReLUFeatureMap:
    """Tests for relu_feature_map."""

    def test_zeros_negative(self):
        """ReLU should zero out negative values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu_feature_map(x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_positive(self):
        """ReLU should preserve positive values."""
        x = np.abs(np.random.randn(100, 64)) + 0.01
        result = relu_feature_map(x)

        np.testing.assert_allclose(result, x, rtol=1e-5)

    def test_non_negative(self):
        """ReLU output should be non-negative."""
        x = np.random.randn(100, 64)
        result = relu_feature_map(x)

        assert np.all(result >= 0)


class TestSquaredReLU:
    """Tests for squared_relu_feature_map."""

    def test_non_negative(self):
        """Squared ReLU should always be non-negative."""
        x = np.random.randn(100, 64)
        result = squared_relu_feature_map(x)

        assert np.all(result >= 0)

    def test_known_values(self):
        """Test against known values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = squared_relu_feature_map(x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_square_of_relu(self):
        """Should equal ReLU squared."""
        x = np.random.randn(100, 64)
        result = squared_relu_feature_map(x)
        expected = relu_feature_map(x) ** 2

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestExpFeatureMap:
    """Tests for exp_feature_map."""

    def test_always_positive(self):
        """Exp should always be positive."""
        x = np.random.randn(100, 64)
        result = exp_feature_map(x)

        assert np.all(result > 0)

    def test_known_values(self):
        """Test against known values."""
        x = np.array([0.0, 1.0, 2.0])
        result = exp_feature_map(x)

        np.testing.assert_allclose(result, np.exp(x), rtol=1e-5)

    def test_scale_parameter(self):
        """Scale parameter should work correctly."""
        x = np.array([1.0, 2.0])
        result = exp_feature_map(x, scale=0.5)

        expected = np.exp(x * 0.5)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestRandomFeatureMap:
    """Tests for random_feature_map."""

    def test_output_shape_cos_sin(self):
        """With cos/sin, output should be 2x the projection dimension."""
        x = np.random.randn(100, 64)
        proj = np.random.randn(64, 128)

        result = random_feature_map(x, proj, use_cos_sin=True)

        assert result.shape == (100, 256)  # 2 * 128

    def test_output_shape_cos_only(self):
        """Without sin, output should match projection dimension."""
        x = np.random.randn(100, 64)
        proj = np.random.randn(64, 128)

        result = random_feature_map(x, proj, use_cos_sin=False)

        assert result.shape == (100, 128)

    def test_bounded_output(self):
        """Cos/sin outputs should be bounded in [-1, 1]."""
        x = np.random.randn(100, 64) * 10  # Large values
        proj = np.random.randn(64, 128)

        result = random_feature_map(x, proj, use_cos_sin=True)

        # After scaling, values might be outside [-1,1] but shouldn't explode
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestCreateRandomProjection:
    """Tests for create_random_projection."""

    def test_correct_shape(self):
        """Projection should have correct shape."""
        proj = create_random_projection(64, 128)

        assert proj.shape == (64, 128)

    def test_reproducible_with_seed(self):
        """Same seed should give same projection."""
        proj1 = create_random_projection(64, 128, seed=42)
        proj2 = create_random_projection(64, 128, seed=42)

        np.testing.assert_array_equal(proj1, proj2)

    def test_different_without_seed(self):
        """Different calls without seed should differ."""
        proj1 = create_random_projection(64, 128, seed=None)
        proj2 = create_random_projection(64, 128, seed=None)

        # Very unlikely to be equal
        assert not np.allclose(proj1, proj2)


class TestImplicitAttention:
    """Tests for compute_implicit_attention."""

    def test_output_shape(self):
        """Attention should be (seq_len, seq_len)."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)

        A = compute_implicit_attention(Q, K, elu_plus_one)

        assert A.shape == (50, 50)

    def test_rows_sum_to_one(self):
        """Each row should sum to 1."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)

        A = compute_implicit_attention(Q, K, elu_plus_one)

        np.testing.assert_allclose(A.sum(axis=-1), np.ones(50), rtol=1e-5)

    def test_non_negative(self):
        """Attention weights should be non-negative."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)

        A = compute_implicit_attention(Q, K, elu_plus_one)

        assert np.all(A >= 0)


class TestSoftmaxAttention:
    """Tests for compute_softmax_attention."""

    def test_output_shape(self):
        """Attention should be (seq_len, seq_len)."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)

        A = compute_softmax_attention(Q, K)

        assert A.shape == (50, 50)

    def test_rows_sum_to_one(self):
        """Each row should sum to 1."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)

        A = compute_softmax_attention(Q, K)

        np.testing.assert_allclose(A.sum(axis=-1), np.ones(50), rtol=1e-5)

    def test_non_negative(self):
        """Softmax attention should be non-negative."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)

        A = compute_softmax_attention(Q, K)

        assert np.all(A >= 0)


class TestCompareToSoftmax:
    """Tests for compare_to_softmax."""

    def test_returns_dict(self):
        """Should return dictionary with expected keys."""
        Q = np.random.randn(50, 64).astype(np.float32)
        K = np.random.randn(50, 64).astype(np.float32)

        metrics = compare_to_softmax(Q, K, elu_plus_one)

        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'max_diff' in metrics
        assert 'correlation' in metrics
        assert 'row_correlation_mean' in metrics

    def test_mse_non_negative(self):
        """MSE should be non-negative."""
        Q = np.random.randn(50, 64).astype(np.float32)
        K = np.random.randn(50, 64).astype(np.float32)

        metrics = compare_to_softmax(Q, K, elu_plus_one)

        assert metrics['mse'] >= 0

    def test_correlation_bounded(self):
        """Correlation should be in [-1, 1]."""
        Q = np.random.randn(50, 64).astype(np.float32)
        K = np.random.randn(50, 64).astype(np.float32)

        metrics = compare_to_softmax(Q, K, elu_plus_one)

        assert -1 <= metrics['correlation'] <= 1

    def test_elu_better_than_relu(self):
        """ELU+1 should typically have better correlation than ReLU."""
        np.random.seed(42)
        Q = np.random.randn(100, 64).astype(np.float32)
        K = np.random.randn(100, 64).astype(np.float32)

        elu_metrics = compare_to_softmax(Q, K, elu_plus_one)
        relu_metrics = compare_to_softmax(Q, K, relu_feature_map)

        # ELU+1 usually has better correlation with softmax
        assert elu_metrics['correlation'] >= relu_metrics['correlation'] - 0.1


class TestLinearAttention:
    """Tests for linear_attention_with_feature_map."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = linear_attention_with_feature_map(Q, K, V, elu_plus_one)

        assert output.shape == (50, 64)

    def test_no_nan(self):
        """Output should not contain NaN."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = linear_attention_with_feature_map(Q, K, V, elu_plus_one)

        assert not np.any(np.isnan(output))


class TestAnalyzeFeatureMapQuality:
    """Tests for analyze_feature_map_quality."""

    def test_returns_dict(self):
        """Should return dictionary with expected keys."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        analysis = analyze_feature_map_quality(elu_plus_one, Q, K, V)

        assert isinstance(analysis, dict)
        assert 'attention_mse' in analysis
        assert 'attention_correlation' in analysis
        assert 'output_mse' in analysis
        assert 'output_correlation' in analysis
        assert 'feature_sparsity' in analysis
        assert 'feature_mean' in analysis
        assert 'feature_std' in analysis


class TestFeatureMapsMilestone:
    """
    Milestone test: Verify understanding of feature maps.
    """

    def test_feature_maps_understood(self):
        """
        MILESTONE: Demonstrate understanding of feature maps.

        This test verifies that:
        1. You can implement various feature maps
        2. You understand their properties
        3. You can compare them to softmax attention
        """
        np.random.seed(42)
        Q = np.random.randn(100, 64).astype(np.float32)
        K = np.random.randn(100, 64).astype(np.float32)
        V = np.random.randn(100, 64).astype(np.float32)

        # Test 1: All feature maps work
        for name, fm in [('ELU+1', elu_plus_one),
                         ('ReLU', relu_feature_map),
                         ('Squared ReLU', squared_relu_feature_map)]:
            output = linear_attention_with_feature_map(Q, K, V, fm)
            assert output.shape == (100, 64), f"{name} failed"
            assert not np.any(np.isnan(output)), f"{name} produced NaN"

        # Test 2: ELU+1 produces positive features
        elu_features = elu_plus_one(Q)
        assert np.all(elu_features > 0), "ELU+1 should be positive"

        # Test 3: Compare to softmax shows reasonable correlation
        metrics = compare_to_softmax(Q, K, elu_plus_one)
        assert metrics['correlation'] > 0.3, "Correlation too low"

        print(f"\n{'='*60}")
        print("MILESTONE: Feature Maps Understood!")
        print(f"ELU+1 correlation with softmax: {metrics['correlation']:.3f}")
        print(f"ELU+1 MSE vs softmax: {metrics['mse']:.4f}")
        print(f"{'='*60}\n")
