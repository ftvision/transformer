"""Tests for shared utilities."""

import numpy as np
import pytest

from shared.utils import softmax, relu, gelu, layer_norm


class TestSoftmax:
    def test_softmax_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)

        # Should sum to 1
        assert np.isclose(result.sum(), 1.0)

        # Higher values should have higher probabilities
        assert result[2] > result[1] > result[0]

    def test_softmax_numerical_stability(self):
        # Large values that would overflow without stability trick
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)

        # Should still sum to 1 and not be NaN
        assert np.isclose(result.sum(), 1.0)
        assert not np.any(np.isnan(result))

    def test_softmax_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=-1)

        # Each row should sum to 1
        assert np.allclose(result.sum(axis=-1), [1.0, 1.0])

    def test_softmax_uniform(self):
        # All equal values should give uniform distribution
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = softmax(x)

        assert np.allclose(result, [0.25, 0.25, 0.25, 0.25])


class TestRelu:
    def test_relu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, x)

    def test_relu_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0, 2.0])


class TestGelu:
    def test_gelu_zero(self):
        # GELU(0) = 0
        assert np.isclose(gelu(np.array([0.0]))[0], 0.0)

    def test_gelu_positive(self):
        # For large positive x, GELU(x) ≈ x
        x = np.array([5.0])
        assert np.isclose(gelu(x)[0], 5.0, atol=0.01)

    def test_gelu_negative(self):
        # For large negative x, GELU(x) ≈ 0
        x = np.array([-5.0])
        assert np.isclose(gelu(x)[0], 0.0, atol=0.01)

    def test_gelu_shape(self):
        x = np.random.randn(3, 4, 5)
        result = gelu(x)
        assert result.shape == x.shape


class TestLayerNorm:
    def test_layer_norm_basic(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        beta = np.zeros(4)

        result = layer_norm(x, gamma, beta)

        # Mean should be ~0
        assert np.isclose(result.mean(), 0.0, atol=1e-5)

        # Std should be ~1
        assert np.isclose(result.std(), 1.0, atol=0.1)

    def test_layer_norm_with_params(self):
        x = np.array([[0.0, 0.0, 0.0, 0.0]])
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0])

        result = layer_norm(x, gamma, beta)

        # Should be shifted by beta
        np.testing.assert_array_almost_equal(result, [[1.0, 1.0, 1.0, 1.0]])

    def test_layer_norm_batch(self):
        x = np.random.randn(2, 3, 4)
        gamma = np.ones(4)
        beta = np.zeros(4)

        result = layer_norm(x, gamma, beta)

        # Shape should be preserved
        assert result.shape == x.shape
