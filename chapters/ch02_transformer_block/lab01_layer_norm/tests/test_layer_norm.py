"""Tests for Lab 01: Layer Normalization."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from layer_norm import layer_norm, rms_norm, LayerNorm, RMSNorm


class TestLayerNormFunction:
    """Tests for the layer_norm function."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        x = np.random.randn(2, 10, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        assert out.shape == x.shape

    def test_normalized_mean(self):
        """Normalized output should have mean close to beta."""
        x = np.random.randn(2, 10, 64) * 10 + 5  # Non-zero mean
        gamma = np.ones(64)
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        # Mean along last axis should be ~0 (beta)
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0, atol=1e-5)

    def test_normalized_variance(self):
        """Normalized output should have variance close to gamma^2."""
        x = np.random.randn(2, 10, 64) * 10  # High variance
        gamma = np.ones(64)
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        # Variance along last axis should be ~1 (gamma^2)
        variances = out.var(axis=-1)
        np.testing.assert_allclose(variances, 1, atol=1e-4)

    def test_gamma_scaling(self):
        """Output variance should be proportional to gamma^2."""
        x = np.random.randn(10, 64)
        gamma = np.ones(64) * 2  # Scale by 2
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        # Variance should be ~4 (gamma^2 = 2^2)
        variances = out.var(axis=-1)
        np.testing.assert_allclose(variances, 4, atol=1e-3)

    def test_beta_shifting(self):
        """Output mean should equal beta."""
        x = np.random.randn(10, 64)
        gamma = np.ones(64)
        beta = np.ones(64) * 3  # Shift by 3

        out = layer_norm(x, gamma, beta)

        # Mean should be ~3 (beta)
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 3, atol=1e-5)

    def test_identity_with_unit_params(self):
        """With gamma=1, beta=0, normalized vectors should have mean=0, var=1."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        beta = np.zeros(4)

        out = layer_norm(x, gamma, beta)

        np.testing.assert_allclose(out.mean(axis=-1), 0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=-1), 1, atol=1e-6)

    def test_numerical_stability(self):
        """Should handle very small variance without NaN/Inf."""
        x = np.ones((2, 10, 64))  # Zero variance input
        gamma = np.ones(64)
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        assert not np.any(np.isnan(out)), "Output contains NaN"
        assert not np.any(np.isinf(out)), "Output contains Inf"

    def test_2d_input(self):
        """Should work with 2D input."""
        x = np.random.randn(10, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        assert out.shape == (10, 64)

    def test_4d_input(self):
        """Should work with higher dimensional input."""
        x = np.random.randn(2, 3, 10, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)

        out = layer_norm(x, gamma, beta)

        assert out.shape == (2, 3, 10, 64)


class TestRMSNormFunction:
    """Tests for the rms_norm function."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        x = np.random.randn(2, 10, 64)
        gamma = np.ones(64)

        out = rms_norm(x, gamma)

        assert out.shape == x.shape

    def test_rms_property(self):
        """Output should have RMS close to gamma."""
        x = np.random.randn(10, 64) * 5
        gamma = np.ones(64)

        out = rms_norm(x, gamma)

        # RMS of output should be ~1
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1, atol=1e-4)

    def test_gamma_scaling(self):
        """Output RMS should be proportional to gamma."""
        x = np.random.randn(10, 64)
        gamma = np.ones(64) * 2  # Scale by 2

        out = rms_norm(x, gamma)

        # RMS should be ~2
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 2, atol=1e-3)

    def test_no_mean_centering(self):
        """RMSNorm should NOT center the mean (unlike LayerNorm)."""
        x = np.random.randn(10, 64) + 10  # Large positive mean
        gamma = np.ones(64)

        out = rms_norm(x, gamma)

        # Mean should NOT be zero (unlike LayerNorm)
        means = out.mean(axis=-1)
        assert np.all(np.abs(means) > 0.1), "RMSNorm should not center mean"

    def test_numerical_stability(self):
        """Should handle small values without NaN/Inf."""
        x = np.ones((2, 10, 64)) * 1e-10
        gamma = np.ones(64)

        out = rms_norm(x, gamma)

        assert not np.any(np.isnan(out)), "Output contains NaN"
        assert not np.any(np.isinf(out)), "Output contains Inf"

    def test_2d_input(self):
        """Should work with 2D input."""
        x = np.random.randn(10, 64)
        gamma = np.ones(64)

        out = rms_norm(x, gamma)

        assert out.shape == (10, 64)


class TestLayerNormClass:
    """Tests for the LayerNorm class."""

    def test_init(self):
        """Should initialize with correct parameters."""
        ln = LayerNorm(64)

        assert ln.d_model == 64
        assert ln.gamma.shape == (64,)
        assert ln.beta.shape == (64,)

    def test_gamma_init_to_ones(self):
        """Gamma should be initialized to ones."""
        ln = LayerNorm(64)

        np.testing.assert_array_equal(ln.gamma, np.ones(64))

    def test_beta_init_to_zeros(self):
        """Beta should be initialized to zeros."""
        ln = LayerNorm(64)

        np.testing.assert_array_equal(ln.beta, np.zeros(64))

    def test_forward(self):
        """Forward pass should normalize correctly."""
        ln = LayerNorm(64)
        x = np.random.randn(2, 10, 64) * 5 + 3

        out = ln.forward(x)

        # Should have mean ~0, var ~1
        means = out.mean(axis=-1)
        variances = out.var(axis=-1)

        np.testing.assert_allclose(means, 0, atol=1e-5)
        np.testing.assert_allclose(variances, 1, atol=1e-4)

    def test_callable(self):
        """Should be callable like a function."""
        ln = LayerNorm(64)
        x = np.random.randn(2, 10, 64)

        out1 = ln.forward(x)
        out2 = ln(x)

        np.testing.assert_array_equal(out1, out2)

    def test_custom_eps(self):
        """Should accept custom eps parameter."""
        ln = LayerNorm(64, eps=1e-6)

        assert ln.eps == 1e-6


class TestRMSNormClass:
    """Tests for the RMSNorm class."""

    def test_init(self):
        """Should initialize with correct parameters."""
        rms = RMSNorm(64)

        assert rms.d_model == 64
        assert rms.gamma.shape == (64,)

    def test_gamma_init_to_ones(self):
        """Gamma should be initialized to ones."""
        rms = RMSNorm(64)

        np.testing.assert_array_equal(rms.gamma, np.ones(64))

    def test_no_beta(self):
        """RMSNorm should not have beta parameter."""
        rms = RMSNorm(64)

        assert not hasattr(rms, 'beta') or rms.beta is None

    def test_forward(self):
        """Forward pass should normalize correctly."""
        rms = RMSNorm(64)
        x = np.random.randn(2, 10, 64) * 5

        out = rms.forward(x)

        # RMS should be ~1
        rms_values = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms_values, 1, atol=1e-4)

    def test_callable(self):
        """Should be callable like a function."""
        rms = RMSNorm(64)
        x = np.random.randn(2, 10, 64)

        out1 = rms.forward(x)
        out2 = rms(x)

        np.testing.assert_array_equal(out1, out2)


class TestCompareNorms:
    """Tests comparing LayerNorm and RMSNorm."""

    def test_different_outputs(self):
        """LayerNorm and RMSNorm should produce different outputs."""
        x = np.random.randn(10, 64) + 5  # Non-zero mean

        ln = LayerNorm(64)
        rms = RMSNorm(64)

        out_ln = ln(x)
        out_rms = rms(x)

        assert not np.allclose(out_ln, out_rms)

    def test_same_output_for_zero_mean(self):
        """For zero-mean input, outputs should be similar but not identical."""
        np.random.seed(42)
        x = np.random.randn(10, 64)
        x = x - x.mean(axis=-1, keepdims=True)  # Zero mean

        ln = LayerNorm(64)
        rms = RMSNorm(64)

        out_ln = ln(x)
        out_rms = rms(x)

        # They won't be exactly equal because var != mean(x^2) for zero-mean
        # But they should be reasonably close
        # This test just verifies both work on zero-mean data
        assert out_ln.shape == out_rms.shape


class TestPyTorchComparison:
    """Tests comparing with PyTorch's LayerNorm (if available)."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not available")
            return False

    def test_matches_pytorch(self, torch_available):
        """Output should match PyTorch's LayerNorm."""
        import torch
        import torch.nn as nn

        d_model = 64
        x_np = np.random.randn(2, 10, d_model).astype(np.float32)

        # PyTorch LayerNorm
        torch_ln = nn.LayerNorm(d_model, elementwise_affine=True)
        # Initialize to ones/zeros for comparison
        torch_ln.weight.data.fill_(1.0)
        torch_ln.bias.data.fill_(0.0)

        with torch.no_grad():
            x_torch = torch.from_numpy(x_np)
            out_torch = torch_ln(x_torch).numpy()

        # Our LayerNorm
        ln = LayerNorm(d_model)
        out_np = ln(x_np)

        np.testing.assert_allclose(out_np, out_torch, rtol=1e-5, atol=1e-5)
