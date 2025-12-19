"""Tests for Lab 03: Feed-Forward Network."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feed_forward import (
    relu,
    sigmoid,
    gelu,
    silu,
    FeedForward,
    SwiGLUFeedForward,
    count_parameters,
)


class TestReLU:
    """Tests for ReLU activation."""

    def test_positive_unchanged(self):
        """Positive values should be unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_zeroed(self):
        """Negative values should become zero."""
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu(x), [0, 0, 0])

    def test_mixed(self):
        """Mixed values should be handled correctly."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(relu(x), expected)


class TestSigmoid:
    """Tests for sigmoid activation."""

    def test_at_zero(self):
        """sigmoid(0) = 0.5."""
        np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5])

    def test_large_positive(self):
        """Large positive values should approach 1."""
        x = np.array([10.0, 100.0])
        result = sigmoid(x)
        assert np.all(result > 0.99)

    def test_large_negative(self):
        """Large negative values should approach 0."""
        x = np.array([-10.0, -100.0])
        result = sigmoid(x)
        assert np.all(result < 0.01)

    def test_bounded(self):
        """Output should be in (0, 1)."""
        x = np.random.randn(1000)
        result = sigmoid(x)
        assert np.all(result > 0) and np.all(result < 1)


class TestGELU:
    """Tests for GELU activation."""

    def test_at_zero(self):
        """GELU(0) = 0."""
        np.testing.assert_allclose(gelu(np.array([0.0])), [0.0], atol=1e-6)

    def test_positive_values(self):
        """Large positive values should be close to identity."""
        x = np.array([3.0])
        result = gelu(x)
        # GELU(x) ≈ x for large positive x
        assert result[0] > 2.9

    def test_negative_values(self):
        """Negative values should be small but not zero."""
        x = np.array([-1.0])
        result = gelu(x)
        # GELU(-1) ≈ -0.158
        np.testing.assert_allclose(result, [-0.158], atol=0.01)

    def test_smooth(self):
        """GELU should be smooth (no sharp corners like ReLU)."""
        # Check that derivative is continuous around 0
        x = np.linspace(-0.1, 0.1, 100)
        y = gelu(x)
        dy = np.diff(y) / np.diff(x)

        # Derivative should change smoothly
        ddy = np.diff(dy)
        assert np.all(np.abs(ddy) < 0.5)

    def test_approximation_accuracy(self):
        """Approximation should be close to exact GELU."""
        from scipy.special import erf

        def gelu_exact(x):
            return 0.5 * x * (1 + erf(x / np.sqrt(2)))

        x = np.linspace(-3, 3, 100)
        approx = gelu(x)
        exact = gelu_exact(x)

        np.testing.assert_allclose(approx, exact, atol=0.02)


class TestSiLU:
    """Tests for SiLU (Swish) activation."""

    def test_at_zero(self):
        """SiLU(0) = 0 * sigmoid(0) = 0."""
        np.testing.assert_allclose(silu(np.array([0.0])), [0.0], atol=1e-6)

    def test_positive_values(self):
        """Large positive values should be close to identity."""
        x = np.array([5.0])
        result = silu(x)
        # SiLU(x) ≈ x for large positive x (sigmoid ≈ 1)
        assert result[0] > 4.9

    def test_negative_values(self):
        """SiLU allows negative outputs (unlike ReLU)."""
        x = np.array([-1.0])
        result = silu(x)
        # SiLU(-1) = -1 * sigmoid(-1) ≈ -0.268
        np.testing.assert_allclose(result, [-0.268], atol=0.01)

    def test_formula(self):
        """Verify SiLU = x * sigmoid(x)."""
        x = np.random.randn(100)
        result = silu(x)
        expected = x * sigmoid(x)
        np.testing.assert_allclose(result, expected)


class TestFeedForward:
    """Tests for standard feed-forward network."""

    def test_init(self):
        """Should initialize with correct shapes."""
        ffn = FeedForward(512, 2048)

        assert ffn.d_model == 512
        assert ffn.d_ff == 2048
        assert ffn.W1.shape == (512, 2048)
        assert ffn.W2.shape == (2048, 512)

    def test_default_d_ff(self):
        """Default d_ff should be 4 * d_model."""
        ffn = FeedForward(512)

        assert ffn.d_ff == 2048  # 4 * 512

    def test_output_shape_2d(self):
        """Output shape should match input for 2D input."""
        ffn = FeedForward(64, 256)
        x = np.random.randn(10, 64)

        out = ffn(x)

        assert out.shape == (10, 64)

    def test_output_shape_3d(self):
        """Output shape should match input for 3D input."""
        ffn = FeedForward(64, 256)
        x = np.random.randn(2, 10, 64)

        out = ffn(x)

        assert out.shape == (2, 10, 64)

    def test_relu_activation(self):
        """Should work with ReLU activation."""
        ffn = FeedForward(64, 256, activation='relu')
        x = np.random.randn(2, 10, 64)

        out = ffn(x)

        assert out.shape == (2, 10, 64)

    def test_gelu_activation(self):
        """Should work with GELU activation."""
        ffn = FeedForward(64, 256, activation='gelu')
        x = np.random.randn(2, 10, 64)

        out = ffn(x)

        assert out.shape == (2, 10, 64)

    def test_no_bias(self):
        """Should work without bias."""
        ffn = FeedForward(64, 256, use_bias=False)

        assert ffn.b1 is None or not hasattr(ffn, 'b1')
        assert ffn.b2 is None or not hasattr(ffn, 'b2')

    def test_callable(self):
        """Should be callable like a function."""
        ffn = FeedForward(64, 256)
        x = np.random.randn(2, 10, 64)

        out1 = ffn.forward(x)
        out2 = ffn(x)

        np.testing.assert_array_equal(out1, out2)

    def test_deterministic(self):
        """Same input should give same output."""
        ffn = FeedForward(64, 256)
        x = np.random.randn(2, 10, 64)

        out1 = ffn(x)
        out2 = ffn(x)

        np.testing.assert_array_equal(out1, out2)

    def test_different_inputs(self):
        """Different inputs should give different outputs."""
        ffn = FeedForward(64, 256)

        x1 = np.random.randn(2, 10, 64)
        x2 = np.random.randn(2, 10, 64)

        out1 = ffn(x1)
        out2 = ffn(x2)

        assert not np.allclose(out1, out2)


class TestSwiGLUFeedForward:
    """Tests for SwiGLU feed-forward network."""

    def test_init(self):
        """Should initialize with correct shapes."""
        ffn = SwiGLUFeedForward(512)

        assert ffn.d_model == 512
        assert ffn.W1.shape[0] == 512
        assert ffn.W2.shape[0] == 512
        assert ffn.W3.shape[1] == 512

    def test_three_weights(self):
        """Should have W1, W2, W3 (no biases)."""
        ffn = SwiGLUFeedForward(512)

        assert hasattr(ffn, 'W1')
        assert hasattr(ffn, 'W2')
        assert hasattr(ffn, 'W3')

    def test_custom_d_ff(self):
        """Should accept custom d_ff."""
        ffn = SwiGLUFeedForward(512, d_ff=1024)

        assert ffn.d_ff == 1024
        assert ffn.W1.shape == (512, 1024)
        assert ffn.W2.shape == (512, 1024)
        assert ffn.W3.shape == (1024, 512)

    def test_output_shape_2d(self):
        """Output shape should match input for 2D input."""
        ffn = SwiGLUFeedForward(64)
        x = np.random.randn(10, 64)

        out = ffn(x)

        assert out.shape == (10, 64)

    def test_output_shape_3d(self):
        """Output shape should match input for 3D input."""
        ffn = SwiGLUFeedForward(64)
        x = np.random.randn(2, 10, 64)

        out = ffn(x)

        assert out.shape == (2, 10, 64)

    def test_callable(self):
        """Should be callable like a function."""
        ffn = SwiGLUFeedForward(64)
        x = np.random.randn(2, 10, 64)

        out1 = ffn.forward(x)
        out2 = ffn(x)

        np.testing.assert_array_equal(out1, out2)

    def test_gating_effect(self):
        """Gating should affect output (gate != all ones)."""
        ffn = SwiGLUFeedForward(64, d_ff=256)
        x = np.random.randn(2, 10, 64)

        # Get intermediate values
        gate = silu(x @ ffn.W1)
        value = x @ ffn.W2

        # Gate should have varying values (not all 1s)
        assert gate.std() > 0.1

        # Some elements should be significantly gated
        gated = gate * value
        # The gating should make some values smaller
        assert np.mean(np.abs(gated)) < np.mean(np.abs(value)) * 1.5


class TestParameterCounts:
    """Tests for parameter counting."""

    def test_standard_ffn_params(self):
        """Standard FFN parameter count."""
        d_model, d_ff = 512, 2048

        ffn = FeedForward(d_model, d_ff, use_bias=True)
        params = count_parameters(ffn)

        # W1: 512*2048, b1: 2048, W2: 2048*512, b2: 512
        expected = d_model * d_ff + d_ff + d_ff * d_model + d_model
        assert params == expected

    def test_standard_ffn_no_bias_params(self):
        """Standard FFN without bias parameter count."""
        d_model, d_ff = 512, 2048

        ffn = FeedForward(d_model, d_ff, use_bias=False)
        params = count_parameters(ffn)

        # W1: 512*2048, W2: 2048*512
        expected = d_model * d_ff + d_ff * d_model
        assert params == expected

    def test_swiglu_params(self):
        """SwiGLU FFN parameter count."""
        d_model = 512
        d_ff = 1365  # Typical SwiGLU hidden dim

        ffn = SwiGLUFeedForward(d_model, d_ff)
        params = count_parameters(ffn)

        # W1: 512*1365, W2: 512*1365, W3: 1365*512
        expected = d_model * d_ff + d_model * d_ff + d_ff * d_model
        assert params == expected

    def test_swiglu_matches_standard(self):
        """SwiGLU with default d_ff should have similar params to standard FFN."""
        d_model = 512

        standard = FeedForward(d_model, use_bias=False)
        swiglu = SwiGLUFeedForward(d_model)

        standard_params = count_parameters(standard)
        swiglu_params = count_parameters(swiglu)

        # Should be within 20% of each other
        ratio = swiglu_params / standard_params
        assert 0.8 < ratio < 1.2


class TestPyTorchComparison:
    """Tests comparing with PyTorch (if available)."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not available")
            return False

    def test_gelu_matches_pytorch(self, torch_available):
        """GELU should match PyTorch's GELU."""
        import torch
        import torch.nn.functional as F

        x_np = np.random.randn(100).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Our GELU (approximation)
        out_np = gelu(x_np)

        # PyTorch GELU (also uses approximation with 'tanh')
        out_torch = F.gelu(x_torch, approximate='tanh').numpy()

        np.testing.assert_allclose(out_np, out_torch, rtol=1e-4, atol=1e-4)

    def test_silu_matches_pytorch(self, torch_available):
        """SiLU should match PyTorch's SiLU."""
        import torch
        import torch.nn.functional as F

        x_np = np.random.randn(100).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        out_np = silu(x_np)
        out_torch = F.silu(x_torch).numpy()

        np.testing.assert_allclose(out_np, out_torch, rtol=1e-5, atol=1e-5)
