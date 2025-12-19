"""Tests for Lab 02: Positional Encodings."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from positional_encoding import (
    sinusoidal_encoding,
    LearnedPositionalEmbedding,
    precompute_freqs_cis,
    apply_rotary_emb,
    rotate_half,
    apply_rotary_emb_real,
)


class TestSinusoidal:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self):
        """Output should have correct shape."""
        pe = sinusoidal_encoding(100, 512)
        assert pe.shape == (100, 512)

    def test_bounded_values(self):
        """Values should be between -1 and 1 (sin/cos range)."""
        pe = sinusoidal_encoding(100, 512)
        assert np.all(pe >= -1) and np.all(pe <= 1)

    def test_unique_positions(self):
        """Each position should have a unique encoding."""
        pe = sinusoidal_encoding(100, 512)

        # Check that no two rows are identical
        for i in range(99):
            assert not np.allclose(pe[i], pe[i + 1])

    def test_position_zero(self):
        """Position 0 should have sin(0)=0 for even dims, cos(0)=1 for odd dims."""
        pe = sinusoidal_encoding(10, 8)

        # Even dimensions: sin(0) = 0
        np.testing.assert_allclose(pe[0, ::2], 0, atol=1e-7)

        # Odd dimensions: cos(0) = 1
        np.testing.assert_allclose(pe[0, 1::2], 1, atol=1e-7)

    def test_different_frequencies(self):
        """Different dimensions should have different frequencies."""
        pe = sinusoidal_encoding(1000, 512)

        # Low dimension (high freq) should oscillate more
        low_dim_changes = np.abs(np.diff(pe[:, 0])).sum()
        high_dim_changes = np.abs(np.diff(pe[:, -2])).sum()

        assert low_dim_changes > high_dim_changes

    def test_deterministic(self):
        """Same inputs should give same outputs."""
        pe1 = sinusoidal_encoding(100, 512)
        pe2 = sinusoidal_encoding(100, 512)

        np.testing.assert_array_equal(pe1, pe2)

    def test_small_example(self):
        """Test with small known values."""
        pe = sinusoidal_encoding(2, 4)

        # Position 0: [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
        np.testing.assert_allclose(pe[0], [0, 1, 0, 1], atol=1e-6)

        # Position 1 should have sin(theta) and cos(theta) for various theta
        assert pe[1, 0] != 0  # sin(theta) for dim 0
        assert pe[1, 1] != 1  # cos(theta) for dim 0


class TestLearnedPositionalEmbedding:
    """Tests for learned positional embeddings."""

    def test_init(self):
        """Should initialize with correct shapes."""
        pe = LearnedPositionalEmbedding(1024, 512)

        assert pe.max_seq_len == 1024
        assert pe.d_model == 512
        assert pe.embedding.shape == (1024, 512)

    def test_forward_shape(self):
        """Forward should return correct shape."""
        pe = LearnedPositionalEmbedding(1024, 512)

        out = pe.forward(100)

        assert out.shape == (100, 512)

    def test_forward_subset(self):
        """Forward should return first seq_len embeddings."""
        pe = LearnedPositionalEmbedding(1024, 512)

        out = pe.forward(100)

        np.testing.assert_array_equal(out, pe.embedding[:100])

    def test_callable(self):
        """Should be callable like a function."""
        pe = LearnedPositionalEmbedding(1024, 512)

        out1 = pe.forward(100)
        out2 = pe(100)

        np.testing.assert_array_equal(out1, out2)

    def test_exceeds_max_len(self):
        """Should raise error if seq_len > max_seq_len."""
        pe = LearnedPositionalEmbedding(100, 512)

        with pytest.raises(ValueError):
            pe.forward(200)

    def test_embeddings_initialized(self):
        """Embeddings should not be all zeros."""
        pe = LearnedPositionalEmbedding(1024, 512)

        assert not np.allclose(pe.embedding, 0)


class TestPrecomputeFreqsCis:
    """Tests for RoPE frequency precomputation."""

    def test_output_shape(self):
        """Output should have shape (max_seq_len, d_model // 2)."""
        freqs = precompute_freqs_cis(64, 100)

        assert freqs.shape == (100, 32)

    def test_complex_dtype(self):
        """Output should be complex."""
        freqs = precompute_freqs_cis(64, 100)

        assert np.iscomplexobj(freqs)

    def test_unit_magnitude(self):
        """All values should have magnitude 1 (unit complex numbers)."""
        freqs = precompute_freqs_cis(64, 100)

        magnitudes = np.abs(freqs)
        np.testing.assert_allclose(magnitudes, 1, atol=1e-6)

    def test_position_zero(self):
        """Position 0 should be all ones (exp(0) = 1)."""
        freqs = precompute_freqs_cis(64, 100)

        np.testing.assert_allclose(freqs[0], 1, atol=1e-6)

    def test_different_frequencies(self):
        """Different dimension pairs should have different frequencies."""
        freqs = precompute_freqs_cis(64, 100)

        # Check angles change at different rates
        angle_0 = np.angle(freqs[:, 0])  # First dim pair
        angle_last = np.angle(freqs[:, -1])  # Last dim pair

        # First should change faster (higher frequency)
        changes_0 = np.abs(np.diff(angle_0)).sum()
        changes_last = np.abs(np.diff(angle_last)).sum()

        assert changes_0 > changes_last

    def test_custom_base(self):
        """Should accept custom base parameter."""
        freqs_10k = precompute_freqs_cis(64, 100, base=10000)
        freqs_1k = precompute_freqs_cis(64, 100, base=1000)

        # Different bases should give different frequencies
        assert not np.allclose(freqs_10k, freqs_1k)


class TestApplyRotaryEmb:
    """Tests for applying rotary embeddings."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        x = np.random.randn(2, 10, 8, 64)
        freqs = precompute_freqs_cis(64, 10)

        out = apply_rotary_emb(x, freqs)

        assert out.shape == x.shape

    def test_real_output(self):
        """Output should be real (not complex)."""
        x = np.random.randn(2, 10, 8, 64)
        freqs = precompute_freqs_cis(64, 10)

        out = apply_rotary_emb(x, freqs)

        assert not np.iscomplexobj(out)

    def test_position_zero_unchanged(self):
        """At position 0, input should be unchanged (rotation by 0)."""
        x = np.random.randn(2, 1, 8, 64)
        freqs = precompute_freqs_cis(64, 1)

        out = apply_rotary_emb(x, freqs)

        # freqs[0] = 1, so x should be unchanged
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_preserves_norm(self):
        """Rotation should preserve vector norms."""
        x = np.random.randn(2, 10, 8, 64)
        freqs = precompute_freqs_cis(64, 10)

        out = apply_rotary_emb(x, freqs)

        # Norm of each vector should be preserved
        in_norms = np.linalg.norm(x, axis=-1)
        out_norms = np.linalg.norm(out, axis=-1)

        np.testing.assert_allclose(in_norms, out_norms, rtol=1e-5)

    def test_2d_input(self):
        """Should work with 2D input (seq_len, d_k)."""
        x = np.random.randn(10, 64)
        freqs = precompute_freqs_cis(64, 10)

        out = apply_rotary_emb(x, freqs)

        assert out.shape == (10, 64)

    def test_3d_input(self):
        """Should work with 3D input (batch, seq_len, d_k)."""
        x = np.random.randn(4, 10, 64)
        freqs = precompute_freqs_cis(64, 10)

        out = apply_rotary_emb(x, freqs)

        assert out.shape == (4, 10, 64)


class TestRotateHalf:
    """Tests for rotate_half function."""

    def test_basic(self):
        """Test basic rotation."""
        x = np.array([1, 2, 3, 4])
        out = rotate_half(x)

        expected = np.array([-3, -4, 1, 2])
        np.testing.assert_array_equal(out, expected)

    def test_2d(self):
        """Should work with 2D input."""
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        out = rotate_half(x)

        expected = np.array([[-3, -4, 1, 2], [-7, -8, 5, 6]])
        np.testing.assert_array_equal(out, expected)

    def test_output_shape(self):
        """Output should have same shape as input."""
        x = np.random.randn(2, 10, 8, 64)
        out = rotate_half(x)

        assert out.shape == x.shape


class TestApplyRotaryEmbReal:
    """Tests for real-arithmetic RoPE implementation."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        seq_len, d_k = 10, 64
        x = np.random.randn(2, seq_len, 8, d_k)

        # Create cos/sin
        freqs = 1 / (10000 ** (2 * np.arange(d_k // 2) / d_k))
        angles = np.arange(seq_len)[:, None] * freqs[None, :]
        cos = np.cos(angles)
        sin = np.sin(angles)

        # Interleave to match d_k
        cos_full = np.repeat(cos, 2, axis=-1)
        sin_full = np.repeat(sin, 2, axis=-1)

        out = apply_rotary_emb_real(x, cos_full, sin_full)

        assert out.shape == x.shape

    def test_position_zero(self):
        """At position 0, cos=1, sin=0, so input unchanged."""
        d_k = 64
        x = np.random.randn(2, 1, 8, d_k)

        cos = np.ones((1, d_k))
        sin = np.zeros((1, d_k))

        out = apply_rotary_emb_real(x, cos, sin)

        np.testing.assert_allclose(out, x, atol=1e-6)


class TestRoPERelativePosition:
    """Tests verifying RoPE encodes relative position."""

    def test_dot_product_relative(self):
        """Dot product should depend on relative position, not absolute."""
        d_k = 64
        freqs = precompute_freqs_cis(d_k, 100)

        # Create two vectors
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)

        # Rotate q to position 10, k to position 5 (relative = 5)
        q_10 = apply_rotary_emb(q[None, :], freqs[10:11]).squeeze()
        k_5 = apply_rotary_emb(k[None, :], freqs[5:6]).squeeze()
        dot_10_5 = np.dot(q_10, k_5)

        # Rotate q to position 20, k to position 15 (relative = 5)
        q_20 = apply_rotary_emb(q[None, :], freqs[20:21]).squeeze()
        k_15 = apply_rotary_emb(k[None, :], freqs[15:16]).squeeze()
        dot_20_15 = np.dot(q_20, k_15)

        # The dot products should be equal (same relative position)
        np.testing.assert_allclose(dot_10_5, dot_20_15, rtol=1e-5)

    def test_different_relative_different_dot(self):
        """Different relative positions should give different dot products."""
        d_k = 64
        freqs = precompute_freqs_cis(d_k, 100)

        q = np.random.randn(d_k)
        k = np.random.randn(d_k)

        # Relative position = 5
        q_10 = apply_rotary_emb(q[None, :], freqs[10:11]).squeeze()
        k_5 = apply_rotary_emb(k[None, :], freqs[5:6]).squeeze()
        dot_rel_5 = np.dot(q_10, k_5)

        # Relative position = 10
        q_20 = apply_rotary_emb(q[None, :], freqs[20:21]).squeeze()
        k_10 = apply_rotary_emb(k[None, :], freqs[10:11]).squeeze()
        dot_rel_10 = np.dot(q_20, k_10)

        # Different relative positions -> different dot products
        assert not np.isclose(dot_rel_5, dot_rel_10)
