"""Tests for Lab 03: JAX Introduction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jax_attention import (
    softmax,
    scaled_dot_product_attention,
    linear,
    multi_head_attention,
    feedforward,
    attention_reference,
)


class TestSoftmax:
    """Tests for softmax function."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert jnp.allclose(jnp.sum(result), 1.0, atol=1e-6)

    def test_softmax_2d_axis_minus_1(self):
        """Softmax along last axis should sum to 1 for each row."""
        x = jnp.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax(x, axis=-1)

        # Each row should sum to 1
        row_sums = jnp.sum(result, axis=-1)
        assert jnp.allclose(row_sums, jnp.ones(2), atol=1e-6)

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)

        # Should not have NaN or Inf
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))

        # Should still sum to 1
        assert jnp.allclose(jnp.sum(result), 1.0, atol=1e-6)

    def test_softmax_matches_jax(self):
        """Should match JAX's built-in softmax."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (10, 20))

        result = softmax(x, axis=-1)
        expected = jax.nn.softmax(x, axis=-1)

        assert jnp.allclose(result, expected, atol=1e-6)

    def test_softmax_shape_preserved(self):
        """Output shape should match input shape."""
        x = jnp.ones((3, 4, 5))
        result = softmax(x, axis=-1)
        assert result.shape == x.shape


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_basic_attention(self):
        """Basic attention should work correctly."""
        seq_q, seq_k, d_k, d_v = 4, 6, 8, 16

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        Q = jax.random.normal(keys[0], (seq_q, d_k))
        K = jax.random.normal(keys[1], (seq_k, d_k))
        V = jax.random.normal(keys[2], (seq_k, d_v))

        result = scaled_dot_product_attention(Q, K, V)
        expected = attention_reference(Q, K, V)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_attention_output_shape(self):
        """Output shape should be (seq_q, d_v)."""
        Q = jnp.ones((4, 8))
        K = jnp.ones((6, 8))
        V = jnp.ones((6, 16))

        result = scaled_dot_product_attention(Q, K, V)
        assert result.shape == (4, 16)

    def test_attention_with_mask(self):
        """Masked positions should not contribute."""
        Q = jnp.ones((4, 8))
        K = jnp.ones((4, 8))
        V = jnp.arange(32).reshape(4, 8).astype(jnp.float32)

        # Mask out all but first key
        mask = jnp.array([
            [False, True, True, True],
            [False, True, True, True],
            [False, True, True, True],
            [False, True, True, True],
        ])

        result = scaled_dot_product_attention(Q, K, V, mask=mask)
        expected = attention_reference(Q, K, V, mask=mask)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_attention_self_attention(self):
        """Self-attention (Q=K=V) should work."""
        x = jnp.ones((4, 8))
        result = scaled_dot_product_attention(x, x, x)
        assert result.shape == (4, 8)

    def test_attention_uniform_when_equal(self):
        """With uniform Q and K, attention weights should be uniform."""
        Q = jnp.ones((4, 8))
        K = jnp.ones((6, 8))
        V = jnp.ones((6, 16)) * 3.14

        result = scaled_dot_product_attention(Q, K, V)

        # All outputs should be 3.14 (uniform weights on uniform values)
        expected = jnp.ones((4, 16)) * 3.14
        assert jnp.allclose(result, expected, atol=1e-5)


class TestLinear:
    """Tests for linear transformation."""

    def test_linear_basic(self):
        """Basic linear transformation."""
        x = jnp.ones((4, 8))
        weight = jnp.ones((8, 16))
        bias = jnp.zeros((16,))

        result = linear(x, weight, bias)
        expected = x @ weight + bias

        assert jnp.allclose(result, expected)

    def test_linear_no_bias(self):
        """Linear without bias."""
        x = jnp.ones((4, 8))
        weight = jnp.ones((8, 16))

        result = linear(x, weight)
        expected = x @ weight

        assert jnp.allclose(result, expected)

    def test_linear_shape(self):
        """Output shape should be correct."""
        x = jnp.ones((4, 8))
        weight = jnp.ones((8, 16))

        result = linear(x, weight)
        assert result.shape == (4, 16)

    def test_linear_with_random(self):
        """Linear with random inputs."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        x = jax.random.normal(keys[0], (10, 32))
        weight = jax.random.normal(keys[1], (32, 64))
        bias = jax.random.normal(keys[2], (64,))

        result = linear(x, weight, bias)
        expected = x @ weight + bias

        assert jnp.allclose(result, expected, atol=1e-5)


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_mha_output_shape(self):
        """Output shape should be (seq_q, d_model)."""
        seq_q, seq_k, d_model, num_heads = 4, 6, 64, 8

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 7)

        Q = jax.random.normal(keys[0], (seq_q, d_model))
        K = jax.random.normal(keys[1], (seq_k, d_model))
        V = jax.random.normal(keys[2], (seq_k, d_model))
        W_q = jax.random.normal(keys[3], (d_model, d_model)) * 0.01
        W_k = jax.random.normal(keys[4], (d_model, d_model)) * 0.01
        W_v = jax.random.normal(keys[5], (d_model, d_model)) * 0.01
        W_o = jax.random.normal(keys[6], (d_model, d_model)) * 0.01

        result = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
        assert result.shape == (seq_q, d_model)

    def test_mha_self_attention(self):
        """Self-attention should work."""
        seq_len, d_model, num_heads = 8, 32, 4

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        x = jax.random.normal(keys[0], (seq_len, d_model))
        W_q = jax.random.normal(keys[1], (d_model, d_model)) * 0.01
        W_k = jax.random.normal(keys[2], (d_model, d_model)) * 0.01
        W_v = jax.random.normal(keys[3], (d_model, d_model)) * 0.01
        W_o = jax.random.normal(keys[4], (d_model, d_model)) * 0.01

        result = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
        assert result.shape == (seq_len, d_model)

    def test_mha_different_num_heads(self):
        """Should work with different number of heads."""
        seq_len, d_model = 4, 64

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        x = jax.random.normal(keys[0], (seq_len, d_model))
        W_q = jax.random.normal(keys[1], (d_model, d_model)) * 0.01
        W_k = jax.random.normal(keys[2], (d_model, d_model)) * 0.01
        W_v = jax.random.normal(keys[3], (d_model, d_model)) * 0.01
        W_o = jax.random.normal(keys[4], (d_model, d_model)) * 0.01

        for num_heads in [1, 2, 4, 8, 16]:
            result = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
            assert result.shape == (seq_len, d_model)

    def test_mha_no_nan_or_inf(self):
        """Output should not contain NaN or Inf."""
        seq_len, d_model, num_heads = 8, 64, 8

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        x = jax.random.normal(keys[0], (seq_len, d_model))
        W_q = jax.random.normal(keys[1], (d_model, d_model)) * 0.01
        W_k = jax.random.normal(keys[2], (d_model, d_model)) * 0.01
        W_v = jax.random.normal(keys[3], (d_model, d_model)) * 0.01
        W_o = jax.random.normal(keys[4], (d_model, d_model)) * 0.01

        result = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class TestFeedforward:
    """Tests for feedforward network."""

    def test_feedforward_output_shape(self):
        """Output shape should match input shape."""
        seq_len, d_model, d_ff = 4, 64, 256

        x = jnp.ones((seq_len, d_model))
        W1 = jnp.ones((d_model, d_ff)) * 0.01
        b1 = jnp.zeros((d_ff,))
        W2 = jnp.ones((d_ff, d_model)) * 0.01
        b2 = jnp.zeros((d_model,))

        result = feedforward(x, W1, b1, W2, b2)
        assert result.shape == (seq_len, d_model)

    def test_feedforward_random(self):
        """Feedforward with random inputs."""
        seq_len, d_model, d_ff = 8, 64, 256

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        x = jax.random.normal(keys[0], (seq_len, d_model))
        W1 = jax.random.normal(keys[1], (d_model, d_ff)) * 0.01
        b1 = jax.random.normal(keys[2], (d_ff,)) * 0.01
        W2 = jax.random.normal(keys[3], (d_ff, d_model)) * 0.01
        b2 = jax.random.normal(keys[4], (d_model,)) * 0.01

        result = feedforward(x, W1, b1, W2, b2)

        # Compute expected
        hidden = x @ W1 + b1
        hidden = jax.nn.gelu(hidden)
        expected = hidden @ W2 + b2

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_feedforward_no_nan(self):
        """Output should not contain NaN."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        x = jax.random.normal(keys[0], (8, 64))
        W1 = jax.random.normal(keys[1], (64, 256)) * 0.01
        b1 = jnp.zeros((256,))
        W2 = jax.random.normal(keys[2], (256, 64)) * 0.01
        b2 = jnp.zeros((64,))

        result = feedforward(x, W1, b1, W2, b2)

        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class TestMilestone:
    """Chapter 12 Lab 03 Milestone."""

    def test_jax_fundamentals_milestone(self):
        """
        MILESTONE: JAX attention implementation works correctly.

        This demonstrates understanding of:
        - JAX's functional programming model
        - Array operations with jax.numpy
        - Implementing attention from scratch
        """
        # Test dimensions
        seq_q, seq_k = 16, 16
        d_model = 64
        d_ff = 256
        num_heads = 8

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 10)

        # Create inputs
        Q = jax.random.normal(keys[0], (seq_q, d_model))
        K = jax.random.normal(keys[1], (seq_k, d_model))
        V = jax.random.normal(keys[2], (seq_k, d_model))

        # Create weights
        W_q = jax.random.normal(keys[3], (d_model, d_model)) * 0.02
        W_k = jax.random.normal(keys[4], (d_model, d_model)) * 0.02
        W_v = jax.random.normal(keys[5], (d_model, d_model)) * 0.02
        W_o = jax.random.normal(keys[6], (d_model, d_model)) * 0.02

        W1 = jax.random.normal(keys[7], (d_model, d_ff)) * 0.02
        b1 = jnp.zeros((d_ff,))
        W2 = jax.random.normal(keys[8], (d_ff, d_model)) * 0.02
        b2 = jnp.zeros((d_model,))

        # Test softmax
        soft_result = softmax(Q[0], axis=-1)
        assert jnp.allclose(jnp.sum(soft_result), 1.0, atol=1e-6)

        # Test attention
        attn_result = scaled_dot_product_attention(Q, K, V)
        assert attn_result.shape == (seq_q, d_model)

        # Test multi-head attention
        mha_result = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
        assert mha_result.shape == (seq_q, d_model)

        # Test feedforward
        ff_result = feedforward(Q, W1, b1, W2, b2)
        assert ff_result.shape == (seq_q, d_model)

        # Check no NaN/Inf
        for result in [soft_result, attn_result, mha_result, ff_result]:
            assert not jnp.any(jnp.isnan(result))
            assert not jnp.any(jnp.isinf(result))

        print("\n" + "=" * 60)
        print("Lab 03 Milestone Achieved!")
        print("JAX attention fundamentals working correctly.")
        print("=" * 60 + "\n")
