"""Tests for Lab 04: JAX Transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transforms import (
    attention_single,
    jit_attention,
    jit_attention_cached,
    batched_attention,
    batched_attention_shared_kv,
    attention_gradient,
    value_and_gradient,
    train_step,
    simple_model,
    jit_batched_attention,
    jit_batched_attention_cached,
    multi_head_batched,
)


class TestJit:
    """Tests for JIT compilation."""

    def test_jit_correctness(self):
        """JIT output should match non-JIT output."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        Q = jax.random.normal(keys[0], (8, 16))
        K = jax.random.normal(keys[1], (12, 16))
        V = jax.random.normal(keys[2], (12, 32))

        result = jit_attention(Q, K, V)
        expected = attention_single(Q, K, V)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_jit_shape(self):
        """JIT output shape should be correct."""
        Q = jnp.ones((4, 8))
        K = jnp.ones((6, 8))
        V = jnp.ones((6, 16))

        result = jit_attention(Q, K, V)
        assert result.shape == (4, 16)

    def test_jit_cached_performance(self):
        """Cached JIT should be fast after warmup."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        Q = jax.random.normal(keys[0], (64, 64))
        K = jax.random.normal(keys[1], (64, 64))
        V = jax.random.normal(keys[2], (64, 64))

        # Warmup
        _ = jit_attention_cached(Q, K, V)
        _ = jit_attention_cached(Q, K, V)

        # Should complete quickly (no recompilation)
        start = time.time()
        for _ in range(10):
            result = jit_attention_cached(Q, K, V)
        result.block_until_ready()
        elapsed = time.time() - start

        # Should be reasonably fast (less than 1 second for 10 iterations)
        assert elapsed < 1.0


class TestVmap:
    """Tests for vmap vectorization."""

    def test_batched_attention_shape(self):
        """Batched attention should have correct output shape."""
        Q = jnp.ones((8, 4, 16))
        K = jnp.ones((8, 6, 16))
        V = jnp.ones((8, 6, 32))

        result = batched_attention(Q, K, V)
        assert result.shape == (8, 4, 32)

    def test_batched_attention_correctness(self):
        """Batched results should match individual computations."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch = 4
        Q = jax.random.normal(keys[0], (batch, 8, 16))
        K = jax.random.normal(keys[1], (batch, 12, 16))
        V = jax.random.normal(keys[2], (batch, 12, 32))

        result = batched_attention(Q, K, V)

        # Check each batch element
        for i in range(batch):
            expected_i = attention_single(Q[i], K[i], V[i])
            assert jnp.allclose(result[i], expected_i, atol=1e-5)

    def test_batched_attention_shared_kv_shape(self):
        """Shared K/V attention should have correct shape."""
        Q = jnp.ones((8, 4, 16))
        K = jnp.ones((6, 16))  # Not batched
        V = jnp.ones((6, 32))  # Not batched

        result = batched_attention_shared_kv(Q, K, V)
        assert result.shape == (8, 4, 32)

    def test_batched_attention_shared_kv_correctness(self):
        """Shared K/V should broadcast correctly."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch = 4
        Q = jax.random.normal(keys[0], (batch, 8, 16))
        K = jax.random.normal(keys[1], (12, 16))  # Shared
        V = jax.random.normal(keys[2], (12, 32))  # Shared

        result = batched_attention_shared_kv(Q, K, V)

        # Check each batch element uses same K, V
        for i in range(batch):
            expected_i = attention_single(Q[i], K, V)
            assert jnp.allclose(result[i], expected_i, atol=1e-5)


class TestGrad:
    """Tests for automatic differentiation."""

    def test_gradient_shapes(self):
        """Gradient shapes should match input shapes."""
        Q = jnp.ones((4, 8))
        K = jnp.ones((6, 8))
        V = jnp.ones((6, 16))

        dQ, dK, dV = attention_gradient(Q, K, V)

        assert dQ.shape == Q.shape
        assert dK.shape == K.shape
        assert dV.shape == V.shape

    def test_gradient_not_nan(self):
        """Gradients should not contain NaN."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        Q = jax.random.normal(keys[0], (8, 16))
        K = jax.random.normal(keys[1], (12, 16))
        V = jax.random.normal(keys[2], (12, 32))

        dQ, dK, dV = attention_gradient(Q, K, V)

        assert not jnp.any(jnp.isnan(dQ))
        assert not jnp.any(jnp.isnan(dK))
        assert not jnp.any(jnp.isnan(dV))

    def test_value_and_gradient(self):
        """value_and_grad should return correct loss and gradients."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        Q = jax.random.normal(keys[0], (4, 8))
        K = jax.random.normal(keys[1], (6, 8))
        V = jax.random.normal(keys[2], (6, 16))

        loss, (dQ, dK, dV) = value_and_gradient(Q, K, V)

        # Loss should be a scalar
        assert loss.shape == ()

        # Gradients should have correct shapes
        assert dQ.shape == Q.shape
        assert dK.shape == K.shape
        assert dV.shape == V.shape

        # Gradients should match separate computation
        dQ_sep, dK_sep, dV_sep = attention_gradient(Q, K, V)
        assert jnp.allclose(dQ, dQ_sep, atol=1e-5)
        assert jnp.allclose(dK, dK_sep, atol=1e-5)
        assert jnp.allclose(dV, dV_sep, atol=1e-5)

    def test_gradient_numerical_check(self):
        """Gradient should approximately match finite differences."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        Q = jax.random.normal(keys[0], (2, 4))
        K = jax.random.normal(keys[1], (3, 4))
        V = jax.random.normal(keys[2], (3, 4))

        # Compute analytical gradient
        dQ_analytical, _, _ = attention_gradient(Q, K, V)

        # Compute numerical gradient for a few elements
        eps = 1e-4
        from transforms import attention_loss

        for i in range(min(2, Q.shape[0])):
            for j in range(min(2, Q.shape[1])):
                Q_plus = Q.at[i, j].add(eps)
                Q_minus = Q.at[i, j].add(-eps)

                loss_plus = attention_loss(Q_plus, K, V)
                loss_minus = attention_loss(Q_minus, K, V)

                dQ_numerical = (loss_plus - loss_minus) / (2 * eps)

                assert jnp.allclose(dQ_analytical[i, j], dQ_numerical, atol=1e-3)


class TestTrainStep:
    """Tests for training step."""

    def test_train_step_returns_updated_params(self):
        """Train step should return updated parameters."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        params = {
            'W1': jax.random.normal(keys[0], (4, 8)) * 0.1,
            'b1': jnp.zeros(8),
            'W2': jax.random.normal(keys[1], (8, 2)) * 0.1,
            'b2': jnp.zeros(2),
        }
        x = jax.random.normal(keys[2], (16, 4))
        y = jax.random.normal(keys[3], (16, 2))

        new_params, loss = train_step(params, x, y)

        # Should have same structure
        assert set(new_params.keys()) == set(params.keys())

        # Parameters should be updated (different from original)
        for k in params:
            assert not jnp.allclose(new_params[k], params[k])

    def test_train_step_decreases_loss(self):
        """Multiple train steps should decrease loss."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        params = {
            'W1': jax.random.normal(keys[0], (4, 8)) * 0.1,
            'b1': jnp.zeros(8),
            'W2': jax.random.normal(keys[1], (8, 2)) * 0.1,
            'b2': jnp.zeros(2),
        }
        x = jax.random.normal(keys[2], (16, 4))
        y = jax.random.normal(keys[3], (16, 2))

        # Run several training steps
        losses = []
        for _ in range(10):
            params, loss = train_step(params, x, y, lr=0.1)
            losses.append(loss)

        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestComposed:
    """Tests for composed transformations."""

    def test_jit_batched_correctness(self):
        """JIT+vmap should match unbatched computation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch = 4
        Q = jax.random.normal(keys[0], (batch, 8, 16))
        K = jax.random.normal(keys[1], (batch, 12, 16))
        V = jax.random.normal(keys[2], (batch, 12, 32))

        result = jit_batched_attention(Q, K, V)

        # Check each batch element
        for i in range(batch):
            expected_i = attention_single(Q[i], K[i], V[i])
            assert jnp.allclose(result[i], expected_i, atol=1e-5)

    def test_jit_batched_shape(self):
        """JIT+vmap should produce correct shape."""
        Q = jnp.ones((8, 4, 16))
        K = jnp.ones((8, 6, 16))
        V = jnp.ones((8, 6, 32))

        result = jit_batched_attention(Q, K, V)
        assert result.shape == (8, 4, 32)


class TestMultiHeadBatched:
    """Tests for batched multi-head attention."""

    def test_multi_head_output_shape(self):
        """Multi-head batched should have correct output shape."""
        Q = jnp.ones((4, 8, 64))
        K = jnp.ones((4, 8, 64))
        V = jnp.ones((4, 8, 64))

        result = multi_head_batched(Q, K, V, num_heads=8)
        assert result.shape == (4, 8, 64)

    def test_multi_head_different_configs(self):
        """Should work with different head configurations."""
        for num_heads in [1, 2, 4, 8]:
            d_model = 64
            Q = jnp.ones((2, 4, d_model))
            K = jnp.ones((2, 6, d_model))
            V = jnp.ones((2, 6, d_model))

            result = multi_head_batched(Q, K, V, num_heads=num_heads)
            assert result.shape == (2, 4, d_model)

    def test_multi_head_no_nan(self):
        """Output should not contain NaN."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        Q = jax.random.normal(keys[0], (4, 8, 64))
        K = jax.random.normal(keys[1], (4, 12, 64))
        V = jax.random.normal(keys[2], (4, 12, 64))

        result = multi_head_batched(Q, K, V, num_heads=8)

        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class TestMilestone:
    """Chapter 12 Lab 04 Milestone."""

    def test_jax_transforms_milestone(self):
        """
        MILESTONE: JAX transformations working correctly.

        This demonstrates understanding of:
        - JIT compilation
        - Automatic vectorization (vmap)
        - Automatic differentiation (grad)
        - Composing transformations
        """
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)

        # Test dimensions
        batch = 4
        seq_q, seq_k = 8, 12
        d_k, d_v = 16, 32

        Q = jax.random.normal(keys[0], (seq_q, d_k))
        K = jax.random.normal(keys[1], (seq_k, d_k))
        V = jax.random.normal(keys[2], (seq_k, d_v))

        Q_batch = jax.random.normal(keys[3], (batch, seq_q, d_k))
        K_batch = jax.random.normal(keys[4], (batch, seq_k, d_k))
        V_batch = jax.random.normal(keys[5], (batch, seq_k, d_v))

        # Test JIT
        jit_result = jit_attention(Q, K, V)
        assert jit_result.shape == (seq_q, d_v)

        # Test vmap
        vmap_result = batched_attention(Q_batch, K_batch, V_batch)
        assert vmap_result.shape == (batch, seq_q, d_v)

        # Test grad
        dQ, dK, dV = attention_gradient(Q, K, V)
        assert dQ.shape == Q.shape
        assert dK.shape == K.shape
        assert dV.shape == V.shape

        # Test value_and_grad
        loss, (dQ2, dK2, dV2) = value_and_gradient(Q, K, V)
        assert loss.shape == ()

        # Test composed
        composed_result = jit_batched_attention(Q_batch, K_batch, V_batch)
        assert composed_result.shape == (batch, seq_q, d_v)

        # Test train step
        params = {
            'W1': jax.random.normal(keys[0], (4, 8)) * 0.1,
            'b1': jnp.zeros(8),
            'W2': jax.random.normal(keys[1], (8, 2)) * 0.1,
            'b2': jnp.zeros(2),
        }
        x = jax.random.normal(keys[2], (16, 4))
        y = jax.random.normal(keys[3], (16, 2))
        new_params, train_loss = train_step(params, x, y)
        assert set(new_params.keys()) == set(params.keys())

        print("\n" + "=" * 60)
        print("Lab 04 Milestone Achieved!")
        print("JAX transformations (jit, vmap, grad) working correctly.")
        print("=" * 60 + "\n")
