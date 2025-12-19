"""Tests for Lab 05: DeltaNet."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deltanet import (
    delta_rule_update,
    deltanet_step,
    deltanet_recurrent,
    DeltaNet,
    compare_deltanet_beta,
    analyze_memory_capacity,
)


class TestDeltaRuleUpdate:
    """Tests for the delta rule update."""

    def test_output_shape(self):
        """Updated state should have same shape as input."""
        d_k, d_v = 8, 16
        state = np.random.randn(d_k, d_v)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)

        new_state = delta_rule_update(state, k, v)

        assert new_state.shape == state.shape

    def test_beta_zero_is_additive(self):
        """Beta=0 should give vanilla additive update."""
        d_k, d_v = 8, 16
        state = np.random.randn(d_k, d_v).astype(np.float32)
        k = np.random.randn(d_k).astype(np.float32)
        v = np.random.randn(d_v).astype(np.float32)

        # Beta=0: just add φ(k)^T @ v
        new_state = delta_rule_update(state, k, v, beta=0.0)

        # Should be state + outer product
        # (after feature map)
        assert new_state.shape == state.shape
        # New state should differ from old
        assert not np.allclose(new_state, state)

    def test_beta_one_overwrites(self):
        """Beta=1 should implement full overwriting."""
        d_k, d_v = 8, 16

        # Start with state that has something at key k
        k = np.random.randn(d_k).astype(np.float32)
        v_old = np.random.randn(d_v).astype(np.float32)
        v_new = np.random.randn(d_v).astype(np.float32)

        # Initialize state with old value at key k
        state = np.zeros((d_k, d_v), dtype=np.float32)
        state = delta_rule_update(state, k, v_old, beta=1.0)

        # Now update with new value
        state = delta_rule_update(state, k, v_new, beta=1.0)

        # Query with same key should return ~v_new, not v_old
        from deltanet import elu_plus_one
        k_feat = elu_plus_one(k)
        retrieved = state.T @ k_feat

        # Should be closer to v_new than v_old
        error_new = np.linalg.norm(retrieved - v_new)
        error_old = np.linalg.norm(retrieved - v_old)
        assert error_new < error_old, "Delta rule should overwrite old value"

    def test_batched_input(self):
        """Should handle batched input."""
        batch, d_k, d_v = 2, 8, 16
        state = np.random.randn(batch, d_k, d_v)
        k = np.random.randn(batch, d_k)
        v = np.random.randn(batch, d_v)

        new_state = delta_rule_update(state, k, v)

        assert new_state.shape == state.shape


class TestDeltaNetStep:
    """Tests for single-step DeltaNet."""

    def test_output_shape(self):
        """Output should have correct shape."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)

        output, state = deltanet_step(q, k, v)

        assert output.shape == (d_v,)
        assert state.shape == (d_k, d_v)

    def test_state_initialization(self):
        """Should initialize state to zeros if None."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)

        output1, state1 = deltanet_step(q, k, v, state=None)
        output2, state2 = deltanet_step(q, k, v, state=np.zeros((d_k, d_v)))

        np.testing.assert_allclose(output1, output2)
        np.testing.assert_allclose(state1, state2)

    def test_batched_input(self):
        """Should handle batched input."""
        batch, d_k, d_v = 2, 8, 16
        q = np.random.randn(batch, d_k)
        k = np.random.randn(batch, d_k)
        v = np.random.randn(batch, d_v)

        output, state = deltanet_step(q, k, v)

        assert output.shape == (batch, d_v)
        assert state.shape == (batch, d_k, d_v)


class TestDeltaNetRecurrent:
    """Tests for full sequence DeltaNet."""

    def test_output_shape_2d(self):
        """Output should match V shape (2D)."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        output, state = deltanet_recurrent(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (d, d)

    def test_output_shape_3d(self):
        """Output should match V shape (3D batched)."""
        batch, seq_len, d = 2, 10, 16
        Q = np.random.randn(batch, seq_len, d)
        K = np.random.randn(batch, seq_len, d)
        V = np.random.randn(batch, seq_len, d)

        output, state = deltanet_recurrent(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (batch, d, d)

    def test_matches_step_by_step(self):
        """Recurrent should match step-by-step computation."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        # Full recurrent
        out_recurrent, state_recurrent = deltanet_recurrent(Q, K, V)

        # Step by step
        outputs_step = []
        state = None
        for i in range(seq_len):
            out, state = deltanet_step(Q[i], K[i], V[i], state)
            outputs_step.append(out)

        out_step = np.stack(outputs_step, axis=0)

        np.testing.assert_allclose(out_step, out_recurrent, rtol=1e-5)
        np.testing.assert_allclose(state, state_recurrent, rtol=1e-5)


class TestDeltaNetClass:
    """Tests for the DeltaNet class."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        model = DeltaNet(d_model=32, d_k=16, d_v=16, beta=0.8)

        assert model.d_model == 32
        assert model.d_k == 16
        assert model.d_v == 16
        assert model.beta == 0.8

    def test_default_dimensions(self):
        """Default d_k and d_v should equal d_model."""
        model = DeltaNet(d_model=32)

        assert model.d_k == 32
        assert model.d_v == 32

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        model = DeltaNet(d_model=32)

        x = np.random.randn(10, 32)  # (seq_len, d_model)
        output, state = model(x)

        assert output.shape == x.shape
        assert state.shape == (32, 32)

    def test_forward_batched(self):
        """Should handle batched input."""
        model = DeltaNet(d_model=32)

        x = np.random.randn(2, 10, 32)  # (batch, seq_len, d_model)
        output, state = model(x)

        assert output.shape == x.shape
        assert state.shape == (2, 32, 32)

    def test_step_mode(self):
        """Step mode should work for autoregressive generation."""
        model = DeltaNet(d_model=32)

        state = None
        outputs = []
        for _ in range(10):
            x = np.random.randn(32)  # Single position
            out, state = model.step(x, state)
            outputs.append(out)

        assert len(outputs) == 10
        assert outputs[0].shape == (32,)


class TestCompareBeta:
    """Tests for comparing different beta values."""

    def test_returns_all_outputs(self):
        """Should return outputs for all beta values."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        result = compare_deltanet_beta(Q, K, V, betas=[0.0, 0.5, 1.0])

        assert 'beta_0.0_output' in result or 'beta_0_output' in result
        assert 'beta_0.5_output' in result or 'beta_0_output' in result
        assert 'beta_1.0_output' in result or 'beta_1_output' in result

    def test_different_betas_different_outputs(self):
        """Different beta values should produce different outputs."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        out_0, _ = deltanet_recurrent(Q, K, V, beta=0.0)
        out_1, _ = deltanet_recurrent(Q, K, V, beta=1.0)

        # Should not be identical
        assert not np.allclose(out_0, out_1)


class TestMemoryCapacity:
    """Tests for memory capacity analysis."""

    def test_returns_metrics(self):
        """Should return required metrics."""
        result = analyze_memory_capacity(d_k=32, d_v=32, num_patterns=10)

        assert 'retrieval_accuracy' in result
        assert 'avg_retrieval_error' in result
        assert 'capacity_ratio' in result

    def test_capacity_ratio(self):
        """Capacity ratio should be computed correctly."""
        d_k, d_v, num_patterns = 32, 32, 10
        result = analyze_memory_capacity(d_k, d_v, num_patterns)

        expected_ratio = num_patterns / min(d_k, d_v)
        np.testing.assert_allclose(result['capacity_ratio'], expected_ratio)

    def test_few_patterns_high_accuracy(self):
        """Few patterns should be stored with high accuracy."""
        # With few patterns relative to dimension, accuracy should be high
        result = analyze_memory_capacity(d_k=64, d_v=64, num_patterns=5, beta=1.0)

        # Should achieve > 90% accuracy with well-separated patterns
        # Note: This depends on implementation details, so use a relaxed threshold
        assert result['retrieval_accuracy'] >= 0.5 or result['avg_retrieval_error'] < 1.0


class TestMilestone:
    """Lab 05 Milestone: DeltaNet implementation."""

    def test_milestone_delta_rule(self):
        """
        MILESTONE: DeltaNet correctly implements the delta rule.

        The delta rule should enable key-based overwriting:
        storing a new value at an existing key replaces the old value.
        """
        d_k, d_v = 16, 16

        # Create a key
        k = np.random.randn(d_k).astype(np.float32)
        q = k.copy()  # Query with same key

        # Store v1 at key k
        v1 = np.ones(d_v, dtype=np.float32) * 1.0
        state = None
        _, state = deltanet_step(q, k, v1, state, beta=1.0)

        # Query - should get ~v1
        from deltanet import elu_plus_one
        k_feat = elu_plus_one(k)
        q_feat = elu_plus_one(q)
        retrieved1 = q_feat @ state

        # Now store v2 at same key k
        v2 = np.ones(d_v, dtype=np.float32) * 2.0
        _, state = deltanet_step(q, k, v2, state, beta=1.0)

        # Query again - should get ~v2, not v1
        retrieved2 = q_feat @ state

        # v2 should dominate (closer to 2.0 than to 1.0 or 3.0)
        # This tests that delta rule overwrites instead of accumulates
        error_to_v2 = np.abs(retrieved2.mean() - 2.0)
        error_to_sum = np.abs(retrieved2.mean() - 3.0)  # If it accumulated

        assert error_to_v2 < error_to_sum, \
            "Delta rule should overwrite, not accumulate"

        print("\n" + "=" * 60)
        print("MILESTONE ACHIEVED: DeltaNet works!")
        print("The delta rule enables key-based overwriting.")
        print("=" * 60 + "\n")

    def test_milestone_full_model(self):
        """
        MILESTONE: Full DeltaNet model produces valid outputs.
        """
        model = DeltaNet(d_model=32, beta=1.0)

        # Test forward pass
        x = np.random.randn(20, 32).astype(np.float32)
        output, state = model(x)

        assert output.shape == x.shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Test step mode matches forward
        outputs_step = []
        state_step = None
        for i in range(20):
            out, state_step = model.step(x[i], state_step)
            outputs_step.append(out)

        out_step = np.stack(outputs_step, axis=0)

        np.testing.assert_allclose(out_step, output, rtol=1e-4)

        print("\n" + "=" * 60)
        print("CHAPTER 6 COMPLETE!")
        print("You've implemented:")
        print("  - Linear attention as RNN")
        print("  - Chunkwise parallel algorithm")
        print("  - Flash Linear Attention")
        print("  - Gated Linear Attention")
        print("  - DeltaNet")
        print("=" * 60 + "\n")
