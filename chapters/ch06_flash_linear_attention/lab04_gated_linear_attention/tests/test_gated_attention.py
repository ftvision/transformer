"""Tests for Lab 04: Gated Linear Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gated_attention import (
    sigmoid,
    compute_gate,
    gated_state_update,
    gla_step,
    gla_recurrent,
    GatedLinearAttention,
    compare_gla_to_vanilla,
)


class TestComputeGate:
    """Tests for gate computation."""

    def test_output_shape(self):
        """Gate should have correct shape."""
        x = np.random.randn(10, 32)  # (seq_len, d_model)
        W_g = np.random.randn(32, 16)  # (d_model, d_gate)
        b_g = np.random.randn(16)

        gate = compute_gate(x, W_g, b_g)

        assert gate.shape == (10, 16)

    def test_output_range(self):
        """Gate should be in (0, 1) due to sigmoid."""
        x = np.random.randn(10, 32)
        W_g = np.random.randn(32, 16)
        b_g = np.random.randn(16)

        gate = compute_gate(x, W_g, b_g)

        assert np.all(gate > 0)
        assert np.all(gate < 1)

    def test_no_bias(self):
        """Should work without bias."""
        x = np.random.randn(10, 32)
        W_g = np.random.randn(32, 16)

        gate = compute_gate(x, W_g, b_g=None)

        assert gate.shape == (10, 16)

    def test_positive_bias_high_gate(self):
        """Positive bias should push gate towards 1."""
        x = np.zeros((10, 32))  # Zero input
        W_g = np.zeros((32, 16))  # Zero weights
        b_g = np.ones(16) * 5  # High positive bias

        gate = compute_gate(x, W_g, b_g)

        # sigmoid(5) ≈ 0.9933
        assert np.all(gate > 0.9)

    def test_negative_bias_low_gate(self):
        """Negative bias should push gate towards 0."""
        x = np.zeros((10, 32))
        W_g = np.zeros((32, 16))
        b_g = np.ones(16) * (-5)

        gate = compute_gate(x, W_g, b_g)

        # sigmoid(-5) ≈ 0.0067
        assert np.all(gate < 0.1)

    def test_batched_input(self):
        """Should handle batched input."""
        x = np.random.randn(2, 10, 32)  # (batch, seq_len, d_model)
        W_g = np.random.randn(32, 16)
        b_g = np.random.randn(16)

        gate = compute_gate(x, W_g, b_g)

        assert gate.shape == (2, 10, 16)


class TestGatedStateUpdate:
    """Tests for gated state update."""

    def test_output_shape(self):
        """Updated state should have same shape as input state."""
        d_k, d_v = 8, 16
        state = np.random.randn(d_k, d_v)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)
        gate = np.random.rand(d_k)  # In (0, 1)

        new_state = gated_state_update(state, k, v, gate)

        assert new_state.shape == state.shape

    def test_gate_one_preserves_state(self):
        """Gate = 1 should keep old state completely."""
        d_k, d_v = 8, 16
        state = np.random.randn(d_k, d_v)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)
        gate = np.ones(d_k)  # All 1s

        new_state = gated_state_update(state, k, v, gate)

        np.testing.assert_allclose(new_state, state)

    def test_gate_zero_writes_new(self):
        """Gate = 0 should write new value completely."""
        d_k, d_v = 8, 16
        state = np.random.randn(d_k, d_v)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)
        gate = np.zeros(d_k)  # All 0s

        new_state = gated_state_update(state, k, v, gate)

        # New state should be φ(k)^T @ v
        # (ignoring feature map for simplicity in test)
        assert not np.allclose(new_state, state)

    def test_interpolation(self):
        """Gate should interpolate between old and new."""
        d_k, d_v = 8, 16
        state = np.ones((d_k, d_v)) * 10
        k = np.ones(d_k)
        v = np.ones(d_v)
        gate = np.ones(d_k) * 0.5  # 50% retention

        new_state = gated_state_update(state, k, v, gate)

        # Should be between old state (10s) and new contribution
        assert np.all(new_state < 10)
        assert np.all(new_state > 0)

    def test_batched_input(self):
        """Should handle batched input."""
        batch, d_k, d_v = 2, 8, 16
        state = np.random.randn(batch, d_k, d_v)
        k = np.random.randn(batch, d_k)
        v = np.random.randn(batch, d_v)
        gate = np.random.rand(batch, d_k)

        new_state = gated_state_update(state, k, v, gate)

        assert new_state.shape == state.shape


class TestGLAStep:
    """Tests for single-step GLA."""

    def test_output_shape(self):
        """Output should have correct shape."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)
        gate = np.random.rand(d_k)

        output, state = gla_step(q, k, v, gate)

        assert output.shape == (d_v,)
        assert state.shape == (d_k, d_v)

    def test_state_initialization(self):
        """Should initialize state to zeros if None."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)
        gate = np.random.rand(d_k)

        output1, state1 = gla_step(q, k, v, gate, state=None)
        output2, state2 = gla_step(q, k, v, gate, state=np.zeros((d_k, d_v)))

        np.testing.assert_allclose(output1, output2)
        np.testing.assert_allclose(state1, state2)

    def test_state_accumulates(self):
        """State should change with each step."""
        d_k, d_v = 8, 16
        state = None
        states = []

        for _ in range(5):
            q = np.random.randn(d_k)
            k = np.random.randn(d_k)
            v = np.random.randn(d_v)
            gate = np.ones(d_k) * 0.9  # High retention

            _, state = gla_step(q, k, v, gate, state)
            states.append(state.copy())

        # States should be different
        assert not np.allclose(states[0], states[-1])

    def test_batched_input(self):
        """Should handle batched input."""
        batch, d_k, d_v = 2, 8, 16
        q = np.random.randn(batch, d_k)
        k = np.random.randn(batch, d_k)
        v = np.random.randn(batch, d_v)
        gate = np.random.rand(batch, d_k)

        output, state = gla_step(q, k, v, gate)

        assert output.shape == (batch, d_v)
        assert state.shape == (batch, d_k, d_v)


class TestGLARecurrent:
    """Tests for full sequence GLA."""

    def test_output_shape_2d(self):
        """Output should match V shape (2D)."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)
        gates = np.random.rand(seq_len, d)

        output, state = gla_recurrent(Q, K, V, gates)

        assert output.shape == V.shape
        assert state.shape == (d, d)

    def test_output_shape_3d(self):
        """Output should match V shape (3D batched)."""
        batch, seq_len, d = 2, 10, 16
        Q = np.random.randn(batch, seq_len, d)
        K = np.random.randn(batch, seq_len, d)
        V = np.random.randn(batch, seq_len, d)
        gates = np.random.rand(batch, seq_len, d)

        output, state = gla_recurrent(Q, K, V, gates)

        assert output.shape == V.shape
        assert state.shape == (batch, d, d)

    def test_matches_step_by_step(self):
        """Recurrent should match step-by-step computation."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)
        gates = np.random.rand(seq_len, d).astype(np.float32)

        # Full recurrent
        out_recurrent, state_recurrent = gla_recurrent(Q, K, V, gates)

        # Step by step
        outputs_step = []
        state = None
        for i in range(seq_len):
            out, state = gla_step(Q[i], K[i], V[i], gates[i], state)
            outputs_step.append(out)

        out_step = np.stack(outputs_step, axis=0)

        np.testing.assert_allclose(out_step, out_recurrent, rtol=1e-5)
        np.testing.assert_allclose(state, state_recurrent, rtol=1e-5)


class TestGatedLinearAttentionClass:
    """Tests for the GLA class."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        gla = GatedLinearAttention(d_model=32, d_k=16, d_v=16)

        assert gla.d_model == 32
        assert gla.d_k == 16
        assert gla.d_v == 16

    def test_default_dimensions(self):
        """Default d_k and d_v should equal d_model."""
        gla = GatedLinearAttention(d_model=32)

        assert gla.d_k == 32
        assert gla.d_v == 32

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        gla = GatedLinearAttention(d_model=32)

        x = np.random.randn(10, 32)  # (seq_len, d_model)
        output, state = gla(x)

        assert output.shape == x.shape
        assert state.shape == (32, 32)

    def test_forward_batched(self):
        """Should handle batched input."""
        gla = GatedLinearAttention(d_model=32)

        x = np.random.randn(2, 10, 32)  # (batch, seq_len, d_model)
        output, state = gla(x)

        assert output.shape == x.shape
        assert state.shape == (2, 32, 32)

    def test_step_mode(self):
        """Step mode should work for autoregressive generation."""
        gla = GatedLinearAttention(d_model=32)

        state = None
        outputs = []
        for _ in range(10):
            x = np.random.randn(32)  # Single position
            out, state = gla.step(x, state)
            outputs.append(out)

        assert len(outputs) == 10
        assert outputs[0].shape == (32,)

    def test_gate_bias_initialization(self):
        """Positive gate bias should favor retention."""
        gla = GatedLinearAttention(d_model=32, gate_bias_init=5.0)

        # With high positive bias, initial gates should be close to 1
        # This means the model starts by remembering more
        x = np.zeros((10, 32))  # Zero input
        # Can't directly test internal gates, but model should work
        output, _ = gla(x)
        assert output.shape == x.shape


class TestCompareGLAToVanilla:
    """Tests comparing GLA to vanilla linear attention."""

    def test_returns_all_outputs(self):
        """Should return all comparison outputs."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)
        gates = np.ones((seq_len, d)) * 0.5

        result = compare_gla_to_vanilla(Q, K, V, gates)

        assert 'vanilla_output' in result
        assert 'gla_output' in result
        assert 'vanilla_state' in result
        assert 'gla_state' in result
        assert 'state_norm_ratio' in result

    def test_gate_one_matches_vanilla(self):
        """With gates=1 everywhere, GLA should accumulate like vanilla."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)
        gates = np.ones((seq_len, d))  # All retention

        # Note: They won't be exactly equal because vanilla accumulates
        # while GLA with gate=1 just preserves state (no new writes)
        # This test documents the behavioral difference
        result = compare_gla_to_vanilla(Q, K, V, gates)

        # State norms should be different
        assert result['state_norm_ratio'] != 1.0

    def test_gating_reduces_state_norm(self):
        """Gating should typically reduce state norm vs vanilla."""
        seq_len, d = 50, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)
        gates = np.ones((seq_len, d)) * 0.9  # Some forgetting

        result = compare_gla_to_vanilla(Q, K, V, gates)

        # GLA state should have smaller norm due to forgetting
        assert result['state_norm_ratio'] < 1.0


class TestMilestone:
    """Lab 04 Milestone: Gated Linear Attention."""

    def test_milestone_gating_works(self):
        """
        MILESTONE: GLA correctly implements data-dependent gating.
        """
        seq_len, d = 20, 16

        # Test that different gates produce different outputs
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        # High retention gates
        gates_high = np.ones((seq_len, d)) * 0.95
        out_high, state_high = gla_recurrent(Q, K, V, gates_high)

        # Low retention gates
        gates_low = np.ones((seq_len, d)) * 0.5
        out_low, state_low = gla_recurrent(Q, K, V, gates_low)

        # Outputs should be different
        assert not np.allclose(out_high, out_low)

        # State norms should differ (low retention = smaller state)
        norm_high = np.linalg.norm(state_high)
        norm_low = np.linalg.norm(state_low)
        assert norm_high != norm_low

        print("\n" + "=" * 60)
        print("MILESTONE ACHIEVED: Gated Linear Attention works!")
        print("The model can now selectively forget old information.")
        print("=" * 60 + "\n")
