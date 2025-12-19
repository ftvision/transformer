"""Tests for Lab 01: RNN View of Linear Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rnn_view import (
    feature_map,
    linear_attention_parallel,
    linear_attention_recurrent,
    linear_attention_step,
    compare_parallel_recurrent,
)


class TestFeatureMap:
    """Tests for the feature_map function."""

    def test_elu_plus_one_positive_input(self):
        """ELU+1 on positive input should return x + 1."""
        x = np.array([1.0, 2.0, 3.0])
        result = feature_map(x, 'elu_plus_one')

        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_elu_plus_one_negative_input(self):
        """ELU+1 on negative input should be positive."""
        x = np.array([-1.0, -2.0, -3.0])
        result = feature_map(x, 'elu_plus_one')

        # All outputs should be positive (> 0)
        assert np.all(result > 0)

    def test_elu_plus_one_zero(self):
        """ELU+1 at zero should return 1."""
        x = np.array([0.0])
        result = feature_map(x, 'elu_plus_one')

        np.testing.assert_allclose(result, [1.0])

    def test_relu(self):
        """ReLU should zero out negatives."""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = feature_map(x, 'relu')

        np.testing.assert_allclose(result, [0.0, 0.0, 1.0, 2.0])

    def test_identity(self):
        """Identity should return input unchanged."""
        x = np.array([-1.0, 0.0, 1.0])
        result = feature_map(x, 'identity')

        np.testing.assert_allclose(result, x)

    def test_preserves_shape(self):
        """Feature map should preserve input shape."""
        x = np.random.randn(2, 10, 64)
        result = feature_map(x, 'elu_plus_one')

        assert result.shape == x.shape

    def test_softmax_kernel(self):
        """Softmax kernel should compute exp(x)."""
        x = np.array([0.0, 1.0, 2.0])
        result = feature_map(x, 'softmax_kernel')

        np.testing.assert_allclose(result, np.exp(x))


class TestLinearAttentionParallel:
    """Tests for parallel linear attention."""

    def test_output_shape_2d(self):
        """Output should match V shape for 2D input."""
        seq_len, d_k, d_v = 10, 8, 16
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)

        output, state = linear_attention_parallel(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (d_k, d_v)

    def test_output_shape_3d(self):
        """Output should match V shape for 3D (batched) input."""
        batch, seq_len, d_k, d_v = 2, 10, 8, 16
        Q = np.random.randn(batch, seq_len, d_k)
        K = np.random.randn(batch, seq_len, d_k)
        V = np.random.randn(batch, seq_len, d_v)

        output, state = linear_attention_parallel(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (batch, d_k, d_v)

    def test_causal_property(self):
        """Output at position i should only depend on positions <= i."""
        seq_len, d = 5, 4
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        # Get full output
        output_full, _ = linear_attention_parallel(Q, K, V)

        # Get output with truncated sequence
        output_partial, _ = linear_attention_parallel(Q[:3], K[:3], V[:3])

        # First 3 positions should match
        np.testing.assert_allclose(output_full[:3], output_partial, rtol=1e-5)

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        seq_len, d = 10, 8
        Q1, K1, V1 = np.random.randn(3, seq_len, d)
        Q2, K2, V2 = np.random.randn(3, seq_len, d)

        out1, _ = linear_attention_parallel(Q1, K1, V1)
        out2, _ = linear_attention_parallel(Q2, K2, V2)

        assert not np.allclose(out1, out2)

    def test_returns_final_state(self):
        """Should return the final accumulated state."""
        seq_len, d_k, d_v = 10, 8, 16
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)

        _, state = linear_attention_parallel(Q, K, V)

        # State should be a d_k x d_v matrix
        assert state.shape == (d_k, d_v)
        # State should not be all zeros (accumulated information)
        assert not np.allclose(state, 0)


class TestLinearAttentionRecurrent:
    """Tests for recurrent linear attention."""

    def test_output_shape_2d(self):
        """Output should match V shape for 2D input."""
        seq_len, d_k, d_v = 10, 8, 16
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)

        output, state = linear_attention_recurrent(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (d_k, d_v)

    def test_output_shape_3d(self):
        """Output should match V shape for 3D (batched) input."""
        batch, seq_len, d_k, d_v = 2, 10, 8, 16
        Q = np.random.randn(batch, seq_len, d_k)
        K = np.random.randn(batch, seq_len, d_k)
        V = np.random.randn(batch, seq_len, d_v)

        output, state = linear_attention_recurrent(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (batch, d_k, d_v)

    def test_state_accumulates(self):
        """State should accumulate information from all positions."""
        seq_len, d_k, d_v = 10, 8, 16
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)

        _, final_state = linear_attention_recurrent(Q, K, V)

        # Final state is sum of all k^T @ v contributions
        # (after feature map)
        assert final_state.shape == (d_k, d_v)
        assert not np.allclose(final_state, 0)


class TestParallelRecurrentEquivalence:
    """Tests verifying parallel and recurrent forms are equivalent."""

    def test_outputs_match_2d(self):
        """Parallel and recurrent should give identical outputs (2D)."""
        seq_len, d_k, d_v = 10, 8, 16
        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_v).astype(np.float32)

        out_parallel, state_parallel = linear_attention_parallel(Q, K, V)
        out_recurrent, state_recurrent = linear_attention_recurrent(Q, K, V)

        np.testing.assert_allclose(out_parallel, out_recurrent, rtol=1e-5)
        np.testing.assert_allclose(state_parallel, state_recurrent, rtol=1e-5)

    def test_outputs_match_3d(self):
        """Parallel and recurrent should give identical outputs (3D batched)."""
        batch, seq_len, d_k, d_v = 2, 10, 8, 16
        Q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        K = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        V = np.random.randn(batch, seq_len, d_v).astype(np.float32)

        out_parallel, state_parallel = linear_attention_parallel(Q, K, V)
        out_recurrent, state_recurrent = linear_attention_recurrent(Q, K, V)

        np.testing.assert_allclose(out_parallel, out_recurrent, rtol=1e-5)
        np.testing.assert_allclose(state_parallel, state_recurrent, rtol=1e-5)

    def test_outputs_match_long_sequence(self):
        """Should match even for longer sequences."""
        seq_len, d = 100, 32
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        out_parallel, _ = linear_attention_parallel(Q, K, V)
        out_recurrent, _ = linear_attention_recurrent(Q, K, V)

        np.testing.assert_allclose(out_parallel, out_recurrent, rtol=1e-4)

    def test_compare_function(self):
        """compare_parallel_recurrent should correctly detect matches."""
        seq_len, d = 10, 8
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        match, max_diff = compare_parallel_recurrent(Q, K, V)

        assert match, f"Should match, but max_diff={max_diff}"
        assert max_diff < 1e-5


class TestLinearAttentionStep:
    """Tests for single-step linear attention."""

    def test_output_shape_1d(self):
        """Output should be (d_v,) for 1D input."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)

        output, state = linear_attention_step(q, k, v)

        assert output.shape == (d_v,)
        assert state.shape == (d_k, d_v)

    def test_output_shape_2d(self):
        """Output should be (batch, d_v) for 2D input."""
        batch, d_k, d_v = 2, 8, 16
        q = np.random.randn(batch, d_k)
        k = np.random.randn(batch, d_k)
        v = np.random.randn(batch, d_v)

        output, state = linear_attention_step(q, k, v)

        assert output.shape == (batch, d_v)
        assert state.shape == (batch, d_k, d_v)

    def test_state_initialized_to_zeros(self):
        """When state is None, should initialize to zeros."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)

        # First step with no state
        output1, state1 = linear_attention_step(q, k, v, state=None)

        # First step with explicit zero state
        zero_state = np.zeros((d_k, d_v))
        output2, state2 = linear_attention_step(q, k, v, state=zero_state)

        np.testing.assert_allclose(output1, output2)
        np.testing.assert_allclose(state1, state2)

    def test_state_accumulates(self):
        """State should grow as we add more steps."""
        d = 8
        state = None

        # Process several steps
        for _ in range(5):
            q = np.random.randn(d)
            k = np.random.randn(d)
            v = np.random.randn(d)
            _, state = linear_attention_step(q, k, v, state)

        # State should have accumulated information
        assert not np.allclose(state, 0)

    def test_matches_recurrent(self):
        """Step-by-step should match recurrent implementation."""
        seq_len, d_k, d_v = 10, 8, 16
        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_v).astype(np.float32)

        # Get output using recurrent (whole sequence)
        out_recurrent, state_recurrent = linear_attention_recurrent(Q, K, V)

        # Get output using step-by-step
        outputs_step = []
        state = None
        for i in range(seq_len):
            out, state = linear_attention_step(Q[i], K[i], V[i], state)
            outputs_step.append(out)

        out_step = np.stack(outputs_step, axis=0)

        np.testing.assert_allclose(out_step, out_recurrent, rtol=1e-5)
        np.testing.assert_allclose(state, state_recurrent, rtol=1e-5)


class TestStateProperties:
    """Tests for state matrix properties."""

    def test_state_is_matrix(self):
        """State should be a d_k x d_v matrix."""
        d_k, d_v = 8, 16
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        v = np.random.randn(d_v)

        _, state = linear_attention_step(q, k, v)

        assert state.ndim == 2
        assert state.shape == (d_k, d_v)

    def test_state_rank_one_update(self):
        """Each step adds a rank-1 outer product to state."""
        d_k, d_v = 8, 16
        k = np.random.randn(d_k).astype(np.float32)
        v = np.random.randn(d_v).astype(np.float32)
        q = np.random.randn(d_k).astype(np.float32)

        # With zero initial state, the state after one step
        # should be exactly φ(k)^T @ v (a rank-1 matrix)
        _, state = linear_attention_step(q, k, v, state=None)

        # Check it's approximately rank 1
        # (allow some tolerance due to feature map)
        u, s, vh = np.linalg.svd(state)
        # First singular value should dominate
        assert s[0] > 0.99 * s.sum(), "State should be approximately rank-1"

    def test_state_constant_size(self):
        """State size should be constant regardless of sequence length."""
        d_k, d_v = 8, 16
        state = None

        state_sizes = []
        for seq_len in [10, 100, 1000]:
            state = None
            for _ in range(seq_len):
                q = np.random.randn(d_k)
                k = np.random.randn(d_k)
                v = np.random.randn(d_v)
                _, state = linear_attention_step(q, k, v, state)

            state_sizes.append(state.size)

        # All state sizes should be the same
        assert len(set(state_sizes)) == 1, "State size should be constant"
        assert state_sizes[0] == d_k * d_v


class TestMilestone:
    """Lab 01 Milestone: Verify understanding of the RNN view."""

    def test_milestone_equivalence(self):
        """
        MILESTONE: Parallel and recurrent forms produce identical outputs.

        This verifies you understand that linear attention can be computed
        either way, which is the key insight for efficient training (parallel)
        and inference (recurrent).
        """
        # Test with various configurations
        configs = [
            (10, 8, 8),    # Small
            (50, 32, 32),  # Medium
            (2, 20, 16, 16),  # Batched
        ]

        for config in configs:
            if len(config) == 3:
                seq_len, d_k, d_v = config
                Q = np.random.randn(seq_len, d_k).astype(np.float32)
                K = np.random.randn(seq_len, d_k).astype(np.float32)
                V = np.random.randn(seq_len, d_v).astype(np.float32)
            else:
                batch, seq_len, d_k, d_v = config
                Q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
                K = np.random.randn(batch, seq_len, d_k).astype(np.float32)
                V = np.random.randn(batch, seq_len, d_v).astype(np.float32)

            match, max_diff = compare_parallel_recurrent(Q, K, V)
            assert match, f"Config {config} failed with max_diff={max_diff}"

        print("\n" + "=" * 60)
        print("MILESTONE ACHIEVED: RNN view understood!")
        print("Parallel and recurrent forms are equivalent.")
        print("=" * 60 + "\n")
