"""Tests for Lab 02: Chunkwise Parallel Linear Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunkwise import (
    chunk_sequence,
    unchunk_sequence,
    intra_chunk_attention,
    inter_chunk_contribution,
    chunkwise_linear_attention,
    compare_chunkwise_to_full,
)


class TestChunkSequence:
    """Tests for sequence chunking."""

    def test_exact_division(self):
        """Sequence length exactly divisible by chunk size."""
        x = np.arange(12).reshape(12, 1)
        chunks, orig_len = chunk_sequence(x, chunk_size=4)

        assert chunks.shape == (3, 4, 1)  # 12 / 4 = 3 chunks
        assert orig_len == 12

    def test_with_padding(self):
        """Sequence needs padding."""
        x = np.arange(10).reshape(10, 1)
        chunks, orig_len = chunk_sequence(x, chunk_size=4)

        # 10 -> 12 (padded), 12 / 4 = 3 chunks
        assert chunks.shape == (3, 4, 1)
        assert orig_len == 10

    def test_padding_values(self):
        """Padded values should be pad_value."""
        x = np.ones((10, 1))
        chunks, _ = chunk_sequence(x, chunk_size=4, pad_value=0.0)

        # Last chunk should have 2 padded zeros
        assert chunks[2, 2, 0] == 0.0  # Position 10 (padded)
        assert chunks[2, 3, 0] == 0.0  # Position 11 (padded)
        assert chunks[2, 0, 0] == 1.0  # Position 8 (original)
        assert chunks[2, 1, 0] == 1.0  # Position 9 (original)

    def test_batched_input(self):
        """Should handle batched input."""
        x = np.random.randn(2, 10, 8)
        chunks, orig_len = chunk_sequence(x, chunk_size=4)

        assert chunks.shape == (2, 3, 4, 8)  # (batch, chunks, chunk_size, d)
        assert orig_len == 10

    def test_preserves_values(self):
        """Original values should be preserved."""
        x = np.arange(10).reshape(10, 1).astype(float)
        chunks, _ = chunk_sequence(x, chunk_size=4)

        # Check first chunk
        np.testing.assert_array_equal(chunks[0, :, 0], [0, 1, 2, 3])
        # Check second chunk
        np.testing.assert_array_equal(chunks[1, :, 0], [4, 5, 6, 7])


class TestUnchunkSequence:
    """Tests for unchunking."""

    def test_roundtrip_exact(self):
        """chunk then unchunk should give back original (exact division)."""
        x = np.random.randn(12, 8)
        chunks, orig_len = chunk_sequence(x, chunk_size=4)
        recovered = unchunk_sequence(chunks, orig_len)

        np.testing.assert_array_equal(recovered, x)

    def test_roundtrip_padded(self):
        """chunk then unchunk should give back original (with padding)."""
        x = np.random.randn(10, 8)
        chunks, orig_len = chunk_sequence(x, chunk_size=4)
        recovered = unchunk_sequence(chunks, orig_len)

        np.testing.assert_array_equal(recovered, x)

    def test_roundtrip_batched(self):
        """Should work for batched input."""
        x = np.random.randn(2, 10, 8)
        chunks, orig_len = chunk_sequence(x, chunk_size=4)
        recovered = unchunk_sequence(chunks, orig_len)

        np.testing.assert_array_equal(recovered, x)


class TestIntraChunkAttention:
    """Tests for intra-chunk attention."""

    def test_output_shape(self):
        """Output should match V_chunk shape."""
        chunk_size, d_k, d_v = 4, 8, 16
        Q = np.random.randn(chunk_size, d_k)
        K = np.random.randn(chunk_size, d_k)
        V = np.random.randn(chunk_size, d_v)

        output, state = intra_chunk_attention(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (d_k, d_v)

    def test_output_shape_batched(self):
        """Should handle batched input."""
        batch, chunk_size, d_k, d_v = 2, 4, 8, 16
        Q = np.random.randn(batch, chunk_size, d_k)
        K = np.random.randn(batch, chunk_size, d_k)
        V = np.random.randn(batch, chunk_size, d_v)

        output, state = intra_chunk_attention(Q, K, V)

        assert output.shape == V.shape
        assert state.shape == (batch, d_k, d_v)

    def test_causal_within_chunk(self):
        """Attention should be causal within the chunk."""
        chunk_size, d = 4, 8
        Q = np.random.randn(chunk_size, d)
        K = np.random.randn(chunk_size, d)
        V = np.random.randn(chunk_size, d)

        # Full chunk
        output_full, _ = intra_chunk_attention(Q, K, V)

        # Truncated chunk (first 2 positions)
        output_partial, _ = intra_chunk_attention(Q[:2], K[:2], V[:2])

        # First 2 positions should match
        np.testing.assert_allclose(output_full[:2], output_partial, rtol=1e-5)

    def test_state_is_sum(self):
        """Chunk state should be sum of outer products."""
        chunk_size, d_k, d_v = 4, 8, 16
        Q = np.random.randn(chunk_size, d_k)
        K = np.random.randn(chunk_size, d_k)
        V = np.random.randn(chunk_size, d_v)

        _, state = intra_chunk_attention(Q, K, V)

        # State should not be zero
        assert not np.allclose(state, 0)
        # State should be d_k x d_v
        assert state.shape == (d_k, d_v)


class TestInterChunkContribution:
    """Tests for inter-chunk contribution."""

    def test_output_shape(self):
        """Output should match expected shape."""
        chunk_size, d_k, d_v = 4, 8, 16
        Q = np.random.randn(chunk_size, d_k)
        state = np.random.randn(d_k, d_v)

        output = inter_chunk_contribution(Q, state)

        assert output.shape == (chunk_size, d_v)

    def test_output_shape_batched(self):
        """Should handle batched input."""
        batch, chunk_size, d_k, d_v = 2, 4, 8, 16
        Q = np.random.randn(batch, chunk_size, d_k)
        state = np.random.randn(batch, d_k, d_v)

        output = inter_chunk_contribution(Q, state)

        assert output.shape == (batch, chunk_size, d_v)

    def test_zero_state_zero_output(self):
        """Zero state should give zero contribution."""
        chunk_size, d_k, d_v = 4, 8, 16
        Q = np.random.randn(chunk_size, d_k)
        state = np.zeros((d_k, d_v))

        output = inter_chunk_contribution(Q, state)

        np.testing.assert_allclose(output, 0, atol=1e-10)

    def test_all_positions_use_same_state(self):
        """All positions should query the same state (just different q)."""
        chunk_size, d_k, d_v = 4, 8, 16
        Q = np.random.randn(chunk_size, d_k)
        state = np.random.randn(d_k, d_v)

        output = inter_chunk_contribution(Q, state)

        # Output at each position should be φ(q_i) @ state
        # Different positions have different q, so outputs differ
        assert not np.allclose(output[0], output[1])


class TestChunkwiseLinearAttention:
    """Tests for the full chunkwise algorithm."""

    def test_output_shape_2d(self):
        """Output should match V shape."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        output, state = chunkwise_linear_attention(Q, K, V, chunk_size=4)

        assert output.shape == V.shape
        assert state.shape == (d, d)

    def test_output_shape_3d(self):
        """Should handle batched input."""
        batch, seq_len, d = 2, 20, 16
        Q = np.random.randn(batch, seq_len, d)
        K = np.random.randn(batch, seq_len, d)
        V = np.random.randn(batch, seq_len, d)

        output, state = chunkwise_linear_attention(Q, K, V, chunk_size=4)

        assert output.shape == V.shape
        assert state.shape == (batch, d, d)

    def test_different_chunk_sizes(self):
        """Should work with various chunk sizes."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        outputs = []
        for chunk_size in [2, 4, 5, 10, 20]:
            out, _ = chunkwise_linear_attention(Q, K, V, chunk_size=chunk_size)
            outputs.append(out)

        # All should give the same result
        for i in range(1, len(outputs)):
            np.testing.assert_allclose(outputs[0], outputs[i], rtol=1e-4)

    def test_chunk_size_larger_than_seq(self):
        """Should work when chunk_size > seq_len."""
        seq_len, d = 10, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        output, _ = chunkwise_linear_attention(Q, K, V, chunk_size=100)

        assert output.shape == V.shape

    def test_causal_property(self):
        """Output at position i should only depend on positions <= i."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        # Full sequence
        out_full, _ = chunkwise_linear_attention(Q, K, V, chunk_size=4)

        # Truncated sequence
        out_partial, _ = chunkwise_linear_attention(Q[:10], K[:10], V[:10], chunk_size=4)

        # First 10 positions should match
        np.testing.assert_allclose(out_full[:10], out_partial, rtol=1e-5)


class TestChunkwiseEquivalence:
    """Tests verifying chunkwise matches full linear attention."""

    def test_matches_full_attention_2d(self):
        """Chunkwise should match full linear attention (2D)."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        match, max_diff = compare_chunkwise_to_full(Q, K, V, chunk_size=4)

        assert match, f"Should match, but max_diff={max_diff}"

    def test_matches_full_attention_3d(self):
        """Chunkwise should match full linear attention (3D batched)."""
        batch, seq_len, d = 2, 20, 16
        Q = np.random.randn(batch, seq_len, d).astype(np.float32)
        K = np.random.randn(batch, seq_len, d).astype(np.float32)
        V = np.random.randn(batch, seq_len, d).astype(np.float32)

        match, max_diff = compare_chunkwise_to_full(Q, K, V, chunk_size=4)

        assert match, f"Should match, but max_diff={max_diff}"

    def test_matches_across_chunk_sizes(self):
        """Should match full attention for various chunk sizes."""
        seq_len, d = 30, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        for chunk_size in [3, 5, 7, 10, 15]:
            match, max_diff = compare_chunkwise_to_full(
                Q, K, V, chunk_size=chunk_size
            )
            assert match, f"chunk_size={chunk_size} failed with max_diff={max_diff}"


class TestEfficiency:
    """Tests for efficiency properties (memory/compute characteristics)."""

    def test_state_size_constant(self):
        """State size should be constant regardless of sequence length."""
        d = 16

        state_sizes = []
        for seq_len in [10, 50, 100, 500]:
            Q = np.random.randn(seq_len, d)
            K = np.random.randn(seq_len, d)
            V = np.random.randn(seq_len, d)

            _, state = chunkwise_linear_attention(Q, K, V, chunk_size=8)
            state_sizes.append(state.size)

        # All should be the same (d × d)
        assert len(set(state_sizes)) == 1
        assert state_sizes[0] == d * d


class TestMilestone:
    """Lab 02 Milestone: Chunkwise parallel algorithm."""

    def test_milestone_correctness(self):
        """
        MILESTONE: Chunkwise algorithm matches full linear attention.

        This verifies you understand how to split computation into chunks
        while maintaining mathematical equivalence.
        """
        configs = [
            (30, 16, 4),   # (seq_len, d, chunk_size)
            (50, 32, 8),
            (100, 16, 16),
        ]

        for seq_len, d, chunk_size in configs:
            Q = np.random.randn(seq_len, d).astype(np.float32)
            K = np.random.randn(seq_len, d).astype(np.float32)
            V = np.random.randn(seq_len, d).astype(np.float32)

            match, max_diff = compare_chunkwise_to_full(Q, K, V, chunk_size)
            assert match, f"Config ({seq_len}, {d}, {chunk_size}) failed: max_diff={max_diff}"

        print("\n" + "=" * 60)
        print("MILESTONE ACHIEVED: Chunkwise parallel algorithm works!")
        print("This is the foundation for Flash Linear Attention.")
        print("=" * 60 + "\n")
