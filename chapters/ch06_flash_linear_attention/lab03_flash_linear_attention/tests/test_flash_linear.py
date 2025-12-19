"""Tests for Lab 03: Flash Linear Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flash_linear import (
    naive_linear_attention,
    tiled_forward,
    compute_memory_footprint,
    FlashLinearAttention,
    flash_vs_naive_comparison,
    optimal_chunk_size,
)


class TestNaiveLinearAttention:
    """Tests for the naive (reference) implementation."""

    def test_output_shape(self):
        """Output should match V shape."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        output, cumsum = naive_linear_attention(Q, K, V)

        assert output.shape == V.shape
        assert cumsum.shape == (seq_len, d, d)

    def test_cumsum_grows(self):
        """Cumsum tensor should accumulate information."""
        seq_len, d = 10, 8
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        _, cumsum = naive_linear_attention(Q, K, V)

        # Frobenius norm should generally increase
        norms = [np.linalg.norm(cumsum[i]) for i in range(seq_len)]
        assert norms[-1] > norms[0], "State should accumulate"


class TestTiledForward:
    """Tests for the tiled (memory-efficient) forward pass."""

    def test_output_shape_2d(self):
        """Output should match V shape (2D input)."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        output, saved_states, final_state = tiled_forward(Q, K, V, chunk_size=4)

        assert output.shape == V.shape
        assert final_state.shape == (d, d)

    def test_output_shape_3d(self):
        """Output should match V shape (3D batched input)."""
        batch, seq_len, d = 2, 20, 16
        Q = np.random.randn(batch, seq_len, d)
        K = np.random.randn(batch, seq_len, d)
        V = np.random.randn(batch, seq_len, d)

        output, saved_states, final_state = tiled_forward(Q, K, V, chunk_size=4)

        assert output.shape == V.shape
        assert final_state.shape == (batch, d, d)

    def test_matches_naive(self):
        """Tiled output should match naive output."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        naive_out, _ = naive_linear_attention(Q, K, V)
        tiled_out, _, _ = tiled_forward(Q, K, V, chunk_size=4)

        np.testing.assert_allclose(tiled_out, naive_out, rtol=1e-5)

    def test_saved_states_count(self):
        """Should save one state per chunk."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        _, saved_states, _ = tiled_forward(Q, K, V, chunk_size=4)

        # 20 / 4 = 5 chunks (exactly)
        assert len(saved_states) == 5

    def test_saved_states_count_with_padding(self):
        """Should handle non-divisible sequence lengths."""
        seq_len, d = 22, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        _, saved_states, _ = tiled_forward(Q, K, V, chunk_size=4)

        # ceil(22 / 4) = 6 chunks
        assert len(saved_states) == 6

    def test_no_save_option(self):
        """Should not save states when save_states=False."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        output, saved_states, _ = tiled_forward(
            Q, K, V, chunk_size=4, save_states=False
        )

        assert len(saved_states) == 0
        assert output.shape == V.shape


class TestMemoryFootprint:
    """Tests for memory footprint computation."""

    def test_naive_memory(self):
        """Naive should report large intermediate memory."""
        seq_len, d = 100, 64
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        stats = compute_memory_footprint(Q, K, V, method='naive')

        assert 'intermediate_bytes' in stats
        assert 'total_bytes' in stats
        # Naive stores cumsum: seq_len × d × d × bytes_per_float
        expected_cumsum_bytes = seq_len * d * d * 8  # float64
        assert stats['intermediate_bytes'] >= expected_cumsum_bytes * 0.9

    def test_tiled_memory_less(self):
        """Tiled should use less memory than naive."""
        seq_len, d = 100, 64
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        naive_stats = compute_memory_footprint(Q, K, V, method='naive')
        tiled_stats = compute_memory_footprint(Q, K, V, method='tiled')

        assert tiled_stats['intermediate_bytes'] < naive_stats['intermediate_bytes']


class TestFlashLinearAttention:
    """Tests for the FlashLinearAttention class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        fla = FlashLinearAttention(chunk_size=32, feature_map='elu_plus_one')

        assert fla.chunk_size == 32

    def test_forward(self):
        """Forward should produce correct output."""
        fla = FlashLinearAttention(chunk_size=4)

        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        output = fla(Q, K, V)
        naive_out, _ = naive_linear_attention(Q, K, V)

        np.testing.assert_allclose(output, naive_out, rtol=1e-5)

    def test_saved_states(self):
        """Should save states for backward pass."""
        fla = FlashLinearAttention(chunk_size=4)

        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        _ = fla(Q, K, V)
        states = fla.get_saved_states()

        assert len(states) == 5  # 20 / 4 = 5 chunks

    def test_memory_stats(self):
        """Should track memory statistics."""
        fla = FlashLinearAttention(chunk_size=4)

        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        _ = fla(Q, K, V)
        stats = fla.get_memory_stats()

        assert 'num_chunks' in stats
        assert stats['num_chunks'] == 5

    def test_callable(self):
        """Should be callable like a function."""
        fla = FlashLinearAttention(chunk_size=4)

        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        out1 = fla.forward(Q, K, V)
        out2 = fla(Q, K, V)

        np.testing.assert_array_equal(out1, out2)


class TestFlashVsNaiveComparison:
    """Tests for the comparison function."""

    def test_outputs_match(self):
        """Flash and naive should produce matching outputs."""
        seq_len, d = 50, 32
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        result = flash_vs_naive_comparison(Q, K, V, chunk_size=8)

        assert result['outputs_match']
        assert result['max_diff'] < 1e-5

    def test_memory_savings(self):
        """Flash should use less memory than naive."""
        seq_len, d = 100, 64
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        result = flash_vs_naive_comparison(Q, K, V, chunk_size=8)

        assert result['flash_memory'] < result['naive_memory']
        assert result['memory_ratio'] > 1.0  # Flash uses less

    def test_returns_all_stats(self):
        """Should return all required statistics."""
        seq_len, d = 20, 16
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)

        result = flash_vs_naive_comparison(Q, K, V)

        required_keys = [
            'outputs_match', 'max_diff',
            'naive_memory', 'flash_memory', 'memory_ratio'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestOptimalChunkSize:
    """Tests for optimal chunk size computation."""

    def test_respects_memory_constraint(self):
        """Chunk size should fit in available memory."""
        seq_len, d_k, d_v = 1000, 64, 64
        available = 1024 * 1024  # 1 MB

        chunk_size = optimal_chunk_size(seq_len, d_k, d_v, available)

        # Working memory ≈ chunk_size × d_k × d_v × 8 bytes
        working_memory = chunk_size * d_k * d_v * 8
        assert working_memory < available

    def test_returns_reasonable_size(self):
        """Should return a reasonable chunk size."""
        seq_len, d_k, d_v = 1000, 64, 64
        available = 10 * 1024 * 1024  # 10 MB

        chunk_size = optimal_chunk_size(seq_len, d_k, d_v, available)

        assert chunk_size >= 1
        assert chunk_size <= seq_len

    def test_scales_with_dimension(self):
        """Larger dimensions should give smaller chunk sizes."""
        seq_len = 1000
        available = 1024 * 1024  # 1 MB

        chunk_small = optimal_chunk_size(seq_len, 32, 32, available)
        chunk_large = optimal_chunk_size(seq_len, 128, 128, available)

        assert chunk_large <= chunk_small


class TestMilestone:
    """Lab 03 Milestone: Memory-efficient linear attention."""

    def test_milestone_correctness(self):
        """
        MILESTONE: Flash Linear Attention produces correct outputs.
        """
        configs = [
            (50, 32),
            (100, 64),
            (200, 32),
        ]

        for seq_len, d in configs:
            Q = np.random.randn(seq_len, d).astype(np.float32)
            K = np.random.randn(seq_len, d).astype(np.float32)
            V = np.random.randn(seq_len, d).astype(np.float32)

            result = flash_vs_naive_comparison(Q, K, V, chunk_size=8)
            assert result['outputs_match'], \
                f"Config ({seq_len}, {d}): max_diff={result['max_diff']}"

    def test_milestone_memory_efficiency(self):
        """
        MILESTONE: Flash Linear Attention uses less memory than naive.
        """
        seq_len, d = 256, 64
        Q = np.random.randn(seq_len, d).astype(np.float32)
        K = np.random.randn(seq_len, d).astype(np.float32)
        V = np.random.randn(seq_len, d).astype(np.float32)

        result = flash_vs_naive_comparison(Q, K, V, chunk_size=16)

        assert result['memory_ratio'] > 2.0, \
            f"Expected >2x memory savings, got {result['memory_ratio']:.1f}x"

        print("\n" + "=" * 60)
        print("MILESTONE ACHIEVED: Flash Linear Attention works!")
        print(f"Memory ratio: {result['memory_ratio']:.1f}x savings")
        print(f"Max output diff: {result['max_diff']:.2e}")
        print("=" * 60 + "\n")
