"""Tests for Lab 03: Tiled Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiled_attention import (
    TiledAttention,
    standard_attention,
    count_blocks,
    verify_memory_usage,
)


class TestTiledInit:
    """Tests for TiledAttention initialization."""

    def test_default_block_sizes(self):
        """Should use default block sizes."""
        tiled = TiledAttention()
        assert tiled.block_size_q == 32
        assert tiled.block_size_kv == 32

    def test_custom_block_sizes(self):
        """Should accept custom block sizes."""
        tiled = TiledAttention(block_size_q=64, block_size_kv=128)
        assert tiled.block_size_q == 64
        assert tiled.block_size_kv == 128


class TestTiledCorrectness:
    """Tests verifying tiled attention matches standard attention."""

    def test_matches_standard_simple(self):
        """Tiled attention should match standard for simple input."""
        np.random.seed(42)
        seq_len, d_k = 32, 16

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        tiled_out = tiled.forward(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)

    def test_matches_standard_larger(self):
        """Tiled attention should match for larger input."""
        np.random.seed(123)
        seq_len, d_k = 128, 64

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=32, block_size_kv=32)
        tiled_out = tiled.forward(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)

    def test_matches_different_block_sizes(self):
        """Should work with different Q and KV block sizes."""
        np.random.seed(456)
        seq_len, d_k = 64, 32

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=16, block_size_kv=32)
        tiled_out = tiled.forward(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)

    def test_non_divisible_seq_len(self):
        """Should handle seq_len not divisible by block_size."""
        np.random.seed(789)
        seq_len, d_k = 50, 16  # 50 not divisible by 32

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=32, block_size_kv=32)
        tiled_out = tiled.forward(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)

    def test_callable(self):
        """Should be callable like a function."""
        np.random.seed(42)
        Q = np.random.randn(32, 16).astype(np.float32)
        K = np.random.randn(32, 16).astype(np.float32)
        V = np.random.randn(32, 16).astype(np.float32)

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        out1 = tiled.forward(Q, K, V)
        out2 = tiled(Q, K, V)

        np.testing.assert_array_equal(out1, out2)


class TestTiledWithMask:
    """Tests for tiled attention with masking."""

    def test_with_mask(self):
        """Should handle attention mask correctly."""
        np.random.seed(42)
        seq_len, d_k = 32, 16

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        # Random mask
        mask = np.random.rand(seq_len, seq_len) > 0.5

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        tiled_out = tiled.forward(Q, K, V, mask=mask)
        standard_out, _ = standard_attention(Q, K, V, mask=mask)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)

    def test_causal_mask(self):
        """Should handle causal mask correctly."""
        np.random.seed(123)
        seq_len, d_k = 32, 16

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        tiled_out = tiled.forward(Q, K, V, mask=causal_mask)
        standard_out, _ = standard_attention(Q, K, V, mask=causal_mask)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)


class TestCausalOptimization:
    """Tests for causal attention with block skipping."""

    def test_causal_matches_standard(self):
        """Causal tiled should match standard causal attention."""
        np.random.seed(42)
        seq_len, d_k = 64, 32

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        tiled = TiledAttention(block_size_q=16, block_size_kv=16)
        tiled_out = tiled.forward_causal(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V, mask=causal_mask)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)


class TestBlockCounting:
    """Tests for block counting utility."""

    def test_full_attention_blocks(self):
        """Full attention should compute all blocks."""
        # 128 / 32 = 4 blocks in each dimension
        # 4 * 4 = 16 blocks total
        count = count_blocks(128, 128, 32, 32, causal=False)
        assert count == 16

    def test_causal_attention_blocks(self):
        """Causal attention should skip upper triangle blocks."""
        # For 4x4 grid with causal masking:
        # Row 0: 1 block (only diagonal and below)
        # Row 1: 2 blocks
        # Row 2: 3 blocks
        # Row 3: 4 blocks
        # Total: 1 + 2 + 3 + 4 = 10 blocks
        count = count_blocks(128, 128, 32, 32, causal=True)
        assert count == 10

    def test_causal_saves_blocks(self):
        """Causal should compute fewer blocks than full attention."""
        full_count = count_blocks(256, 256, 32, 32, causal=False)
        causal_count = count_blocks(256, 256, 32, 32, causal=True)

        assert causal_count < full_count
        # For large sequences, causal is about half
        assert causal_count < full_count * 0.6


class TestMemoryUsage:
    """Tests for memory usage verification."""

    def test_memory_reduction(self):
        """Tiled attention should use less memory than standard."""
        result = verify_memory_usage(1024, 64, block_size=32)

        assert 'standard_attention_matrix' in result
        assert 'tiled_max_block' in result
        assert 'memory_reduction' in result

        # Tiled should use much less memory
        assert result['memory_reduction'] > 10, (
            "Tiled should reduce memory by at least 10x for large sequences"
        )

    def test_memory_values(self):
        """Memory values should be correct."""
        seq_len, d_model, block_size = 512, 64, 32

        result = verify_memory_usage(seq_len, d_model, block_size)

        # Standard: seq_len * seq_len * 4 bytes
        expected_standard = seq_len * seq_len * 4
        assert result['standard_attention_matrix'] == expected_standard

        # Tiled: at most block_size_q * block_size_kv * 4 bytes
        expected_tiled = block_size * block_size * 4
        assert result['tiled_max_block'] == expected_tiled


class TestBatched:
    """Tests for batched attention."""

    def test_batched_attention(self):
        """Should handle batched inputs."""
        np.random.seed(42)
        batch, seq_len, d_k = 2, 32, 16

        Q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        K = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        V = np.random.randn(batch, seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        tiled_out = tiled.forward(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)

    def test_batched_output_shape(self):
        """Batched output should have correct shape."""
        batch, seq_len, d_k, d_v = 4, 64, 32, 32

        Q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        K = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        V = np.random.randn(batch, seq_len, d_v).astype(np.float32)

        tiled = TiledAttention(block_size_q=16, block_size_kv=16)
        output = tiled.forward(Q, K, V)

        assert output.shape == (batch, seq_len, d_v)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_dimension(self):
        """Should handle large key dimension (high variance scores)."""
        np.random.seed(42)
        seq_len, d_k = 32, 256  # Large d_k

        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        tiled_out = tiled.forward(Q, K, V)

        assert not np.any(np.isnan(tiled_out)), "Should not produce NaN"
        assert not np.any(np.isinf(tiled_out)), "Should not produce Inf"

    def test_scaled_inputs(self):
        """Should handle scaled inputs."""
        np.random.seed(123)
        seq_len, d_k = 32, 16

        # Scale up inputs
        Q = np.random.randn(seq_len, d_k).astype(np.float32) * 10
        K = np.random.randn(seq_len, d_k).astype(np.float32) * 10
        V = np.random.randn(seq_len, d_k).astype(np.float32)

        tiled = TiledAttention(block_size_q=8, block_size_kv=8)
        tiled_out = tiled.forward(Q, K, V)
        standard_out, _ = standard_attention(Q, K, V)

        np.testing.assert_allclose(tiled_out, standard_out, rtol=1e-4, atol=1e-5)
