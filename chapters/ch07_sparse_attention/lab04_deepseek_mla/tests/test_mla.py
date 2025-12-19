"""Tests for Lab 04: DeepSeek MLA."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mla import (
    MultiHeadLatentAttention,
    compare_mha_vs_mla,
    MLAWithRoPE,
)


class TestMLAInit:
    """Tests for MLA initialization."""

    def test_basic_init(self):
        """Should initialize with valid parameters."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        assert mla.d_model == 512
        assert mla.num_heads == 8
        assert mla.d_latent == 128
        assert mla.head_dim == 64  # 512 // 8

    def test_custom_head_dim(self):
        """Should accept custom head_dim."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128,
            head_dim=32
        )
        assert mla.head_dim == 32

    def test_invalid_d_model(self):
        """Should raise error for non-divisible d_model."""
        with pytest.raises(ValueError):
            MultiHeadLatentAttention(
                d_model=500,  # Not divisible by 8
                num_heads=8,
                d_latent=128
            )

    def test_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        assert mla.W_Q.shape == (512, 512)  # d_model -> num_heads * head_dim
        assert mla.W_DKV.shape == (512, 128)  # d_model -> d_latent
        assert mla.W_UK.shape == (128, 512)  # d_latent -> num_heads * head_dim
        assert mla.W_UV.shape == (128, 512)  # d_latent -> num_heads * head_dim
        assert mla.W_O.shape == (512, 512)  # num_heads * head_dim -> d_model

    def test_weights_initialized(self):
        """Weights should be initialized (not zeros)."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        assert not np.allclose(mla.W_Q, 0)
        assert not np.allclose(mla.W_DKV, 0)
        assert not np.allclose(mla.W_UK, 0)


class TestMLACompression:
    """Tests for compression and decompression."""

    def test_compress_shape(self):
        """Compression should produce correct shape."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(10, 512).astype(np.float32)

        c_kv = mla.compress_kv(x)

        assert c_kv.shape == (10, 128)

    def test_compress_batched(self):
        """Compression should work with batched input."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        c_kv = mla.compress_kv(x)

        assert c_kv.shape == (2, 10, 128)

    def test_decompress_shape(self):
        """Decompression should produce correct shapes."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        c_kv = np.random.randn(10, 128).astype(np.float32)

        K, V = mla.decompress_kv(c_kv)

        assert K.shape == (10, 512)
        assert V.shape == (10, 512)

    def test_decompress_batched(self):
        """Decompression should work with batched input."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        c_kv = np.random.randn(2, 10, 128).astype(np.float32)

        K, V = mla.decompress_kv(c_kv)

        assert K.shape == (2, 10, 512)
        assert V.shape == (2, 10, 512)


class TestMLAForward:
    """Tests for MLA forward pass."""

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        output, cache = mla(x)

        assert output.shape == x.shape
        assert cache.shape == (2, 10, 128)

    def test_forward_unbatched(self):
        """Should handle unbatched input."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(10, 512).astype(np.float32)

        output, cache = mla(x)

        assert output.shape == x.shape
        assert cache.shape == (10, 128)

    def test_forward_deterministic(self):
        """Same input should give same output."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        out1, _ = mla(x)
        out2, _ = mla(x)

        np.testing.assert_array_equal(out1, out2)

    def test_cache_accumulation(self):
        """Cache should accumulate across calls."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        # Prefill
        x1 = np.random.randn(2, 10, 512).astype(np.float32)
        _, cache1 = mla(x1)
        assert cache1.shape == (2, 10, 128)

        # Decode step 1
        x2 = np.random.randn(2, 1, 512).astype(np.float32)
        _, cache2 = mla(x2, kv_cache=cache1)
        assert cache2.shape == (2, 11, 128)

        # Decode step 2
        x3 = np.random.randn(2, 1, 512).astype(np.float32)
        _, cache3 = mla(x3, kv_cache=cache2)
        assert cache3.shape == (2, 12, 128)

    def test_callable(self):
        """Should be callable like a function."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        out1, cache1 = mla.forward(x)
        out2, cache2 = mla(x)

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(cache1, cache2)


class TestMLAAttentionWeights:
    """Tests for attention weight extraction."""

    def test_weights_shape(self):
        """Attention weights should have correct shape."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        weights = mla.get_attention_weights(x)

        assert weights.shape == (2, 8, 10, 10)

    def test_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        weights = mla.get_attention_weights(x)

        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_weights_with_cache(self):
        """Weights shape should reflect total sequence with cache."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        # Prefill
        x1 = np.random.randn(10, 512).astype(np.float32)
        _, cache = mla(x1)

        # Decode with cache
        x2 = np.random.randn(1, 512).astype(np.float32)
        weights = mla.get_attention_weights(x2, kv_cache=cache)

        # Query is 1 token, Keys are 11 tokens (10 cached + 1 new)
        assert weights.shape == (8, 1, 11)


class TestMLAMemory:
    """Tests for memory efficiency."""

    def test_cache_size(self):
        """Cache size calculation should be correct."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        cache_bytes = mla.get_cache_size_bytes(
            seq_len=1000,
            batch_size=2,
            dtype_bytes=2
        )

        expected = 2 * 1000 * 128 * 2  # batch * seq * latent * dtype
        assert cache_bytes == expected

    def test_compression_ratio(self):
        """Compression ratio should be calculated correctly."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        ratio = mla.get_compression_ratio()

        # Standard: 2 * 8 * 64 = 1024 (K and V)
        # MLA: 128
        expected = (2 * 8 * 64) / 128
        assert ratio == expected
        assert ratio == 8.0

    def test_significant_compression(self):
        """MLA should provide significant compression."""
        mla = MultiHeadLatentAttention(
            d_model=4096,
            num_heads=32,
            d_latent=512
        )

        ratio = mla.get_compression_ratio()

        # Should be at least 4x compression
        assert ratio >= 4


class TestCompareMHAvsMLA:
    """Tests for compare_mha_vs_mla function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = compare_mha_vs_mla(
            d_model=512,
            num_heads=8,
            d_latent=128,
            seq_len=1000,
            num_layers=12
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Should contain all required keys."""
        result = compare_mha_vs_mla(
            d_model=512,
            num_heads=8,
            d_latent=128,
            seq_len=1000,
            num_layers=12
        )

        assert 'mha_cache_bytes' in result
        assert 'mla_cache_bytes' in result
        assert 'compression_ratio' in result
        assert 'mha_cache_gb' in result
        assert 'mla_cache_gb' in result

    def test_mla_smaller_than_mha(self):
        """MLA cache should be smaller than MHA."""
        result = compare_mha_vs_mla(
            d_model=512,
            num_heads=8,
            d_latent=128,
            seq_len=1000,
            num_layers=12
        )

        assert result['mla_cache_bytes'] < result['mha_cache_bytes']
        assert result['compression_ratio'] > 1

    def test_realistic_model(self):
        """Test with realistic model parameters."""
        result = compare_mha_vs_mla(
            d_model=4096,
            num_heads=32,
            d_latent=512,
            seq_len=32768,  # 32K context
            num_layers=60,
            dtype_bytes=2
        )

        # MHA should be many GB
        assert result['mha_cache_gb'] > 1

        # MLA should be much smaller
        assert result['compression_ratio'] >= 4


class TestHeadSplitCombine:
    """Tests for head splitting and combining."""

    def test_split_heads_shape(self):
        """Split heads should produce correct shape."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        split = mla._split_heads(x)

        assert split.shape == (2, 8, 10, 64)

    def test_split_heads_unbatched(self):
        """Split heads should work unbatched."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(10, 512).astype(np.float32)

        split = mla._split_heads(x)

        assert split.shape == (8, 10, 64)

    def test_combine_heads_shape(self):
        """Combine heads should produce correct shape."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 8, 10, 64).astype(np.float32)

        combined = mla._combine_heads(x)

        assert combined.shape == (2, 10, 512)

    def test_split_combine_roundtrip(self):
        """Split then combine should preserve shape."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        split = mla._split_heads(x)
        combined = mla._combine_heads(split)

        assert combined.shape == x.shape


class TestMLAMilestone:
    """Milestone tests for Chapter 7."""

    def test_4x_compression_achieved(self):
        """MLA should achieve at least 4x compression."""
        mla = MultiHeadLatentAttention(
            d_model=4096,
            num_heads=32,
            d_latent=512
        )

        ratio = mla.get_compression_ratio()
        assert ratio >= 4, f"Expected >=4x compression, got {ratio}x"

    def test_valid_attention_outputs(self):
        """Attention should produce valid outputs."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )
        x = np.random.randn(2, 20, 512).astype(np.float32)

        output, cache = mla(x)
        weights = mla.get_attention_weights(x)

        # Output shape correct
        assert output.shape == x.shape

        # Weights sum to 1
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

        # Weights non-negative
        assert np.all(weights >= 0)

    def test_incremental_decoding(self):
        """Incremental decoding should work correctly."""
        mla = MultiHeadLatentAttention(
            d_model=512,
            num_heads=8,
            d_latent=128
        )

        # Prefill
        x_prefill = np.random.randn(1, 100, 512).astype(np.float32)
        out_prefill, cache = mla(x_prefill)

        assert out_prefill.shape == (1, 100, 512)
        assert cache.shape == (1, 100, 128)

        # Decode 10 tokens one at a time
        for i in range(10):
            x_decode = np.random.randn(1, 1, 512).astype(np.float32)
            out_decode, cache = mla(x_decode, kv_cache=cache)

            assert out_decode.shape == (1, 1, 512)
            assert cache.shape == (1, 101 + i, 128)


class TestMLAWithRoPE:
    """Tests for MLA with RoPE (optional advanced feature)."""

    def test_init(self):
        """Should initialize with RoPE parameters."""
        try:
            mla_rope = MLAWithRoPE(
                d_model=512,
                num_heads=8,
                d_latent=128,
                rope_dim=32
            )
            assert mla_rope.rope_dim == 32
        except NotImplementedError:
            pytest.skip("MLAWithRoPE not implemented")

    def test_rope_default_dim(self):
        """RoPE dim should default to head_dim // 2."""
        try:
            mla_rope = MLAWithRoPE(
                d_model=512,
                num_heads=8,
                d_latent=128
            )
            assert mla_rope.rope_dim == 32  # 64 // 2
        except NotImplementedError:
            pytest.skip("MLAWithRoPE not implemented")
