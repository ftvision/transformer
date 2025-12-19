"""Tests for Lab 03: KV Compression."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_compression import (
    compute_kv_cache_size,
    compute_compression_ratio,
    grouped_query_attention,
    compress_kv,
    decompress_kv,
    GroupedQueryAttention,
    LowRankKVAttention,
    compare_memory_usage,
)


class TestComputeKVCacheSize:
    """Tests for compute_kv_cache_size function."""

    def test_basic_calculation(self):
        """Basic cache size calculation."""
        # 2 * layers * seq * heads * dim * dtype
        size = compute_kv_cache_size(
            seq_len=1000,
            num_layers=10,
            num_kv_heads=8,
            head_dim=64,
            dtype_bytes=2
        )
        expected = 2 * 10 * 1000 * 8 * 64 * 2
        assert size == expected

    def test_llama2_7b_style(self):
        """Verify with Llama-2 7B style parameters."""
        size = compute_kv_cache_size(
            seq_len=4096,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            dtype_bytes=2
        )
        # Should be ~2GB
        assert size == 2 * 32 * 4096 * 32 * 128 * 2
        assert size == 2147483648  # 2GB

    def test_fp32(self):
        """Should work with fp32."""
        fp16_size = compute_kv_cache_size(1000, 10, 8, 64, dtype_bytes=2)
        fp32_size = compute_kv_cache_size(1000, 10, 8, 64, dtype_bytes=4)
        assert fp32_size == 2 * fp16_size


class TestCompressionRatio:
    """Tests for compute_compression_ratio function."""

    def test_gqa_compression(self):
        """GQA compression ratio."""
        # 32 heads -> 4 heads = 8x reduction
        ratio = compute_compression_ratio(32, 128, compressed_num_kv_heads=4)
        assert ratio == 8.0

    def test_lowrank_compression(self):
        """Low-rank compression ratio."""
        # 32 heads * 128 dim = 4096 -> 512 = 8x reduction
        ratio = compute_compression_ratio(32, 128, compressed_dim=512)
        assert ratio == 8.0

    def test_no_compression(self):
        """No compression should give ratio 1."""
        ratio = compute_compression_ratio(32, 128, compressed_num_kv_heads=32)
        assert ratio == 1.0


class TestGroupedQueryAttention:
    """Tests for grouped_query_attention function."""

    def test_output_shape(self):
        """Output should have correct shape."""
        num_q_heads, num_kv_heads = 8, 2
        seq_len, head_dim = 10, 64

        Q = np.random.randn(num_q_heads, seq_len, head_dim)
        K = np.random.randn(num_kv_heads, seq_len, head_dim)
        V = np.random.randn(num_kv_heads, seq_len, head_dim)

        output, weights = grouped_query_attention(Q, K, V, num_kv_groups=4)

        assert output.shape == (num_q_heads, seq_len, head_dim)
        assert weights.shape == (num_q_heads, seq_len, seq_len)

    def test_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        Q = np.random.randn(8, 10, 64)
        K = np.random.randn(2, 10, 64)
        V = np.random.randn(2, 10, 64)

        _, weights = grouped_query_attention(Q, K, V, num_kv_groups=4)

        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_batched(self):
        """Should work with batched input."""
        Q = np.random.randn(2, 8, 10, 64)
        K = np.random.randn(2, 2, 10, 64)
        V = np.random.randn(2, 2, 10, 64)

        output, weights = grouped_query_attention(Q, K, V, num_kv_groups=4)

        assert output.shape == (2, 8, 10, 64)
        assert weights.shape == (2, 8, 10, 10)

    def test_same_as_mha_when_equal_heads(self):
        """GQA with equal heads should behave like MHA."""
        Q = np.random.randn(8, 10, 64)
        K = np.random.randn(8, 10, 64)
        V = np.random.randn(8, 10, 64)

        output_gqa, _ = grouped_query_attention(Q, K, V, num_kv_groups=1)

        # Should be same shape and similar behavior
        assert output_gqa.shape == Q.shape


class TestCompressDecompress:
    """Tests for compress_kv and decompress_kv functions."""

    def test_compress_shape(self):
        """Compression should produce correct shape."""
        kv_dim, d_latent = 64, 16
        K = np.random.randn(10, kv_dim)
        V = np.random.randn(10, kv_dim)
        W_down = np.random.randn(2 * kv_dim, d_latent)

        kv_latent = compress_kv(K, V, W_down)

        assert kv_latent.shape == (10, d_latent)

    def test_decompress_shape(self):
        """Decompression should produce correct shape."""
        d_latent, kv_dim = 16, 64
        kv_latent = np.random.randn(10, d_latent)
        W_up_k = np.random.randn(d_latent, kv_dim)
        W_up_v = np.random.randn(d_latent, kv_dim)

        K, V = decompress_kv(kv_latent, W_up_k, W_up_v)

        assert K.shape == (10, kv_dim)
        assert V.shape == (10, kv_dim)

    def test_batched_compress(self):
        """Compression should work with batched input."""
        K = np.random.randn(2, 10, 64)
        V = np.random.randn(2, 10, 64)
        W_down = np.random.randn(128, 16)

        kv_latent = compress_kv(K, V, W_down)

        assert kv_latent.shape == (2, 10, 16)

    def test_batched_decompress(self):
        """Decompression should work with batched input."""
        kv_latent = np.random.randn(2, 10, 16)
        W_up_k = np.random.randn(16, 64)
        W_up_v = np.random.randn(16, 64)

        K, V = decompress_kv(kv_latent, W_up_k, W_up_v)

        assert K.shape == (2, 10, 64)
        assert V.shape == (2, 10, 64)


class TestGQAClass:
    """Tests for GroupedQueryAttention class."""

    def test_init_valid(self):
        """Should initialize with valid parameters."""
        gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)
        assert gqa.d_model == 64
        assert gqa.num_q_heads == 8
        assert gqa.num_kv_heads == 2
        assert gqa.num_kv_groups == 4

    def test_init_invalid_divisibility(self):
        """Should raise error if num_q_heads not divisible by num_kv_heads."""
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=3)

    def test_init_invalid_d_model(self):
        """Should raise error if d_model not divisible by num_q_heads."""
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=65, num_q_heads=8, num_kv_heads=2)

    def test_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)

        assert gqa.W_Q.shape == (64, 64)  # Full Q projection
        assert gqa.W_K.shape == (64, 16)  # Reduced KV projection (2 heads * 8 dim)
        assert gqa.W_V.shape == (64, 16)
        assert gqa.W_O.shape == (64, 64)

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)
        x = np.random.randn(2, 10, 64).astype(np.float32)

        output = gqa(x)

        assert output.shape == x.shape

    def test_forward_unbatched(self):
        """Should handle unbatched input."""
        gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)
        x = np.random.randn(10, 64).astype(np.float32)

        output = gqa(x)

        assert output.shape == x.shape

    def test_kv_cache_size_smaller(self):
        """GQA should have smaller KV cache than MHA."""
        gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)
        gqa_cache = gqa.get_kv_cache_size(seq_len=1000, dtype_bytes=2)

        # MHA would have 8 heads, GQA has 2 heads
        # Cache should be 4x smaller
        mha_cache = 2 * 1000 * 8 * 8 * 2  # 2 * seq * heads * head_dim * dtype
        expected_gqa = 2 * 1000 * 2 * 8 * 2

        assert gqa_cache == expected_gqa
        assert gqa_cache < mha_cache


class TestLowRankKVAttention:
    """Tests for LowRankKVAttention class."""

    def test_init_valid(self):
        """Should initialize with valid parameters."""
        lowrank = LowRankKVAttention(d_model=64, num_heads=8, d_latent=16)
        assert lowrank.d_model == 64
        assert lowrank.num_heads == 8
        assert lowrank.d_latent == 16

    def test_init_invalid_d_model(self):
        """Should raise error if d_model not divisible by num_heads."""
        with pytest.raises(ValueError):
            LowRankKVAttention(d_model=65, num_heads=8, d_latent=16)

    def test_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        lowrank = LowRankKVAttention(d_model=64, num_heads=8, d_latent=16)

        assert lowrank.W_Q.shape == (64, 64)
        assert lowrank.W_down.shape == (64, 16)  # Compress
        assert lowrank.W_up_k.shape == (16, 64)  # Decompress K
        assert lowrank.W_up_v.shape == (16, 64)  # Decompress V
        assert lowrank.W_O.shape == (64, 64)

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        lowrank = LowRankKVAttention(d_model=64, num_heads=8, d_latent=16)
        x = np.random.randn(2, 10, 64).astype(np.float32)

        output, cache = lowrank(x)

        assert output.shape == x.shape
        assert cache.shape == (2, 10, 16)  # Latent cache

    def test_forward_unbatched(self):
        """Should handle unbatched input."""
        lowrank = LowRankKVAttention(d_model=64, num_heads=8, d_latent=16)
        x = np.random.randn(10, 64).astype(np.float32)

        output, cache = lowrank(x)

        assert output.shape == x.shape
        assert cache.shape == (10, 16)

    def test_cache_accumulation(self):
        """KV cache should accumulate across calls."""
        lowrank = LowRankKVAttention(d_model=64, num_heads=8, d_latent=16)

        x1 = np.random.randn(5, 64).astype(np.float32)
        x2 = np.random.randn(1, 64).astype(np.float32)

        _, cache1 = lowrank(x1)
        _, cache2 = lowrank(x2, kv_cache=cache1)

        assert cache1.shape == (5, 16)
        assert cache2.shape == (6, 16)  # Accumulated

    def test_cache_size_smaller(self):
        """Low-rank should have smaller cache than standard."""
        lowrank = LowRankKVAttention(d_model=64, num_heads=8, d_latent=16)
        lowrank_cache = lowrank.get_cache_size(seq_len=1000, dtype_bytes=2)

        # Standard would be: 2 * seq * heads * head_dim * dtype = 2 * 1000 * 8 * 8 * 2
        standard_cache = 2 * 1000 * 8 * 8 * 2

        # Low-rank: seq * d_latent * dtype = 1000 * 16 * 2
        expected_lowrank = 1000 * 16 * 2

        assert lowrank_cache == expected_lowrank
        assert lowrank_cache < standard_cache


class TestCompareMemoryUsage:
    """Tests for compare_memory_usage function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = compare_memory_usage(
            seq_len=1000,
            num_layers=10,
            d_model=512,
            num_heads=8
        )
        assert isinstance(result, dict)

    def test_contains_mha(self):
        """Should contain MHA baseline."""
        result = compare_memory_usage(
            seq_len=1000,
            num_layers=10,
            d_model=512,
            num_heads=8
        )
        assert 'mha' in result

    def test_contains_gqa_when_specified(self):
        """Should contain GQA when gqa_kv_heads specified."""
        result = compare_memory_usage(
            seq_len=1000,
            num_layers=10,
            d_model=512,
            num_heads=8,
            gqa_kv_heads=2
        )
        assert 'gqa' in result
        assert 'gqa_ratio' in result

    def test_contains_lowrank_when_specified(self):
        """Should contain low-rank when lowrank_d_latent specified."""
        result = compare_memory_usage(
            seq_len=1000,
            num_layers=10,
            d_model=512,
            num_heads=8,
            lowrank_d_latent=128
        )
        assert 'lowrank' in result
        assert 'lowrank_ratio' in result

    def test_gqa_smaller_than_mha(self):
        """GQA should use less memory than MHA."""
        result = compare_memory_usage(
            seq_len=1000,
            num_layers=10,
            d_model=512,
            num_heads=8,
            gqa_kv_heads=2
        )
        assert result['gqa'] < result['mha']
        assert result['gqa_ratio'] > 1

    def test_lowrank_smaller_than_mha(self):
        """Low-rank should use less memory than MHA."""
        result = compare_memory_usage(
            seq_len=1000,
            num_layers=10,
            d_model=512,
            num_heads=8,
            lowrank_d_latent=128
        )
        assert result['lowrank'] < result['mha']
        assert result['lowrank_ratio'] > 1


class TestIntegration:
    """Integration tests."""

    def test_gqa_varying_groups(self):
        """GQA should work with various group configurations."""
        for num_kv_heads in [1, 2, 4, 8]:
            gqa = GroupedQueryAttention(
                d_model=64, num_q_heads=8, num_kv_heads=num_kv_heads
            )
            x = np.random.randn(10, 64).astype(np.float32)
            output = gqa(x)
            assert output.shape == x.shape

    def test_lowrank_varying_latent(self):
        """Low-rank should work with various latent dimensions."""
        for d_latent in [8, 16, 32, 64]:
            lowrank = LowRankKVAttention(
                d_model=64, num_heads=8, d_latent=d_latent
            )
            x = np.random.randn(10, 64).astype(np.float32)
            output, cache = lowrank(x)
            assert output.shape == x.shape
            assert cache.shape[-1] == d_latent

    def test_memory_savings_significant(self):
        """Compression techniques should provide significant savings."""
        result = compare_memory_usage(
            seq_len=4096,
            num_layers=32,
            d_model=4096,
            num_heads=32,
            dtype_bytes=2,
            gqa_kv_heads=4,
            lowrank_d_latent=512
        )

        # GQA should give ~8x reduction (32 heads -> 4 heads)
        assert result['gqa_ratio'] >= 7

        # Low-rank should give significant reduction
        assert result['lowrank_ratio'] >= 4
