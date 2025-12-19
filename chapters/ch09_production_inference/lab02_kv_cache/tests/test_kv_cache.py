"""Tests for Lab 02: KV-Cache Management."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_cache import (
    CacheConfig,
    KVCache,
    incremental_attention,
    compute_attention_flops,
    simulate_generation_memory,
    SlidingWindowCache,
)


@pytest.fixture
def basic_config():
    """Create a basic cache configuration for testing."""
    return CacheConfig(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        max_seq_len=128
    )


@pytest.fixture
def large_config():
    """Create a larger config similar to real models."""
    return CacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        max_seq_len=2048
    )


class TestKVCache:
    """Tests for the KVCache class."""

    def test_initialization(self, basic_config):
        """Cache should initialize with correct dimensions."""
        cache = KVCache(basic_config)
        assert cache.current_length == 0

    def test_append_single_token(self, basic_config):
        """Should append a single token to the cache."""
        cache = KVCache(basic_config)

        # Single token: shape (num_heads, head_dim)
        keys = np.random.randn(basic_config.num_heads, basic_config.head_dim)
        values = np.random.randn(basic_config.num_heads, basic_config.head_dim)

        cache.append(layer_idx=0, keys=keys, values=values)
        assert cache.current_length == 1

    def test_append_multiple_tokens(self, basic_config):
        """Should append multiple tokens (prefill)."""
        cache = KVCache(basic_config)

        # Multiple tokens: shape (seq_len, num_heads, head_dim)
        seq_len = 10
        keys = np.random.randn(seq_len, basic_config.num_heads, basic_config.head_dim)
        values = np.random.randn(seq_len, basic_config.num_heads, basic_config.head_dim)

        cache.append(layer_idx=0, keys=keys, values=values)
        assert cache.current_length == seq_len

    def test_append_incremental(self, basic_config):
        """Should support incremental appending."""
        cache = KVCache(basic_config)

        # First append 5 tokens
        keys1 = np.random.randn(5, basic_config.num_heads, basic_config.head_dim)
        values1 = np.random.randn(5, basic_config.num_heads, basic_config.head_dim)
        cache.append(layer_idx=0, keys=keys1, values=values1)
        assert cache.current_length == 5

        # Then append 3 more
        keys2 = np.random.randn(3, basic_config.num_heads, basic_config.head_dim)
        values2 = np.random.randn(3, basic_config.num_heads, basic_config.head_dim)
        cache.append(layer_idx=0, keys=keys2, values=values2)
        assert cache.current_length == 8

    def test_get_retrieves_correct_data(self, basic_config):
        """Get should return exactly what was appended."""
        cache = KVCache(basic_config)

        keys = np.random.randn(5, basic_config.num_heads, basic_config.head_dim)
        values = np.random.randn(5, basic_config.num_heads, basic_config.head_dim)

        cache.append(layer_idx=0, keys=keys, values=values)
        retrieved_k, retrieved_v = cache.get(layer_idx=0)

        np.testing.assert_allclose(retrieved_k, keys)
        np.testing.assert_allclose(retrieved_v, values)

    def test_get_with_range(self, basic_config):
        """Get should support start/end indexing."""
        cache = KVCache(basic_config)

        keys = np.random.randn(10, basic_config.num_heads, basic_config.head_dim)
        values = np.random.randn(10, basic_config.num_heads, basic_config.head_dim)

        cache.append(layer_idx=0, keys=keys, values=values)

        # Get positions 2-5
        k, v = cache.get(layer_idx=0, start=2, end=5)

        assert k.shape[0] == 3
        np.testing.assert_allclose(k, keys[2:5])
        np.testing.assert_allclose(v, values[2:5])

    def test_get_all_layers(self, basic_config):
        """Should retrieve cache for all layers at once."""
        cache = KVCache(basic_config)

        # Fill all layers with same length
        seq_len = 5
        for layer_idx in range(basic_config.num_layers):
            keys = np.random.randn(seq_len, basic_config.num_heads, basic_config.head_dim)
            values = np.random.randn(seq_len, basic_config.num_heads, basic_config.head_dim)
            cache.append(layer_idx=layer_idx, keys=keys, values=values)

        all_k, all_v = cache.get_all_layers()

        assert all_k.shape == (basic_config.num_layers, seq_len, basic_config.num_heads, basic_config.head_dim)
        assert all_v.shape == all_k.shape

    def test_clear(self, basic_config):
        """Clear should reset the cache."""
        cache = KVCache(basic_config)

        keys = np.random.randn(5, basic_config.num_heads, basic_config.head_dim)
        values = np.random.randn(5, basic_config.num_heads, basic_config.head_dim)
        cache.append(layer_idx=0, keys=keys, values=values)

        cache.clear()
        assert cache.current_length == 0

    def test_memory_usage(self, basic_config):
        """Memory usage should match expected calculation."""
        cache = KVCache(basic_config)

        # Memory = num_layers * max_seq_len * num_heads * head_dim * 2 (K+V) * dtype_size
        expected_memory = (
            basic_config.num_layers *
            basic_config.max_seq_len *
            basic_config.num_heads *
            basic_config.head_dim *
            2 *  # K and V
            np.dtype(basic_config.dtype).itemsize
        )

        assert cache.memory_usage_bytes() == expected_memory

    def test_large_config_memory(self, large_config):
        """Test memory calculation for realistic model size."""
        cache = KVCache(large_config)

        # 32 layers * 2048 seq * 32 heads * 128 dim * 2 * 4 bytes
        # = 32 * 2048 * 32 * 128 * 2 * 4 = 2,147,483,648 bytes = 2 GB
        memory_gb = cache.memory_usage_bytes() / (1024**3)

        # Should be about 2 GB
        assert 1.9 < memory_gb < 2.1


class TestIncrementalAttention:
    """Tests for incremental attention with KV-cache."""

    def test_incremental_attention_output_shape(self, basic_config):
        """Output should have correct shape."""
        cache = KVCache(basic_config)

        # First, prefill some context
        prefill_len = 10
        prefill_k = np.random.randn(prefill_len, basic_config.num_heads, basic_config.head_dim)
        prefill_v = np.random.randn(prefill_len, basic_config.num_heads, basic_config.head_dim)
        cache.append(layer_idx=0, keys=prefill_k, values=prefill_v)

        # Now generate one token
        query = np.random.randn(basic_config.num_heads, basic_config.head_dim)
        new_key = np.random.randn(basic_config.num_heads, basic_config.head_dim)
        new_value = np.random.randn(basic_config.num_heads, basic_config.head_dim)

        output, weights = incremental_attention(query, cache, layer_idx=0, new_key=new_key, new_value=new_value)

        assert output.shape == (basic_config.num_heads, basic_config.head_dim)
        assert weights.shape == (basic_config.num_heads, prefill_len + 1)  # +1 for new token

    def test_incremental_attention_weights_sum(self, basic_config):
        """Attention weights should sum to 1 for each head."""
        cache = KVCache(basic_config)

        prefill_len = 5
        prefill_k = np.random.randn(prefill_len, basic_config.num_heads, basic_config.head_dim)
        prefill_v = np.random.randn(prefill_len, basic_config.num_heads, basic_config.head_dim)
        cache.append(layer_idx=0, keys=prefill_k, values=prefill_v)

        query = np.random.randn(basic_config.num_heads, basic_config.head_dim)
        new_key = np.random.randn(basic_config.num_heads, basic_config.head_dim)
        new_value = np.random.randn(basic_config.num_heads, basic_config.head_dim)

        _, weights = incremental_attention(query, cache, layer_idx=0, new_key=new_key, new_value=new_value)

        # Each head's weights should sum to 1
        weight_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(weight_sums, np.ones(basic_config.num_heads), rtol=1e-5)


class TestComputeAttentionFlops:
    """Tests for FLOPs calculation."""

    def test_cache_reduces_flops(self):
        """Using cache should dramatically reduce FLOPs."""
        seq_len = 1000
        num_heads = 32
        head_dim = 128

        flops_no_cache = compute_attention_flops(seq_len, num_heads, head_dim, use_cache=False)
        flops_with_cache = compute_attention_flops(seq_len, num_heads, head_dim, use_cache=True)

        # With cache, FLOPs should be ~seq_len times fewer
        assert flops_with_cache < flops_no_cache
        ratio = flops_no_cache / flops_with_cache
        assert ratio > seq_len * 0.5  # At least 500x reduction for seq_len=1000

    def test_flops_scale_with_sequence(self):
        """FLOPs should scale quadratically without cache."""
        num_heads = 8
        head_dim = 64

        flops_100 = compute_attention_flops(100, num_heads, head_dim, use_cache=False)
        flops_200 = compute_attention_flops(200, num_heads, head_dim, use_cache=False)

        # Doubling seq_len should ~quadruple FLOPs (without cache)
        ratio = flops_200 / flops_100
        assert 3.5 < ratio < 4.5  # Allow some variance

    def test_flops_linear_with_cache(self):
        """With cache, FLOPs should scale linearly with sequence."""
        num_heads = 8
        head_dim = 64

        flops_100 = compute_attention_flops(100, num_heads, head_dim, use_cache=True)
        flops_200 = compute_attention_flops(200, num_heads, head_dim, use_cache=True)

        # Doubling seq_len should ~double FLOPs (with cache)
        ratio = flops_200 / flops_100
        assert 1.8 < ratio < 2.2


class TestSimulateGenerationMemory:
    """Tests for memory simulation."""

    def test_memory_grows_linearly(self, basic_config):
        """Memory should grow linearly with tokens."""
        result = simulate_generation_memory(100, basic_config)

        timeline = result['memory_timeline']
        assert len(timeline) == 100

        # Memory should increase monotonically
        for i in range(1, len(timeline)):
            assert timeline[i] > timeline[i-1]

        # Growth should be linear (constant increment)
        increments = [timeline[i] - timeline[i-1] for i in range(1, len(timeline))]
        assert all(abs(inc - increments[0]) < 1 for inc in increments)

    def test_memory_per_token(self, basic_config):
        """Memory per token should match expected value."""
        result = simulate_generation_memory(10, basic_config)

        # Memory per token = num_layers * num_heads * head_dim * 2 * dtype_size
        expected_per_token = (
            basic_config.num_layers *
            basic_config.num_heads *
            basic_config.head_dim *
            2 *  # K and V
            np.dtype(basic_config.dtype).itemsize
        )

        assert result['memory_per_token'] == expected_per_token

    def test_total_memory(self, basic_config):
        """Total memory should be tokens * memory_per_token."""
        num_tokens = 50
        result = simulate_generation_memory(num_tokens, basic_config)

        expected_total = num_tokens * result['memory_per_token']
        assert result['total_memory'] == expected_total


class TestSlidingWindowCache:
    """Tests for sliding window KV-cache."""

    def test_initialization(self, basic_config):
        """Sliding window cache should initialize correctly."""
        cache = SlidingWindowCache(basic_config, window_size=64)
        assert cache.current_length == 0
        assert cache.total_tokens_seen == 0

    def test_append_within_window(self, basic_config):
        """Appending within window size should work normally."""
        window_size = 64
        cache = SlidingWindowCache(basic_config, window_size=window_size)

        keys = np.random.randn(10, basic_config.num_heads, basic_config.head_dim)
        values = np.random.randn(10, basic_config.num_heads, basic_config.head_dim)

        cache.append(layer_idx=0, keys=keys, values=values)

        assert cache.current_length == 10
        assert cache.total_tokens_seen == 10

    def test_window_eviction(self, basic_config):
        """Should evict old tokens when window is exceeded."""
        window_size = 10
        cache = SlidingWindowCache(basic_config, window_size=window_size)

        # Add 15 tokens - should only keep last 10
        for i in range(15):
            key = np.full((basic_config.num_heads, basic_config.head_dim), i, dtype=np.float32)
            value = np.full((basic_config.num_heads, basic_config.head_dim), i, dtype=np.float32)
            cache.append(layer_idx=0, keys=key, values=value)

        assert cache.current_length == window_size  # Only 10 kept
        assert cache.total_tokens_seen == 15  # But 15 processed

        # Verify we have the LAST 10 tokens (5-14)
        k, v = cache.get(layer_idx=0)
        # The oldest token in cache should be 5 (0-4 were evicted)
        assert k[0, 0, 0] == 5

    def test_window_maintains_order(self, basic_config):
        """Window should maintain correct order of tokens."""
        window_size = 5
        cache = SlidingWindowCache(basic_config, window_size=window_size)

        # Add 8 tokens with values 0,1,2,3,4,5,6,7
        for i in range(8):
            key = np.full((basic_config.num_heads, basic_config.head_dim), i, dtype=np.float32)
            value = np.full((basic_config.num_heads, basic_config.head_dim), i, dtype=np.float32)
            cache.append(layer_idx=0, keys=key, values=value)

        k, v = cache.get(layer_idx=0)

        # Should have tokens 3,4,5,6,7 in order
        expected_values = [3, 4, 5, 6, 7]
        for i, expected in enumerate(expected_values):
            assert k[i, 0, 0] == expected
