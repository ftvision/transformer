"""Tests for Lab 02: KV-Cache Implementation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_cache import (
    KVCache,
    attention_with_kv_cache,
    count_attention_operations,
    generate_with_cache,
    generate_without_cache,
    scaled_dot_product_attention,
)


class TestKVCacheInit:
    """Tests for KVCache initialization."""

    def test_cache_shape(self):
        """Cache should have correct shape."""
        cache = KVCache(max_seq_len=2048, num_heads=32, head_dim=128)
        assert cache.k_cache.shape == (1, 32, 2048, 128)
        assert cache.v_cache.shape == (1, 32, 2048, 128)

    def test_initial_length_zero(self):
        """Cache should start with length 0."""
        cache = KVCache(max_seq_len=2048, num_heads=32, head_dim=128)
        assert cache.length == 0

    def test_dtype(self):
        """Cache should use specified dtype."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64, dtype=np.float16)
        assert cache.k_cache.dtype == np.float16
        assert cache.v_cache.dtype == np.float16


class TestKVCacheUpdate:
    """Tests for KVCache.update."""

    def test_single_update(self):
        """Update with single token."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        k = np.random.randn(1, 4, 1, 64).astype(np.float32)
        v = np.random.randn(1, 4, 1, 64).astype(np.float32)

        cache.update(k, v, position=0)

        assert cache.length == 1

    def test_batch_update(self):
        """Update with multiple tokens at once (prefill)."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        k = np.random.randn(1, 4, 10, 64).astype(np.float32)
        v = np.random.randn(1, 4, 10, 64).astype(np.float32)

        cache.update(k, v, position=0)

        assert cache.length == 10

    def test_sequential_updates(self):
        """Multiple sequential updates (decode steps)."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)

        for i in range(5):
            k = np.random.randn(1, 4, 1, 64).astype(np.float32)
            v = np.random.randn(1, 4, 1, 64).astype(np.float32)
            cache.update(k, v, position=i)

        assert cache.length == 5

    def test_update_values_stored(self):
        """Cached values should match what was stored."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        k = np.ones((1, 4, 3, 64), dtype=np.float32) * 5.0
        v = np.ones((1, 4, 3, 64), dtype=np.float32) * 7.0

        cache.update(k, v, position=0)
        k_cached, v_cached = cache.get()

        np.testing.assert_allclose(k_cached, k)
        np.testing.assert_allclose(v_cached, v)


class TestKVCacheGet:
    """Tests for KVCache.get."""

    def test_get_returns_correct_length(self):
        """Get should return only cached values."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        k = np.random.randn(1, 4, 5, 64).astype(np.float32)
        v = np.random.randn(1, 4, 5, 64).astype(np.float32)

        cache.update(k, v, position=0)
        k_cached, v_cached = cache.get()

        assert k_cached.shape == (1, 4, 5, 64)
        assert v_cached.shape == (1, 4, 5, 64)

    def test_get_with_end_position(self):
        """Get with explicit end position."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        k = np.random.randn(1, 4, 10, 64).astype(np.float32)
        v = np.random.randn(1, 4, 10, 64).astype(np.float32)

        cache.update(k, v, position=0)
        k_cached, v_cached = cache.get(end_position=5)

        assert k_cached.shape == (1, 4, 5, 64)

    def test_get_after_multiple_updates(self):
        """Get after prefill + decode updates."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)

        # Prefill
        k1 = np.random.randn(1, 4, 5, 64).astype(np.float32)
        v1 = np.random.randn(1, 4, 5, 64).astype(np.float32)
        cache.update(k1, v1, position=0)

        # Decode
        k2 = np.random.randn(1, 4, 1, 64).astype(np.float32)
        v2 = np.random.randn(1, 4, 1, 64).astype(np.float32)
        cache.update(k2, v2, position=5)

        k_cached, v_cached = cache.get()
        assert k_cached.shape == (1, 4, 6, 64)


class TestKVCacheReset:
    """Tests for KVCache.reset."""

    def test_reset_clears_length(self):
        """Reset should set length to 0."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        k = np.random.randn(1, 4, 10, 64).astype(np.float32)
        v = np.random.randn(1, 4, 10, 64).astype(np.float32)

        cache.update(k, v, position=0)
        assert cache.length == 10

        cache.reset()
        assert cache.length == 0

    def test_reset_allows_reuse(self):
        """After reset, cache should be reusable."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)

        # First use
        k1 = np.random.randn(1, 4, 5, 64).astype(np.float32)
        v1 = np.random.randn(1, 4, 5, 64).astype(np.float32)
        cache.update(k1, v1, position=0)

        cache.reset()

        # Second use
        k2 = np.random.randn(1, 4, 3, 64).astype(np.float32)
        v2 = np.random.randn(1, 4, 3, 64).astype(np.float32)
        cache.update(k2, v2, position=0)

        assert cache.length == 3
        k_cached, v_cached = cache.get()
        np.testing.assert_allclose(k_cached, k2)


class TestAttentionWithKVCache:
    """Tests for attention_with_kv_cache."""

    def test_prefill_output_shape(self):
        """Prefill should produce correct output shape."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)
        q = np.random.randn(1, 4, 10, 64).astype(np.float32)
        k = np.random.randn(1, 4, 10, 64).astype(np.float32)
        v = np.random.randn(1, 4, 10, 64).astype(np.float32)

        output, cache = attention_with_kv_cache(q, k, v, cache, position=0)

        assert output.shape == (1, 4, 10, 64)
        assert cache.length == 10

    def test_decode_output_shape(self):
        """Decode should produce correct output shape."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)

        # Prefill
        q1 = np.random.randn(1, 4, 10, 64).astype(np.float32)
        k1 = np.random.randn(1, 4, 10, 64).astype(np.float32)
        v1 = np.random.randn(1, 4, 10, 64).astype(np.float32)
        _, cache = attention_with_kv_cache(q1, k1, v1, cache, position=0)

        # Decode
        q2 = np.random.randn(1, 4, 1, 64).astype(np.float32)
        k2 = np.random.randn(1, 4, 1, 64).astype(np.float32)
        v2 = np.random.randn(1, 4, 1, 64).astype(np.float32)
        output, cache = attention_with_kv_cache(q2, k2, v2, cache, position=10)

        assert output.shape == (1, 4, 1, 64)
        assert cache.length == 11

    def test_cache_grows_correctly(self):
        """Cache should grow with each decode step."""
        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)

        # Prefill with 5 tokens
        q = np.random.randn(1, 4, 5, 64).astype(np.float32)
        k = np.random.randn(1, 4, 5, 64).astype(np.float32)
        v = np.random.randn(1, 4, 5, 64).astype(np.float32)
        _, cache = attention_with_kv_cache(q, k, v, cache, position=0)
        assert cache.length == 5

        # 3 decode steps
        for i in range(3):
            q = np.random.randn(1, 4, 1, 64).astype(np.float32)
            k = np.random.randn(1, 4, 1, 64).astype(np.float32)
            v = np.random.randn(1, 4, 1, 64).astype(np.float32)
            _, cache = attention_with_kv_cache(q, k, v, cache, position=5 + i)
            assert cache.length == 6 + i

    def test_no_cache_mode(self):
        """Should work without cache (cache=None)."""
        q = np.random.randn(1, 4, 5, 64).astype(np.float32)
        k = np.random.randn(1, 4, 5, 64).astype(np.float32)
        v = np.random.randn(1, 4, 5, 64).astype(np.float32)

        output, returned_cache = attention_with_kv_cache(q, k, v, cache=None, position=0)

        assert output.shape == (1, 4, 5, 64)
        assert returned_cache is None

    def test_causal_masking_in_prefill(self):
        """Prefill should use causal masking."""
        cache = KVCache(max_seq_len=100, num_heads=1, head_dim=4)

        # Create simple inputs where we can verify causality
        # If causal masking works, position 0 should only see position 0
        q = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]]).astype(np.float32)
        k = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]]).astype(np.float32)
        v = np.array([[[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]]]]).astype(np.float32)

        output, _ = attention_with_kv_cache(q, k, v, cache, position=0)

        # Position 0 should only attend to itself, getting v[0]
        # With softmax on a single score, weight = 1.0
        # So output[0] should be close to v[0]
        np.testing.assert_allclose(output[0, 0, 0], v[0, 0, 0], atol=1e-5)


class TestCountAttentionOperations:
    """Tests for count_attention_operations."""

    def test_with_cache_linear(self):
        """With cache, operations should equal sequence length."""
        assert count_attention_operations(100, with_cache=True) == 100
        assert count_attention_operations(50, with_cache=True) == 50

    def test_without_cache_quadratic(self):
        """Without cache, operations should be N(N+1)/2."""
        # 1 + 2 + 3 + ... + 100 = 5050
        assert count_attention_operations(100, with_cache=False) == 5050
        # 1 + 2 + 3 + ... + 10 = 55
        assert count_attention_operations(10, with_cache=False) == 55

    def test_cache_is_more_efficient(self):
        """Cache should always use fewer operations."""
        for n in [10, 50, 100, 500]:
            with_cache = count_attention_operations(n, with_cache=True)
            without_cache = count_attention_operations(n, with_cache=False)
            assert with_cache < without_cache, f"Cache should be more efficient for n={n}"

    def test_efficiency_ratio(self):
        """Cache should be approximately N/2 times more efficient."""
        n = 1000
        with_cache = count_attention_operations(n, with_cache=True)
        without_cache = count_attention_operations(n, with_cache=False)
        ratio = without_cache / with_cache
        # Should be approximately (n+1)/2
        assert abs(ratio - (n + 1) / 2) < 1


class TestGenerateWithCache:
    """Tests for generate_with_cache."""

    def test_returns_correct_steps(self):
        """Should return correct number of steps."""
        total, per_step = generate_with_cache(10, 5)
        # 1 prefill + 5 decode = 6 steps
        assert len(per_step) == 6

    def test_prefill_ops(self):
        """Prefill should have causal attention ops."""
        total, per_step = generate_with_cache(10, 5)
        # Prefill: causal attention on 10 tokens = 1+2+...+10 = 55
        assert per_step[0] == 55

    def test_decode_ops_linear(self):
        """Each decode step should have linear ops (attend to all previous)."""
        total, per_step = generate_with_cache(10, 5)
        # After prefill (10 tokens), decode attends to:
        # Step 1: 11 positions
        # Step 2: 12 positions
        # etc.
        assert per_step[1] == 11
        assert per_step[2] == 12
        assert per_step[3] == 13
        assert per_step[4] == 14
        assert per_step[5] == 15


class TestGenerateWithoutCache:
    """Tests for generate_without_cache."""

    def test_returns_correct_steps(self):
        """Should return correct number of steps."""
        total, per_step = generate_without_cache(10, 5)
        assert len(per_step) == 6

    def test_prefill_same_as_with_cache(self):
        """Prefill should be the same with or without cache."""
        _, per_step_with = generate_with_cache(10, 5)
        _, per_step_without = generate_without_cache(10, 5)
        assert per_step_with[0] == per_step_without[0]

    def test_decode_ops_quadratic(self):
        """Without cache, decode recomputes full attention."""
        total, per_step = generate_without_cache(10, 5)
        # After prefill (10 tokens), without cache we recompute everything:
        # Step 1: 1+2+...+11 = 66
        # Step 2: 1+2+...+12 = 78
        # etc.
        assert per_step[1] == 66  # 11 * 12 / 2
        assert per_step[2] == 78  # 12 * 13 / 2

    def test_without_cache_uses_more_ops(self):
        """Without cache should use more total operations."""
        total_with, _ = generate_with_cache(10, 100)
        total_without, _ = generate_without_cache(10, 100)
        assert total_without > total_with


class TestMilestone:
    """Integration test for the complete KV-cache implementation."""

    def test_kv_cache_efficiency(self):
        """KV-cache should reduce complexity from O(N²) to O(N)."""
        prompt_len = 100
        gen_len = 400  # Generate 400 tokens
        total_len = prompt_len + gen_len

        total_with, per_step_with = generate_with_cache(prompt_len, gen_len)
        total_without, per_step_without = generate_without_cache(prompt_len, gen_len)

        # Calculate efficiency gain
        efficiency_ratio = total_without / total_with

        print(f"\n✅ Milestone Test - KV-Cache Efficiency")
        print(f"   Prompt length: {prompt_len}")
        print(f"   Generated tokens: {gen_len}")
        print(f"   With cache ops: {total_with:,}")
        print(f"   Without cache ops: {total_without:,}")
        print(f"   Efficiency gain: {efficiency_ratio:.1f}x")

        # Should be roughly (N+1)/2 times more efficient
        expected_ratio = (total_len + 1) / 2
        assert efficiency_ratio > expected_ratio * 0.8, \
            f"Expected ~{expected_ratio:.0f}x efficiency, got {efficiency_ratio:.1f}x"

    def test_kv_cache_correctness(self):
        """KV-cache should produce correct attention output."""
        np.random.seed(42)

        cache = KVCache(max_seq_len=100, num_heads=4, head_dim=64)

        # Create test data
        q = np.random.randn(1, 4, 5, 64).astype(np.float32)
        k = np.random.randn(1, 4, 5, 64).astype(np.float32)
        v = np.random.randn(1, 4, 5, 64).astype(np.float32)

        # Compute with cache
        output_cached, _ = attention_with_kv_cache(q, k, v, cache, position=0)

        # Compute without cache (direct attention with causal mask)
        seq_len = 5
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # Add batch and head dims
        output_direct = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # Should match
        np.testing.assert_allclose(output_cached, output_direct, atol=1e-5)

        print("\n✅ Milestone Test - KV-Cache Correctness")
        print("   Cached attention matches direct attention!")
