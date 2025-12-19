"""Tests for Lab 02: Sliding Window Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sliding_window import (
    create_sliding_window_mask,
    sliding_window_attention,
    SlidingWindowAttention,
    compute_effective_context,
    analyze_attention_pattern,
)


class TestSlidingWindowMask:
    """Tests for create_sliding_window_mask function."""

    def test_mask_shape(self):
        """Mask should have correct shape."""
        mask = create_sliding_window_mask(10, window_size=3)
        assert mask.shape == (10, 10)

    def test_mask_dtype(self):
        """Mask should be boolean."""
        mask = create_sliding_window_mask(10, window_size=3)
        assert mask.dtype == bool

    def test_causal_constraint(self):
        """Upper triangle should be blocked."""
        mask = create_sliding_window_mask(10, window_size=3)
        upper = np.triu(mask, k=1)
        expected = np.triu(np.ones((10, 10), dtype=bool), k=1)
        np.testing.assert_array_equal(upper, expected)

    def test_self_attention_allowed(self):
        """Diagonal should be unblocked."""
        mask = create_sliding_window_mask(10, window_size=3)
        diagonal = np.diag(mask)
        assert not diagonal.any()

    def test_window_size_1(self):
        """Window size 1 should only allow self-attention."""
        mask = create_sliding_window_mask(5, window_size=1)
        # Only diagonal should be False
        expected = ~np.eye(5, dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_window_clips_at_start(self):
        """Early positions should have clipped windows."""
        mask = create_sliding_window_mask(10, window_size=3)
        # Position 1 can attend to [0, 1]
        assert not mask[1, 0]
        assert not mask[1, 1]

    def test_window_slides(self):
        """Window should slide forward."""
        mask = create_sliding_window_mask(10, window_size=3)
        # Position 5 can attend to [3, 4, 5]
        assert mask[5, 2]  # Outside window
        assert not mask[5, 3]  # In window
        assert not mask[5, 4]  # In window
        assert not mask[5, 5]  # Self

    def test_global_positions_row(self):
        """Global position should attend to all (causally)."""
        mask = create_sliding_window_mask(8, window_size=3, global_positions=[0])
        # Position 0 (global) can attend to all previous = just itself
        assert not mask[0, 0]

        mask = create_sliding_window_mask(8, window_size=3, global_positions=[7])
        # Position 7 (global) can attend to all [0..7]
        assert not mask[7].any()

    def test_global_positions_column(self):
        """All positions should attend to global positions."""
        mask = create_sliding_window_mask(8, window_size=3, global_positions=[0])
        # All positions can attend to position 0
        for i in range(8):
            assert not mask[i, 0], f"Position {i} should attend to global position 0"

    def test_multiple_global_positions(self):
        """Multiple global positions should work."""
        mask = create_sliding_window_mask(10, window_size=3, global_positions=[0, 1])
        # All positions can attend to positions 0 and 1
        for i in range(10):
            assert not mask[i, 0]
            if i >= 1:  # Position 0 can't attend to future
                assert not mask[i, 1]

    def test_global_still_causal(self):
        """Global tokens still can't see the future."""
        mask = create_sliding_window_mask(8, window_size=3, global_positions=[2])
        # Position 2 is global but can't see positions 3+
        assert mask[2, 3]
        assert mask[2, 4]


class TestSlidingWindowAttention:
    """Tests for sliding_window_attention function."""

    def test_output_shape(self):
        """Output should have correct shape."""
        seq_len, d_k, d_v = 10, 8, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)

        output, weights = sliding_window_attention(Q, K, V, window_size=3)

        assert output.shape == (seq_len, d_v)
        assert weights.shape == (seq_len, seq_len)

    def test_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        Q = np.random.randn(10, 8)
        K = np.random.randn(10, 8)
        V = np.random.randn(10, 8)

        _, weights = sliding_window_attention(Q, K, V, window_size=3)

        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_masked_positions_zero_weight(self):
        """Positions outside window should have ~0 weight."""
        Q = np.random.randn(10, 8)
        K = np.random.randn(10, 8)
        V = np.random.randn(10, 8)

        _, weights = sliding_window_attention(Q, K, V, window_size=3)

        # Position 5's weights for positions 0, 1 should be ~0
        np.testing.assert_allclose(weights[5, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(weights[5, 1], 0.0, atol=1e-6)

    def test_batched(self):
        """Should work with batched input."""
        Q = np.random.randn(2, 10, 8)
        K = np.random.randn(2, 10, 8)
        V = np.random.randn(2, 10, 8)

        output, weights = sliding_window_attention(Q, K, V, window_size=3)

        assert output.shape == (2, 10, 8)
        assert weights.shape == (2, 10, 10)

    def test_global_mask(self):
        """Global positions should have different attention pattern."""
        Q = np.random.randn(10, 8)
        K = np.random.randn(10, 8)
        V = np.random.randn(10, 8)

        global_mask = np.zeros(10, dtype=bool)
        global_mask[0] = True  # Position 0 is global

        _, weights = sliding_window_attention(Q, K, V, window_size=3, global_mask=global_mask)

        # Position 5 should have non-zero weight for global position 0
        assert weights[5, 0] > 0


class TestSlidingWindowAttentionClass:
    """Tests for SlidingWindowAttention class."""

    def test_init_valid(self):
        """Should initialize with valid parameters."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        assert swa.d_model == 64
        assert swa.num_heads == 8
        assert swa.d_k == 8
        assert swa.window_size == 16

    def test_init_invalid_d_model(self):
        """Should raise error for invalid d_model."""
        with pytest.raises(ValueError):
            SlidingWindowAttention(d_model=65, num_heads=8, window_size=16)

    def test_init_with_global_tokens(self):
        """Should initialize with global tokens."""
        swa = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=2
        )
        assert swa.num_global_tokens == 2

    def test_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        assert swa.W_Q.shape == (64, 64)
        assert swa.W_K.shape == (64, 64)
        assert swa.W_V.shape == (64, 64)
        assert swa.W_O.shape == (64, 64)

    def test_forward_shape_batched(self):
        """Forward output should match input shape."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 100, 64).astype(np.float32)

        output = swa(x)

        assert output.shape == x.shape

    def test_forward_shape_unbatched(self):
        """Should handle unbatched input."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(100, 64).astype(np.float32)

        output = swa(x)

        assert output.shape == x.shape

    def test_forward_deterministic(self):
        """Same input should give same output."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 50, 64).astype(np.float32)

        out1 = swa(x)
        out2 = swa(x)

        np.testing.assert_array_equal(out1, out2)

    def test_forward_with_global_tokens(self):
        """Should work with global tokens."""
        swa = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=2
        )
        x = np.random.randn(2, 50, 64).astype(np.float32)

        output = swa(x)

        assert output.shape == x.shape

    def test_attention_weights_shape(self):
        """Attention weights should have correct shape."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 50, 64).astype(np.float32)

        weights = swa.get_attention_weights(x)

        assert weights.shape == (2, 8, 50, 50)

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 20, 64).astype(np.float32)

        weights = swa.get_attention_weights(x)

        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_window_enforced(self):
        """Positions outside window should have ~0 attention."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=4)
        x = np.random.randn(20, 64).astype(np.float32)

        weights = swa.get_attention_weights(x)

        # For position 10, positions 0-5 should be ~0 (outside window)
        for h in range(8):
            for j in range(6):
                np.testing.assert_allclose(
                    weights[h, 10, j], 0.0, atol=1e-6,
                    err_msg=f"Head {h}, pos 10 should not attend to pos {j}"
                )


class TestSplitCombineHeads:
    """Tests for head splitting and combining."""

    def test_split_heads_batched(self):
        """Split heads should work for batched input."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 10, 64)

        split = swa._split_heads(x)

        assert split.shape == (2, 8, 10, 8)

    def test_split_heads_unbatched(self):
        """Split heads should work for unbatched input."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(10, 64)

        split = swa._split_heads(x)

        assert split.shape == (8, 10, 8)

    def test_combine_heads_batched(self):
        """Combine heads should work for batched input."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 8, 10, 8)

        combined = swa._combine_heads(x)

        assert combined.shape == (2, 10, 64)

    def test_split_combine_roundtrip(self):
        """Split then combine should preserve shape."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
        x = np.random.randn(2, 10, 64)

        split = swa._split_heads(x)
        combined = swa._combine_heads(split)

        assert combined.shape == x.shape


class TestGlobalMask:
    """Tests for global mask creation."""

    def test_no_global_tokens(self):
        """Should return None when num_global_tokens=0."""
        swa = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=0
        )
        mask = swa._create_global_mask(50)
        assert mask is None

    def test_global_mask_shape(self):
        """Global mask should have correct shape."""
        swa = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=2
        )
        mask = swa._create_global_mask(50)
        assert mask.shape == (50,)

    def test_global_mask_values(self):
        """Global mask should mark correct positions."""
        swa = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=3
        )
        mask = swa._create_global_mask(50)

        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == True
        assert mask[3] == False
        assert mask[49] == False


class TestEffectiveContext:
    """Tests for compute_effective_context function."""

    def test_single_layer(self):
        """Single layer should have context = window_size."""
        context = compute_effective_context(1000, window_size=128, num_layers=1)
        assert context == 128

    def test_multiple_layers(self):
        """Multiple layers should expand context."""
        context = compute_effective_context(1000, window_size=128, num_layers=4)
        assert context == 512

    def test_capped_at_seq_len(self):
        """Context should be capped at sequence length."""
        context = compute_effective_context(100, window_size=128, num_layers=10)
        assert context == 100

    def test_zero_layers(self):
        """Zero layers should have zero context."""
        context = compute_effective_context(100, window_size=128, num_layers=0)
        assert context == 0


class TestAnalyzePattern:
    """Tests for analyze_attention_pattern function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        weights = np.random.rand(10, 10)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        result = analyze_attention_pattern(weights)

        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Should contain all required keys."""
        weights = np.random.rand(10, 10)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        result = analyze_attention_pattern(weights)

        assert 'local_attention_ratio' in result
        assert 'global_attention_ratio' in result
        assert 'avg_entropy' in result
        assert 'sparsity' in result

    def test_ratios_in_range(self):
        """Ratios should be between 0 and 1."""
        weights = np.random.rand(10, 10)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        result = analyze_attention_pattern(weights)

        assert 0 <= result['local_attention_ratio'] <= 1
        assert 0 <= result['sparsity'] <= 1

    def test_entropy_non_negative(self):
        """Entropy should be non-negative."""
        weights = np.random.rand(10, 10)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        result = analyze_attention_pattern(weights)

        assert result['avg_entropy'] >= 0


class TestIntegration:
    """Integration tests for sliding window attention."""

    def test_long_sequence(self):
        """Should handle long sequences efficiently."""
        swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=32)
        x = np.random.randn(1, 500, 64).astype(np.float32)

        output = swa(x)

        assert output.shape == x.shape

    def test_varying_window_sizes(self):
        """Should work with various window sizes."""
        for window_size in [4, 16, 64, 128]:
            swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=window_size)
            x = np.random.randn(2, 200, 64).astype(np.float32)

            output = swa(x)

            assert output.shape == x.shape

    def test_global_tokens_affect_output(self):
        """Adding global tokens should change the output."""
        np.random.seed(42)

        swa_no_global = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=0
        )
        swa_with_global = SlidingWindowAttention(
            d_model=64, num_heads=8, window_size=16, num_global_tokens=2
        )

        # Copy weights to ensure fair comparison
        swa_with_global.W_Q = swa_no_global.W_Q.copy()
        swa_with_global.W_K = swa_no_global.W_K.copy()
        swa_with_global.W_V = swa_no_global.W_V.copy()
        swa_with_global.W_O = swa_no_global.W_O.copy()

        x = np.random.randn(50, 64).astype(np.float32)

        out_no_global = swa_no_global(x)
        out_with_global = swa_with_global(x)

        # Outputs should be different
        assert not np.allclose(out_no_global, out_with_global)
