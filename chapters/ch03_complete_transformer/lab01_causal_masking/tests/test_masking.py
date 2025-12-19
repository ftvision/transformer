"""Tests for Lab 01: Causal Masking."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from masking import (
    create_causal_mask,
    apply_mask_to_scores,
    create_padding_mask,
    create_padding_mask_2d,
    combine_masks,
    create_full_mask,
    masked_attention,
    softmax,
)


class TestCausalMask:
    """Tests for create_causal_mask function."""

    def test_causal_mask_shape(self):
        """Mask should have correct shape."""
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_causal_mask_dtype(self):
        """Mask should be boolean."""
        mask = create_causal_mask(4)
        assert mask.dtype == bool

    def test_causal_mask_diagonal(self):
        """Diagonal should be False (can attend to self)."""
        mask = create_causal_mask(4)
        for i in range(4):
            assert mask[i, i] == False, f"Position {i} should attend to itself"

    def test_causal_mask_lower_triangle(self):
        """Lower triangle should be False (can attend to past)."""
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i):
                assert mask[i, j] == False, f"Position {i} should attend to position {j}"

    def test_causal_mask_upper_triangle(self):
        """Upper triangle should be True (cannot attend to future)."""
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == True, f"Position {i} should NOT attend to position {j}"

    def test_causal_mask_seq_len_1(self):
        """Should work for sequence length 1."""
        mask = create_causal_mask(1)
        assert mask.shape == (1, 1)
        assert mask[0, 0] == False

    def test_causal_mask_specific_values(self):
        """Test specific expected values."""
        mask = create_causal_mask(3)
        expected = np.array([
            [False, True, True],
            [False, False, True],
            [False, False, False]
        ])
        np.testing.assert_array_equal(mask, expected)


class TestApplyMask:
    """Tests for apply_mask_to_scores function."""

    def test_apply_mask_basic(self):
        """Masked positions should become -inf."""
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])

        result = apply_mask_to_scores(scores, mask)

        assert result[0, 0] == 1.0
        assert result[0, 1] == -np.inf
        assert result[1, 0] == 3.0
        assert result[1, 1] == 4.0

    def test_apply_mask_preserves_unmasked(self):
        """Unmasked positions should be unchanged."""
        scores = np.random.randn(4, 4)
        mask = np.zeros((4, 4), dtype=bool)

        result = apply_mask_to_scores(scores, mask)

        np.testing.assert_array_equal(result, scores)

    def test_apply_mask_all_masked(self):
        """All masked positions should be -inf."""
        scores = np.random.randn(3, 3)
        mask = np.ones((3, 3), dtype=bool)

        result = apply_mask_to_scores(scores, mask)

        assert np.all(np.isinf(result))

    def test_apply_mask_custom_value(self):
        """Should support custom mask value."""
        scores = np.array([[1.0, 2.0]])
        mask = np.array([[False, True]])

        result = apply_mask_to_scores(scores, mask, mask_value=-1e9)

        assert result[0, 1] == -1e9

    def test_apply_mask_batched(self):
        """Should work with batched scores."""
        scores = np.random.randn(2, 4, 4)
        mask = create_causal_mask(4)

        result = apply_mask_to_scores(scores, mask)

        assert result.shape == (2, 4, 4)
        # Check upper triangle is -inf for all batches
        for b in range(2):
            upper = np.triu(result[b], k=1)
            assert np.all(np.isinf(upper))


class TestPaddingMask:
    """Tests for create_padding_mask function."""

    def test_padding_mask_shape(self):
        """Mask should have correct shape."""
        seq_lengths = np.array([3, 2, 4])
        mask = create_padding_mask(seq_lengths, max_len=5)
        assert mask.shape == (3, 5)

    def test_padding_mask_basic(self):
        """Test basic padding mask creation."""
        seq_lengths = np.array([3, 2])
        mask = create_padding_mask(seq_lengths, max_len=4)

        # Sequence 0: length 3, so position 3 is padding
        assert mask[0, 0] == False
        assert mask[0, 1] == False
        assert mask[0, 2] == False
        assert mask[0, 3] == True

        # Sequence 1: length 2, so positions 2, 3 are padding
        assert mask[1, 0] == False
        assert mask[1, 1] == False
        assert mask[1, 2] == True
        assert mask[1, 3] == True

    def test_padding_mask_no_padding(self):
        """No padding when sequence length equals max length."""
        seq_lengths = np.array([4, 4])
        mask = create_padding_mask(seq_lengths, max_len=4)

        assert not np.any(mask)  # All False

    def test_padding_mask_all_padding(self):
        """Empty sequences (length 0) should be all padding."""
        seq_lengths = np.array([0])
        mask = create_padding_mask(seq_lengths, max_len=3)

        assert np.all(mask)  # All True

    def test_padding_mask_dtype(self):
        """Mask should be boolean."""
        seq_lengths = np.array([2])
        mask = create_padding_mask(seq_lengths, max_len=4)
        assert mask.dtype == bool


class TestPaddingMask2D:
    """Tests for create_padding_mask_2d function."""

    def test_padding_mask_2d_shape(self):
        """Mask should have correct shape."""
        seq_lengths = np.array([3, 2])
        mask = create_padding_mask_2d(seq_lengths, max_len=4)
        assert mask.shape == (2, 4, 4)

    def test_padding_mask_2d_columns(self):
        """Padding columns should be all True."""
        seq_lengths = np.array([2])
        mask = create_padding_mask_2d(seq_lengths, max_len=4)

        # Columns 2, 3 should be all True (padding positions in keys)
        assert np.all(mask[0, :, 2])  # Column 2
        assert np.all(mask[0, :, 3])  # Column 3

        # Columns 0, 1 should be all False (valid positions)
        assert not np.any(mask[0, :, 0])
        assert not np.any(mask[0, :, 1])


class TestCombineMasks:
    """Tests for combine_masks function."""

    def test_combine_masks_or_logic(self):
        """Combined mask should use OR logic."""
        mask1 = np.array([[False, True], [False, False]])
        mask2 = np.array([[False, False], [True, False]])

        result = combine_masks(mask1, mask2)

        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(result, expected)

    def test_combine_masks_both_true(self):
        """Both True should give True."""
        mask1 = np.array([[True]])
        mask2 = np.array([[True]])

        result = combine_masks(mask1, mask2)

        assert result[0, 0] == True

    def test_combine_masks_both_false(self):
        """Both False should give False."""
        mask1 = np.array([[False]])
        mask2 = np.array([[False]])

        result = combine_masks(mask1, mask2)

        assert result[0, 0] == False

    def test_combine_masks_broadcasting(self):
        """Should handle broadcasting."""
        mask1 = np.array([[False, True], [False, False]])  # (2, 2)
        mask2 = np.array([False, True])  # (2,) - broadcasts to columns

        result = combine_masks(mask1, mask2)

        # mask2 broadcasts: column 1 is masked for all rows
        assert result.shape == (2, 2)
        assert result[0, 1] == True  # Both True
        assert result[1, 1] == True  # mask2 says True


class TestFullMask:
    """Tests for create_full_mask function."""

    def test_full_mask_shape(self):
        """Full mask should have correct shape."""
        seq_lengths = np.array([3, 2])
        mask = create_full_mask(seq_lengths, max_len=4, causal=True)
        assert mask.shape == (2, 4, 4)

    def test_full_mask_causal(self):
        """With causal=True, should include causal masking."""
        seq_lengths = np.array([4])  # No padding
        mask = create_full_mask(seq_lengths, max_len=4, causal=True)

        # Upper triangle should be masked
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[0, i, j] == True

    def test_full_mask_padding(self):
        """Should include padding masking."""
        seq_lengths = np.array([2])  # Positions 2, 3 are padding
        mask = create_full_mask(seq_lengths, max_len=4, causal=True)

        # Columns 2, 3 should be masked (can't attend to padding)
        assert np.all(mask[0, :, 2])
        assert np.all(mask[0, :, 3])

    def test_full_mask_no_causal(self):
        """With causal=False, only padding should be masked."""
        seq_lengths = np.array([2])
        mask = create_full_mask(seq_lengths, max_len=4, causal=False)

        # Position (0, 1) should NOT be masked (not padding, not causal)
        assert mask[0, 0, 1] == False

    def test_full_mask_combined(self):
        """Should combine both causal and padding constraints."""
        seq_lengths = np.array([3])  # Position 3 is padding
        mask = create_full_mask(seq_lengths, max_len=4, causal=True)

        # Position 0 can only attend to position 0
        assert mask[0, 0, 0] == False  # Can attend to self
        assert mask[0, 0, 1] == True   # Causal: can't see future
        assert mask[0, 0, 2] == True   # Causal: can't see future
        assert mask[0, 0, 3] == True   # Both: can't see future + padding

        # Position 2 can attend to 0, 1, 2 but not 3 (padding)
        assert mask[0, 2, 0] == False
        assert mask[0, 2, 1] == False
        assert mask[0, 2, 2] == False
        assert mask[0, 2, 3] == True  # Padding


class TestMaskedAttention:
    """Tests for masked_attention function."""

    def test_masked_attention_shapes(self):
        """Output should have correct shapes."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 10)
        mask = create_causal_mask(4)

        output, weights = masked_attention(Q, K, V, mask)

        assert output.shape == (4, 10)
        assert weights.shape == (4, 4)

    def test_masked_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 for each query."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 8)
        mask = create_causal_mask(4)

        _, weights = masked_attention(Q, K, V, mask)

        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), rtol=1e-5)

    def test_masked_attention_masked_weights_zero(self):
        """Masked positions should have zero attention weight."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 8)
        mask = create_causal_mask(4)

        _, weights = masked_attention(Q, K, V, mask)

        # Upper triangle should be ~0
        upper = np.triu(weights, k=1)
        np.testing.assert_allclose(upper, 0, atol=1e-6)

    def test_masked_attention_no_mask(self):
        """Should work without mask (full attention)."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 8)

        output, weights = masked_attention(Q, K, V, mask=None)

        assert output.shape == (4, 8)
        assert weights.shape == (4, 4)
        # All weights should be positive (no masking)
        assert np.all(weights > 0)

    def test_masked_attention_batched(self):
        """Should work with batched inputs."""
        batch_size = 2
        Q = np.random.randn(batch_size, 4, 8)
        K = np.random.randn(batch_size, 4, 8)
        V = np.random.randn(batch_size, 4, 10)
        mask = create_causal_mask(4)

        output, weights = masked_attention(Q, K, V, mask)

        assert output.shape == (batch_size, 4, 10)
        assert weights.shape == (batch_size, 4, 4)

    def test_masked_attention_first_position(self):
        """First position should only attend to itself."""
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.eye(4, 8)  # Identity-like for easy verification
        mask = create_causal_mask(4)

        _, weights = masked_attention(Q, K, V, mask)

        # First position: all weight on position 0
        np.testing.assert_allclose(weights[0, 0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(weights[0, 1:], 0.0, atol=1e-6)


class TestSoftmaxWithMask:
    """Tests for softmax behavior with masked values."""

    def test_softmax_handles_neginf(self):
        """Softmax should give 0 for -inf positions."""
        x = np.array([1.0, 2.0, -np.inf])
        result = softmax(x)

        assert result[2] == 0 or np.isclose(result[2], 0, atol=1e-10)
        np.testing.assert_allclose(result[0] + result[1], 1.0, rtol=1e-5)

    def test_softmax_all_neginf_row(self):
        """Row with all -inf should not produce NaN."""
        # This is an edge case - in practice shouldn't happen
        x = np.array([[-np.inf, -np.inf], [1.0, 2.0]])
        result = softmax(x, axis=-1)

        # Should not have NaN
        assert not np.any(np.isnan(result))


class TestMaskingCorrectness:
    """Integration tests for masking correctness."""

    def test_causal_attention_output(self):
        """Verify causal attention produces different outputs per position."""
        np.random.seed(42)

        # Create input where each position is different
        seq_len = 4
        d_k = 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.arange(seq_len * d_k).reshape(seq_len, d_k).astype(float)

        mask = create_causal_mask(seq_len)
        output, _ = masked_attention(Q, K, V, mask)

        # Position 0 output should only use V[0]
        # (since it can only attend to position 0)
        # The weighted average should be exactly V[0]
        _, weights = masked_attention(Q[:1], K[:1], V[:1])
        expected_pos0 = weights @ V[:1]
        # Note: With only position 0 visible, output[0] depends only on V[0]

    def test_padding_correctly_ignored(self):
        """Verify padding positions are correctly ignored in attention."""
        seq_len = 4
        d_k = 8

        Q = np.random.randn(1, seq_len, d_k)
        K = np.random.randn(1, seq_len, d_k)

        # V where padding positions have very different values
        V = np.ones((1, seq_len, d_k))
        V[0, 2:, :] = 1000  # Padding has large values

        # Create mask where positions 2, 3 are padding
        seq_lengths = np.array([2])
        mask = create_full_mask(seq_lengths, seq_len, causal=False)

        output, weights = masked_attention(Q, K, V, mask)

        # Output should NOT be affected by large padding values
        # Should be close to 1.0 (the non-padding values)
        assert np.all(output < 100), "Output affected by padding values!"
