"""Tests for Lab 01: Sparse Attention Patterns."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_patterns import (
    create_causal_mask,
    create_local_mask,
    create_strided_mask,
    create_block_mask,
    create_combined_mask,
    apply_sparse_mask,
    compute_sparsity,
    visualize_mask_pattern,
)


class TestCausalMask:
    """Tests for causal attention mask."""

    def test_causal_mask_shape(self):
        """Causal mask should have correct shape."""
        mask = create_causal_mask(10)
        assert mask.shape == (10, 10)

    def test_causal_mask_dtype(self):
        """Causal mask should be boolean."""
        mask = create_causal_mask(5)
        assert mask.dtype == bool

    def test_causal_mask_diagonal_allowed(self):
        """Diagonal should be False (can attend to self)."""
        mask = create_causal_mask(5)
        diagonal = np.diag(mask)
        assert not diagonal.any(), "Should be able to attend to self"

    def test_causal_mask_lower_triangle_allowed(self):
        """Lower triangle should be False (can attend to past)."""
        mask = create_causal_mask(5)
        lower = np.tril(mask, k=-1)
        assert not lower.any(), "Should be able to attend to past positions"

    def test_causal_mask_upper_triangle_blocked(self):
        """Upper triangle should be True (cannot attend to future)."""
        mask = create_causal_mask(5)
        upper = np.triu(mask, k=1)
        # All upper triangle should be True (blocked)
        expected_upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
        np.testing.assert_array_equal(upper, expected_upper)

    def test_causal_mask_first_row(self):
        """First position can only attend to itself."""
        mask = create_causal_mask(5)
        expected = np.array([False, True, True, True, True])
        np.testing.assert_array_equal(mask[0], expected)

    def test_causal_mask_last_row(self):
        """Last position can attend to all previous positions."""
        mask = create_causal_mask(5)
        expected = np.array([False, False, False, False, False])
        np.testing.assert_array_equal(mask[4], expected)


class TestLocalMask:
    """Tests for local (sliding window) attention mask."""

    def test_local_mask_shape(self):
        """Local mask should have correct shape."""
        mask = create_local_mask(10, window_size=3)
        assert mask.shape == (10, 10)

    def test_local_mask_includes_causal(self):
        """Local mask should include causal constraint."""
        mask = create_local_mask(10, window_size=5)
        # Upper triangle should be blocked
        upper = np.triu(mask, k=1)
        expected_upper = np.triu(np.ones((10, 10), dtype=bool), k=1)
        np.testing.assert_array_equal(upper, expected_upper)

    def test_local_mask_window_size_1(self):
        """Window size 1 means only self-attention."""
        mask = create_local_mask(5, window_size=1)
        # Should only allow diagonal
        for i in range(5):
            for j in range(5):
                if i == j:
                    assert not mask[i, j], f"Position {i} should attend to itself"
                else:
                    assert mask[i, j], f"Position {i} should NOT attend to {j}"

    def test_local_mask_window_clips_at_start(self):
        """Window should clip at sequence start."""
        mask = create_local_mask(10, window_size=3)
        # Position 0 can only attend to [0]
        assert not mask[0, 0]  # Can attend to self
        assert mask[0, 1]  # Cannot attend to future

        # Position 1 can attend to [0, 1]
        assert not mask[1, 0]
        assert not mask[1, 1]

    def test_local_mask_window_slides(self):
        """Window should slide forward for later positions."""
        mask = create_local_mask(10, window_size=3)

        # Position 5 should attend to [3, 4, 5]
        assert mask[5, 0]  # Too far back
        assert mask[5, 1]  # Too far back
        assert mask[5, 2]  # Too far back
        assert not mask[5, 3]  # In window
        assert not mask[5, 4]  # In window
        assert not mask[5, 5]  # Self (in window)

    def test_local_mask_full_window(self):
        """When window >= seq_len, should be same as causal."""
        seq_len = 5
        mask_local = create_local_mask(seq_len, window_size=seq_len)
        mask_causal = create_causal_mask(seq_len)
        np.testing.assert_array_equal(mask_local, mask_causal)

    def test_local_mask_sparsity(self):
        """Local mask should be sparser than causal for large sequences."""
        seq_len = 100
        window_size = 10

        local_mask = create_local_mask(seq_len, window_size)
        causal_mask = create_causal_mask(seq_len)

        local_sparsity = local_mask.sum() / local_mask.size
        causal_sparsity = causal_mask.sum() / causal_mask.size

        assert local_sparsity > causal_sparsity, \
            "Local mask should be sparser than causal"


class TestStridedMask:
    """Tests for strided (dilated) attention mask."""

    def test_strided_mask_shape(self):
        """Strided mask should have correct shape."""
        mask = create_strided_mask(10, stride=2)
        assert mask.shape == (10, 10)

    def test_strided_mask_includes_causal(self):
        """Strided mask should include causal constraint."""
        mask = create_strided_mask(10, stride=2)
        upper = np.triu(mask, k=1)
        expected_upper = np.triu(np.ones((10, 10), dtype=bool), k=1)
        np.testing.assert_array_equal(upper, expected_upper)

    def test_strided_mask_stride_1(self):
        """Stride 1 should be same as causal (attend to all previous)."""
        mask_strided = create_strided_mask(5, stride=1)
        mask_causal = create_causal_mask(5)
        np.testing.assert_array_equal(mask_strided, mask_causal)

    def test_strided_mask_stride_2(self):
        """Stride 2 should attend to every other position."""
        mask = create_strided_mask(8, stride=2)

        # Position 6 should attend to [0, 2, 4, 6]
        assert not mask[6, 0]  # stride away
        assert mask[6, 1]  # not on stride
        assert not mask[6, 2]  # stride away
        assert mask[6, 3]  # not on stride
        assert not mask[6, 4]  # stride away
        assert mask[6, 5]  # not on stride
        assert not mask[6, 6]  # self

        # Position 7 should attend to [1, 3, 5, 7]
        assert mask[7, 0]  # not on stride
        assert not mask[7, 1]  # stride away
        assert mask[7, 2]  # not on stride
        assert not mask[7, 3]  # stride away
        assert mask[7, 4]  # not on stride
        assert not mask[7, 5]  # stride away
        assert mask[7, 6]  # not on stride
        assert not mask[7, 7]  # self

    def test_strided_mask_always_includes_self(self):
        """Every position should be able to attend to itself."""
        for stride in [1, 2, 3, 4]:
            mask = create_strided_mask(10, stride=stride)
            diagonal = np.diag(mask)
            assert not diagonal.any(), f"Stride {stride}: should attend to self"


class TestBlockMask:
    """Tests for block-sparse attention mask."""

    def test_block_mask_shape(self):
        """Block mask should have correct shape."""
        mask = create_block_mask(10, block_size=4)
        assert mask.shape == (10, 10)

    def test_block_mask_includes_causal(self):
        """Block mask should include causal constraint."""
        mask = create_block_mask(8, block_size=4)
        upper = np.triu(mask, k=1)
        expected_upper = np.triu(np.ones((8, 8), dtype=bool), k=1)
        np.testing.assert_array_equal(upper, expected_upper)

    def test_block_mask_within_block(self):
        """Positions within same block can attend (causally)."""
        mask = create_block_mask(8, block_size=4)

        # Position 2 (block 0) can attend to [0, 1, 2]
        assert not mask[2, 0]
        assert not mask[2, 1]
        assert not mask[2, 2]

        # Position 6 (block 1) can attend to [4, 5, 6]
        assert not mask[6, 4]
        assert not mask[6, 5]
        assert not mask[6, 6]

    def test_block_mask_across_blocks(self):
        """Positions cannot attend across blocks."""
        mask = create_block_mask(8, block_size=4)

        # Position 4 (block 1) cannot attend to block 0
        assert mask[4, 0]
        assert mask[4, 1]
        assert mask[4, 2]
        assert mask[4, 3]

        # Position 4 can attend to itself
        assert not mask[4, 4]

    def test_block_mask_block_size_equals_seq_len(self):
        """Block size = seq_len should be same as causal."""
        seq_len = 8
        mask_block = create_block_mask(seq_len, block_size=seq_len)
        mask_causal = create_causal_mask(seq_len)
        np.testing.assert_array_equal(mask_block, mask_causal)

    def test_block_mask_uneven_blocks(self):
        """Should handle sequence length not divisible by block size."""
        mask = create_block_mask(10, block_size=4)
        # Last block has positions [8, 9]
        # Position 9 can attend to [8, 9]
        assert not mask[9, 8]
        assert not mask[9, 9]
        # Position 9 cannot attend to block 1
        assert mask[9, 4]
        assert mask[9, 5]


class TestCombinedMask:
    """Tests for combined local + strided mask."""

    def test_combined_mask_shape(self):
        """Combined mask should have correct shape."""
        mask = create_combined_mask(16, window_size=4, stride=4)
        assert mask.shape == (16, 16)

    def test_combined_mask_includes_causal(self):
        """Combined mask should include causal constraint."""
        mask = create_combined_mask(16, window_size=4, stride=4)
        upper = np.triu(mask, k=1)
        expected_upper = np.triu(np.ones((16, 16), dtype=bool), k=1)
        np.testing.assert_array_equal(upper, expected_upper)

    def test_combined_mask_includes_local(self):
        """Combined mask should include local attention."""
        mask = create_combined_mask(16, window_size=4, stride=8)
        # Position 10 should attend to local window [7, 8, 9, 10]
        assert not mask[10, 7]
        assert not mask[10, 8]
        assert not mask[10, 9]
        assert not mask[10, 10]

    def test_combined_mask_includes_strided(self):
        """Combined mask should include strided attention."""
        mask = create_combined_mask(16, window_size=2, stride=4)
        # Position 12 should attend to strided [0, 4, 8, 12]
        assert not mask[12, 0]
        assert not mask[12, 4]
        assert not mask[12, 8]
        assert not mask[12, 12]

    def test_combined_union_of_patterns(self):
        """Combined should be union: attend if local OR strided allows."""
        seq_len, window_size, stride = 16, 4, 4
        local_mask = create_local_mask(seq_len, window_size)
        strided_mask = create_strided_mask(seq_len, stride)
        combined_mask = create_combined_mask(seq_len, window_size, stride)

        # Combined allows if either local or strided allows
        expected = local_mask & strided_mask
        np.testing.assert_array_equal(combined_mask, expected)


class TestApplySparseMask:
    """Tests for apply_sparse_mask function."""

    def test_apply_mask_shape(self):
        """Applied mask should preserve shape."""
        scores = np.random.randn(4, 4)
        mask = create_causal_mask(4)
        result = apply_sparse_mask(scores, mask)
        assert result.shape == scores.shape

    def test_apply_mask_unmasked_unchanged(self):
        """Unmasked positions should be unchanged."""
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])
        result = apply_sparse_mask(scores, mask)

        assert result[0, 0] == 1.0
        assert result[1, 0] == 3.0
        assert result[1, 1] == 4.0

    def test_apply_mask_masked_is_neginf(self):
        """Masked positions should be -inf."""
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])
        result = apply_sparse_mask(scores, mask)

        assert result[0, 1] == float('-inf')

    def test_apply_mask_batched(self):
        """Should work with batched input."""
        scores = np.random.randn(2, 4, 4)
        mask = create_causal_mask(4)
        result = apply_sparse_mask(scores, mask)

        assert result.shape == (2, 4, 4)
        # Upper triangle should be -inf
        for b in range(2):
            upper = np.triu(result[b], k=1)
            assert np.all(upper == float('-inf'))

    def test_apply_mask_with_heads(self):
        """Should work with (batch, heads, seq, seq) input."""
        scores = np.random.randn(2, 8, 10, 10)
        mask = create_local_mask(10, window_size=3)
        result = apply_sparse_mask(scores, mask)

        assert result.shape == (2, 8, 10, 10)


class TestComputeSparsity:
    """Tests for compute_sparsity function."""

    def test_sparsity_all_blocked(self):
        """All True mask should have sparsity 1.0."""
        mask = np.ones((4, 4), dtype=bool)
        assert compute_sparsity(mask) == 1.0

    def test_sparsity_none_blocked(self):
        """All False mask should have sparsity 0.0."""
        mask = np.zeros((4, 4), dtype=bool)
        assert compute_sparsity(mask) == 0.0

    def test_sparsity_causal(self):
        """Causal mask sparsity should be ~0.5 for large n."""
        mask = create_causal_mask(100)
        sparsity = compute_sparsity(mask)
        # Upper triangle is n*(n-1)/2 out of n^2
        expected = (100 * 99 / 2) / (100 * 100)
        np.testing.assert_allclose(sparsity, expected, rtol=1e-5)

    def test_sparsity_local_increases_with_sequence(self):
        """Local mask should get sparser as sequence gets longer."""
        window_size = 10
        sparsity_100 = compute_sparsity(create_local_mask(100, window_size))
        sparsity_200 = compute_sparsity(create_local_mask(200, window_size))

        assert sparsity_200 > sparsity_100, \
            "Longer sequence should be sparser with fixed window"


class TestVisualizeMask:
    """Tests for visualize_mask_pattern function."""

    def test_visualize_returns_string(self):
        """Should return a string."""
        mask = create_causal_mask(4)
        result = visualize_mask_pattern(mask)
        assert isinstance(result, str)

    def test_visualize_correct_symbols(self):
        """Should use . for allowed and X for blocked."""
        mask = np.array([[False, True], [False, False]])
        result = visualize_mask_pattern(mask)

        assert '.' in result  # Has allowed positions
        assert 'X' in result  # Has blocked positions

    def test_visualize_causal_pattern(self):
        """Causal mask should show upper triangle blocked."""
        mask = create_causal_mask(3)
        result = visualize_mask_pattern(mask)

        lines = result.strip().split('\n')
        assert len(lines) == 3

        # First row: . X X
        assert lines[0].count('X') == 2
        # Last row: . . .
        assert lines[2].count('X') == 0


class TestMaskProperties:
    """Integration tests for mask properties."""

    def test_all_masks_are_boolean(self):
        """All mask functions should return boolean arrays."""
        seq_len = 10

        masks = [
            create_causal_mask(seq_len),
            create_local_mask(seq_len, 3),
            create_strided_mask(seq_len, 2),
            create_block_mask(seq_len, 4),
            create_combined_mask(seq_len, 3, 2),
        ]

        for mask in masks:
            assert mask.dtype == bool, f"Mask should be boolean, got {mask.dtype}"

    def test_all_masks_block_future(self):
        """All mask variants should block future positions."""
        seq_len = 10

        masks = [
            create_causal_mask(seq_len),
            create_local_mask(seq_len, 3),
            create_strided_mask(seq_len, 2),
            create_block_mask(seq_len, 4),
            create_combined_mask(seq_len, 3, 2),
        ]

        for mask in masks:
            upper = np.triu(mask, k=1)
            expected = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            np.testing.assert_array_equal(
                upper, expected,
                err_msg="All masks should block future positions"
            )

    def test_all_masks_allow_self_attention(self):
        """All masks should allow attending to self."""
        seq_len = 10

        masks = [
            create_causal_mask(seq_len),
            create_local_mask(seq_len, 3),
            create_strided_mask(seq_len, 2),
            create_block_mask(seq_len, 4),
            create_combined_mask(seq_len, 3, 2),
        ]

        for mask in masks:
            diagonal = np.diag(mask)
            assert not diagonal.any(), "All masks should allow self-attention"
