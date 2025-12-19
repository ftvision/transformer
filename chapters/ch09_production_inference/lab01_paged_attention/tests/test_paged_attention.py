"""Tests for Lab 01: PagedAttention Simulation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paged_attention import (
    Block,
    BlockAllocator,
    BlockTable,
    paged_attention_forward,
    compute_memory_efficiency,
    simulate_prefix_sharing,
)


class TestBlock:
    """Tests for the Block class."""

    def test_block_initialization(self):
        """Block should initialize with correct attributes."""
        block = Block(block_id=0, block_size=16)
        assert block.block_id == 0
        assert block.block_size == 16
        assert block.num_filled == 0
        assert block.keys.shape[0] == 16
        assert block.values.shape[0] == 16

    def test_block_is_full(self):
        """Block should report full status correctly."""
        block = Block(block_id=0, block_size=4)
        assert not block.is_full()

        block.num_filled = 4
        assert block.is_full()

    def test_block_slots_available(self):
        """Block should report available slots correctly."""
        block = Block(block_id=0, block_size=16)
        assert block.slots_available() == 16

        block.num_filled = 10
        assert block.slots_available() == 6


class TestBlockAllocator:
    """Tests for the BlockAllocator class."""

    def test_allocator_initialization(self):
        """Allocator should start with all blocks free."""
        allocator = BlockAllocator(num_blocks=10, block_size=16)
        assert allocator.num_free_blocks() == 10

    def test_allocate_single_block(self):
        """Should be able to allocate a single block."""
        allocator = BlockAllocator(num_blocks=10, block_size=16)
        block = allocator.allocate()

        assert block is not None
        assert allocator.num_free_blocks() == 9

    def test_allocate_all_blocks(self):
        """Should be able to allocate all blocks."""
        allocator = BlockAllocator(num_blocks=5, block_size=16)
        blocks = []

        for _ in range(5):
            block = allocator.allocate()
            assert block is not None
            blocks.append(block)

        assert allocator.num_free_blocks() == 0

        # All block IDs should be unique
        block_ids = [b.block_id for b in blocks]
        assert len(set(block_ids)) == 5

    def test_allocate_exhausted(self):
        """Should return None when pool is exhausted."""
        allocator = BlockAllocator(num_blocks=2, block_size=16)
        allocator.allocate()
        allocator.allocate()

        result = allocator.allocate()
        assert result is None

    def test_free_block(self):
        """Freed blocks should be available for reallocation."""
        allocator = BlockAllocator(num_blocks=2, block_size=16)
        b1 = allocator.allocate()
        b2 = allocator.allocate()

        assert allocator.num_free_blocks() == 0

        allocator.free(b1)
        assert allocator.num_free_blocks() == 1

        allocator.free(b2)
        assert allocator.num_free_blocks() == 2

    def test_reallocate_freed_block(self):
        """Should be able to allocate after freeing."""
        allocator = BlockAllocator(num_blocks=1, block_size=16)
        b1 = allocator.allocate()
        assert allocator.allocate() is None

        allocator.free(b1)
        b2 = allocator.allocate()
        assert b2 is not None

    def test_can_allocate(self):
        """can_allocate should correctly predict allocation success."""
        allocator = BlockAllocator(num_blocks=5, block_size=16)

        assert allocator.can_allocate(5)
        assert allocator.can_allocate(3)
        assert not allocator.can_allocate(6)

        allocator.allocate()
        allocator.allocate()
        assert allocator.can_allocate(3)
        assert not allocator.can_allocate(4)

    def test_allocated_block_is_clean(self):
        """Allocated blocks should have num_filled=0."""
        allocator = BlockAllocator(num_blocks=2, block_size=16)
        b1 = allocator.allocate()
        b1.num_filled = 10  # Use the block

        allocator.free(b1)
        b2 = allocator.allocate()

        assert b2.num_filled == 0, "Reallocated block should be reset"


class TestBlockTable:
    """Tests for the BlockTable class."""

    def test_empty_block_table(self):
        """Empty block table should have length 0."""
        table = BlockTable(block_size=16)
        assert len(table) == 0

    def test_append_block(self):
        """Should be able to append blocks to table."""
        table = BlockTable(block_size=16)
        block = Block(block_id=0, block_size=16)

        table.append_block(block)
        assert len(table) == 1

    def test_get_block(self):
        """Should retrieve correct block by logical index."""
        table = BlockTable(block_size=16)
        b1 = Block(block_id=5, block_size=16)
        b2 = Block(block_id=3, block_size=16)

        table.append_block(b1)
        table.append_block(b2)

        assert table.get_block(0).block_id == 5
        assert table.get_block(1).block_id == 3

    def test_get_block_out_of_range(self):
        """Should raise IndexError for invalid logical index."""
        table = BlockTable(block_size=16)
        table.append_block(Block(block_id=0, block_size=16))

        with pytest.raises(IndexError):
            table.get_block(5)

    def test_get_all_blocks(self):
        """Should return all blocks in order."""
        table = BlockTable(block_size=16)
        blocks = [Block(block_id=i, block_size=16) for i in range(3)]

        for b in blocks:
            table.append_block(b)

        result = table.get_all_blocks()
        assert len(result) == 3
        assert [b.block_id for b in result] == [0, 1, 2]

    def test_get_token_position(self):
        """Should correctly map token index to block and slot."""
        table = BlockTable(block_size=16)

        # Token 0 -> block 0, slot 0
        assert table.get_token_position(0) == (0, 0)

        # Token 15 -> block 0, slot 15
        assert table.get_token_position(15) == (0, 15)

        # Token 16 -> block 1, slot 0
        assert table.get_token_position(16) == (1, 0)

        # Token 33 -> block 2, slot 1
        assert table.get_token_position(33) == (2, 1)


class TestPagedAttentionForward:
    """Tests for the paged attention computation."""

    def test_attention_output_shape(self):
        """Output should match query shape."""
        head_dim = 64
        block_size = 16
        scale = 1.0 / np.sqrt(head_dim)

        # Create block table with one partially filled block
        allocator = BlockAllocator(num_blocks=10, block_size=block_size, head_dim=head_dim)
        block = allocator.allocate()
        block.keys[:8] = np.random.randn(8, head_dim)
        block.values[:8] = np.random.randn(8, head_dim)
        block.num_filled = 8

        table = BlockTable(block_size=block_size)
        table.append_block(block)

        query = np.random.randn(head_dim)
        output, weights = paged_attention_forward(query, table, scale)

        assert output.shape == (head_dim,)
        assert weights.shape == (8,)  # 8 filled positions

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        head_dim = 64
        block_size = 16
        scale = 1.0 / np.sqrt(head_dim)

        allocator = BlockAllocator(num_blocks=10, block_size=block_size, head_dim=head_dim)
        block = allocator.allocate()
        block.keys[:10] = np.random.randn(10, head_dim)
        block.values[:10] = np.random.randn(10, head_dim)
        block.num_filled = 10

        table = BlockTable(block_size=block_size)
        table.append_block(block)

        query = np.random.randn(head_dim)
        _, weights = paged_attention_forward(query, table, scale)

        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-5)

    def test_attention_multiple_blocks(self):
        """Should work with multiple blocks."""
        head_dim = 64
        block_size = 16
        scale = 1.0 / np.sqrt(head_dim)

        allocator = BlockAllocator(num_blocks=10, block_size=block_size, head_dim=head_dim)

        # First block: fully filled
        b1 = allocator.allocate()
        b1.keys[:] = np.random.randn(block_size, head_dim)
        b1.values[:] = np.random.randn(block_size, head_dim)
        b1.num_filled = block_size

        # Second block: partially filled
        b2 = allocator.allocate()
        b2.keys[:5] = np.random.randn(5, head_dim)
        b2.values[:5] = np.random.randn(5, head_dim)
        b2.num_filled = 5

        table = BlockTable(block_size=block_size)
        table.append_block(b1)
        table.append_block(b2)

        query = np.random.randn(head_dim)
        output, weights = paged_attention_forward(query, table, scale)

        # Should attend to all 21 positions (16 + 5)
        assert weights.shape == (21,)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-5)

    def test_attention_matches_standard(self):
        """Paged attention should match standard attention computation."""
        head_dim = 64
        block_size = 8
        num_tokens = 12
        scale = 1.0 / np.sqrt(head_dim)

        # Create contiguous K, V
        keys = np.random.randn(num_tokens, head_dim)
        values = np.random.randn(num_tokens, head_dim)
        query = np.random.randn(head_dim)

        # Standard attention
        scores = query @ keys.T * scale
        weights_standard = np.exp(scores - scores.max())
        weights_standard = weights_standard / weights_standard.sum()
        output_standard = weights_standard @ values

        # Paged attention with same data
        allocator = BlockAllocator(num_blocks=10, block_size=block_size, head_dim=head_dim)

        b1 = allocator.allocate()
        b1.keys[:8] = keys[:8]
        b1.values[:8] = values[:8]
        b1.num_filled = 8

        b2 = allocator.allocate()
        b2.keys[:4] = keys[8:12]
        b2.values[:4] = values[8:12]
        b2.num_filled = 4

        table = BlockTable(block_size=block_size)
        table.append_block(b1)
        table.append_block(b2)

        output_paged, weights_paged = paged_attention_forward(query, table, scale)

        # Should match standard attention
        np.testing.assert_allclose(output_paged, output_standard, rtol=1e-5)
        np.testing.assert_allclose(weights_paged, weights_standard, rtol=1e-5)


class TestMemoryEfficiency:
    """Tests for memory efficiency calculations."""

    def test_paged_more_efficient(self):
        """Paged allocation should be more efficient than contiguous."""
        result = compute_memory_efficiency(
            num_requests=100,
            avg_tokens_per_request=500,
            max_tokens=2048,
            block_size=16
        )

        assert result['efficiency_ratio'] < 1.0
        assert result['memory_saved_percent'] > 0
        assert result['paged_usage'] < result['contiguous_usage']

    def test_contiguous_usage_calculation(self):
        """Contiguous usage should be num_requests * max_tokens."""
        result = compute_memory_efficiency(
            num_requests=10,
            avg_tokens_per_request=100,
            max_tokens=1000,
            block_size=16
        )

        expected_contiguous = 10 * 1000
        assert result['contiguous_usage'] == expected_contiguous

    def test_efficiency_with_full_utilization(self):
        """When avg_tokens == max_tokens, savings should be minimal."""
        result = compute_memory_efficiency(
            num_requests=10,
            avg_tokens_per_request=1000,
            max_tokens=1000,
            block_size=16
        )

        # With full utilization, paged might even be slightly worse
        # due to block alignment overhead
        assert result['efficiency_ratio'] >= 0.9  # At most 10% worse

    def test_efficiency_with_low_utilization(self):
        """With low utilization, savings should be substantial."""
        result = compute_memory_efficiency(
            num_requests=100,
            avg_tokens_per_request=50,
            max_tokens=2048,
            block_size=16
        )

        # With 50/2048 utilization, should save ~97% memory
        assert result['memory_saved_percent'] > 90


class TestPrefixSharing:
    """Tests for prefix sharing simulation."""

    def test_sharing_saves_memory(self):
        """Prefix sharing should save memory."""
        result = simulate_prefix_sharing(
            prefix_tokens=500,
            num_requests=100,
            unique_tokens_per_request=200,
            block_size=16
        )

        assert result['savings_percent'] > 0
        assert result['with_sharing'] < result['without_sharing']

    def test_no_sharing_baseline(self):
        """Without sharing, memory should scale with num_requests."""
        result = simulate_prefix_sharing(
            prefix_tokens=100,
            num_requests=10,
            unique_tokens_per_request=100,
            block_size=16
        )

        # Each request has 200 tokens = 13 blocks (ceil(200/16))
        # Total without sharing: 10 * 13 = 130 blocks worth
        expected_without = 10 * (np.ceil(200 / 16) * 16)
        assert result['without_sharing'] == expected_without

    def test_sharing_with_large_prefix(self):
        """Large shared prefix should yield significant savings."""
        result = simulate_prefix_sharing(
            prefix_tokens=1000,  # Large prefix
            num_requests=100,
            unique_tokens_per_request=100,  # Small unique part
            block_size=16
        )

        # Shared prefix is 1000 tokens, unique is only 100
        # Without sharing: 100 * (1000 + 100) = 110,000 tokens
        # With sharing: 1000 + 100*100 = 11,000 tokens
        # Should save ~90%
        assert result['savings_percent'] > 85

    def test_sharing_single_request(self):
        """With single request, no sharing benefit."""
        result = simulate_prefix_sharing(
            prefix_tokens=500,
            num_requests=1,
            unique_tokens_per_request=200,
            block_size=16
        )

        # With only 1 request, sharing doesn't help
        assert result['with_sharing'] == result['without_sharing']
        assert result['savings_percent'] == 0
