"""
Lab 01: PagedAttention Simulation

Implement a simplified version of PagedAttention to understand
how vLLM manages KV-cache memory efficiently.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Block:
    """
    A memory block that stores KV vectors for a fixed number of tokens.

    In real PagedAttention, each block stores key/value tensors for
    multiple tokens. Here we simplify to store just the raw vectors.

    Attributes:
        block_id: Unique identifier for this block
        block_size: Maximum number of tokens this block can hold
        num_filled: Number of tokens currently stored
        keys: Array of key vectors, shape (block_size, head_dim)
        values: Array of value vectors, shape (block_size, head_dim)
    """
    block_id: int
    block_size: int
    num_filled: int = 0
    keys: np.ndarray = field(default=None)
    values: np.ndarray = field(default=None)

    def __post_init__(self):
        # Initialize storage if not provided
        if self.keys is None:
            self.keys = np.zeros((self.block_size, 64))  # Default head_dim=64
        if self.values is None:
            self.values = np.zeros((self.block_size, 64))

    def is_full(self) -> bool:
        """Check if block is at capacity."""
        return self.num_filled >= self.block_size

    def slots_available(self) -> int:
        """Return number of empty slots in this block."""
        return self.block_size - self.num_filled


class BlockAllocator:
    """
    Manages allocation and freeing of memory blocks.

    This is analogous to a memory allocator in an operating system.
    Blocks can be allocated to requests and freed when done.

    Attributes:
        num_blocks: Total number of blocks available
        block_size: Number of tokens per block
        head_dim: Dimension of each key/value vector
    """

    def __init__(self, num_blocks: int, block_size: int, head_dim: int = 64):
        """
        Initialize the block allocator.

        Args:
            num_blocks: Total number of blocks in the pool
            block_size: Number of tokens each block can hold
            head_dim: Dimension of key/value vectors

        Example:
            >>> allocator = BlockAllocator(num_blocks=100, block_size=16)
            >>> allocator.num_free_blocks()
            100
        """
        # YOUR CODE HERE
        # 1. Store num_blocks, block_size, head_dim
        # 2. Create all blocks and add to free pool
        # 3. Initialize a set/list of free block IDs
        raise NotImplementedError("Implement __init__")

    def num_free_blocks(self) -> int:
        """
        Return the number of available blocks.

        Returns:
            Count of blocks that can be allocated

        Example:
            >>> allocator = BlockAllocator(num_blocks=10, block_size=16)
            >>> allocator.num_free_blocks()
            10
            >>> block = allocator.allocate()
            >>> allocator.num_free_blocks()
            9
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement num_free_blocks")

    def allocate(self) -> Optional[Block]:
        """
        Allocate a block from the free pool.

        Returns:
            A fresh Block if available, None if pool is exhausted

        The returned block should be reset (num_filled=0, zeroed storage).

        Example:
            >>> allocator = BlockAllocator(num_blocks=2, block_size=16)
            >>> b1 = allocator.allocate()
            >>> b2 = allocator.allocate()
            >>> b3 = allocator.allocate()  # Pool exhausted
            >>> b3 is None
            True
        """
        # YOUR CODE HERE
        # 1. Check if any free blocks available
        # 2. Get a block ID from free pool
        # 3. Reset the block (zero fill, reset num_filled)
        # 4. Return the block
        raise NotImplementedError("Implement allocate")

    def free(self, block: Block) -> None:
        """
        Return a block to the free pool.

        Args:
            block: The block to free

        After freeing, the block ID should be available for future allocation.

        Example:
            >>> allocator = BlockAllocator(num_blocks=2, block_size=16)
            >>> b1 = allocator.allocate()
            >>> allocator.num_free_blocks()
            1
            >>> allocator.free(b1)
            >>> allocator.num_free_blocks()
            2
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement free")

    def can_allocate(self, num_blocks: int) -> bool:
        """
        Check if we can allocate the requested number of blocks.

        Args:
            num_blocks: Number of blocks needed

        Returns:
            True if allocation would succeed
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement can_allocate")


class BlockTable:
    """
    Maps logical block indices to physical blocks for a single request.

    This is like a page table in virtual memory - it translates
    "logical" block positions (0, 1, 2, ...) to actual Block objects.

    The logical view: tokens 0-15 are in "block 0", 16-31 in "block 1", etc.
    The physical reality: these map to arbitrary Block objects in GPU memory.
    """

    def __init__(self, block_size: int):
        """
        Initialize an empty block table.

        Args:
            block_size: Number of tokens per block
        """
        # YOUR CODE HERE
        # 1. Store block_size
        # 2. Initialize empty list for block mappings
        raise NotImplementedError("Implement __init__")

    def append_block(self, block: Block) -> None:
        """
        Add a new block to the end of this table.

        Args:
            block: The physical block to append

        Example:
            >>> table = BlockTable(block_size=16)
            >>> table.append_block(block)
            >>> len(table)
            1
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement append_block")

    def get_block(self, logical_idx: int) -> Block:
        """
        Get the physical block for a logical index.

        Args:
            logical_idx: The logical block index (0, 1, 2, ...)

        Returns:
            The corresponding physical Block

        Raises:
            IndexError: If logical_idx is out of range
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_block")

    def get_all_blocks(self) -> List[Block]:
        """Return all physical blocks in logical order."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_all_blocks")

    def __len__(self) -> int:
        """Return number of blocks in this table."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement __len__")

    def get_token_position(self, token_idx: int) -> Tuple[int, int]:
        """
        Convert token index to (block_idx, position_within_block).

        Args:
            token_idx: Global token position (0, 1, 2, ...)

        Returns:
            Tuple of (logical_block_idx, slot_within_block)

        Example:
            >>> table = BlockTable(block_size=16)
            >>> table.get_token_position(0)
            (0, 0)
            >>> table.get_token_position(17)
            (1, 1)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_token_position")


def paged_attention_forward(
    query: np.ndarray,
    block_table: BlockTable,
    scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention output using paged KV-cache.

    This implements the core PagedAttention computation:
    1. Iterate over all blocks in the block table
    2. For each block, compute attention scores with cached keys
    3. Apply softmax across ALL positions (not per-block)
    4. Compute weighted sum of cached values

    Args:
        query: Query vector of shape (head_dim,) or (num_heads, head_dim)
        block_table: BlockTable containing the KV-cache blocks
        scale: Scaling factor (typically 1/sqrt(head_dim))

    Returns:
        output: Attention output, same shape as query
        attention_weights: Weights over all cached positions

    The challenge: Keys and values are spread across non-contiguous blocks,
    but we need to compute attention as if they were contiguous.

    Example:
        >>> # Query attending to 20 cached tokens (2 blocks of 16)
        >>> query = np.random.randn(64)  # head_dim=64
        >>> output, weights = paged_attention_forward(query, block_table, scale=0.125)
        >>> output.shape
        (64,)
        >>> weights.shape
        (20,)  # One weight per cached token
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Collect all valid keys and values from blocks
    #    - Only use filled slots (up to block.num_filled for each block)
    # 2. Concatenate into contiguous arrays
    # 3. Compute scaled dot-product attention:
    #    scores = query @ keys.T * scale
    #    weights = softmax(scores)
    #    output = weights @ values
    # 4. Return output and weights
    #
    # Hint: Be careful with the last block which may be partially filled
    raise NotImplementedError("Implement paged_attention_forward")


def compute_memory_efficiency(
    num_requests: int,
    avg_tokens_per_request: int,
    max_tokens: int,
    block_size: int
) -> Dict[str, float]:
    """
    Compare memory efficiency of contiguous vs paged allocation.

    This function calculates theoretical memory usage for both approaches
    to demonstrate the efficiency gains of PagedAttention.

    Args:
        num_requests: Number of concurrent requests
        avg_tokens_per_request: Average actual tokens per request
        max_tokens: Maximum possible tokens (what contiguous must allocate)
        block_size: Block size for paged allocation

    Returns:
        Dictionary with:
        - 'contiguous_usage': Memory units used by contiguous allocation
        - 'paged_usage': Memory units used by paged allocation
        - 'efficiency_ratio': paged_usage / contiguous_usage (< 1 is better)
        - 'memory_saved_percent': Percentage of memory saved by paging

    Example:
        >>> result = compute_memory_efficiency(
        ...     num_requests=100,
        ...     avg_tokens_per_request=500,
        ...     max_tokens=2048,
        ...     block_size=16
        ... )
        >>> result['efficiency_ratio'] < 1.0
        True
    """
    # YOUR CODE HERE
    #
    # Contiguous allocation:
    #   Each request reserves max_tokens, regardless of actual usage
    #   Memory = num_requests * max_tokens
    #
    # Paged allocation:
    #   Each request uses ceil(actual_tokens / block_size) blocks
    #   Memory = num_requests * ceil(avg_tokens / block_size) * block_size
    #   Note: There's slight overhead from partially-filled last blocks
    #
    # Calculate both and return the comparison metrics
    raise NotImplementedError("Implement compute_memory_efficiency")


def simulate_prefix_sharing(
    prefix_tokens: int,
    num_requests: int,
    unique_tokens_per_request: int,
    block_size: int
) -> Dict[str, float]:
    """
    Simulate memory savings from shared prefix blocks.

    When multiple requests share a common prefix (e.g., system prompt),
    PagedAttention can share those blocks instead of duplicating them.

    Args:
        prefix_tokens: Number of shared prefix tokens
        num_requests: Number of requests sharing the prefix
        unique_tokens_per_request: Tokens unique to each request
        block_size: Block size for allocation

    Returns:
        Dictionary with:
        - 'without_sharing': Memory if each request has its own prefix
        - 'with_sharing': Memory with shared prefix blocks
        - 'savings_percent': Percentage of memory saved

    Example:
        >>> result = simulate_prefix_sharing(
        ...     prefix_tokens=500,
        ...     num_requests=100,
        ...     unique_tokens_per_request=200,
        ...     block_size=16
        ... )
        >>> result['savings_percent'] > 0
        True
    """
    # YOUR CODE HERE
    #
    # Without sharing:
    #   Each request stores: prefix_tokens + unique_tokens
    #   Total = num_requests * (prefix + unique) blocks
    #
    # With sharing:
    #   Shared prefix: ceil(prefix_tokens / block_size) blocks (stored once)
    #   Per request: ceil(unique_tokens / block_size) blocks
    #   Total = shared_blocks + num_requests * unique_blocks
    raise NotImplementedError("Implement simulate_prefix_sharing")


# Helper function provided for convenience
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax with numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
