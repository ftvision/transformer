"""
Lab 02: KV-Cache Management

Implement KV-cache storage and retrieval for efficient autoregressive generation.

Your task: Complete the functions and classes below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuration for KV-cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: np.dtype = np.float32


class KVCache:
    """
    Key-Value cache for transformer inference.

    During autoregressive generation, we cache the key and value projections
    for all previous tokens to avoid redundant computation. This cache stores
    K and V tensors for each layer and provides methods to append and retrieve.

    Memory layout:
        K cache: (num_layers, max_seq_len, num_heads, head_dim)
        V cache: (num_layers, max_seq_len, num_heads, head_dim)
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize empty KV-cache.

        Args:
            config: Cache configuration

        Example:
            >>> config = CacheConfig(num_layers=32, num_heads=32, head_dim=128, max_seq_len=2048)
            >>> cache = KVCache(config)
            >>> cache.current_length
            0
        """
        # YOUR CODE HERE
        # 1. Store config
        # 2. Allocate K and V cache tensors with shape
        #    (num_layers, max_seq_len, num_heads, head_dim)
        # 3. Initialize current_length to 0
        raise NotImplementedError("Implement __init__")

    @property
    def current_length(self) -> int:
        """Return the current number of cached positions."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement current_length")

    def append(
        self,
        layer_idx: int,
        keys: np.ndarray,
        values: np.ndarray
    ) -> None:
        """
        Append new key-value pairs to the cache for a specific layer.

        Args:
            layer_idx: Which layer's cache to update
            keys: New keys of shape (seq_len, num_heads, head_dim)
                  or (num_heads, head_dim) for single token
            values: New values of same shape as keys

        This is called during the forward pass to cache the K/V projections.
        For prefill (prompt processing), seq_len may be > 1.
        For decode (token generation), seq_len is typically 1.

        Example:
            >>> cache.append(layer_idx=0, keys=k, values=v)
            >>> cache.current_length
            1  # or seq_len if multiple tokens appended
        """
        # YOUR CODE HERE
        # 1. Handle both single token (2D) and batch (3D) inputs
        # 2. Check that we don't exceed max_seq_len
        # 3. Copy keys and values to the appropriate cache positions
        # 4. Update current_length
        #
        # Note: For simplicity, assume all layers fill at the same rate
        raise NotImplementedError("Implement append")

    def get(
        self,
        layer_idx: int,
        start: int = 0,
        end: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve cached keys and values for a layer.

        Args:
            layer_idx: Which layer's cache to retrieve
            start: Starting position (default: 0)
            end: Ending position (default: current_length)

        Returns:
            Tuple of (keys, values), each of shape (seq_len, num_heads, head_dim)

        Example:
            >>> k, v = cache.get(layer_idx=0)
            >>> k.shape
            (current_length, num_heads, head_dim)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get")

    def get_all_layers(
        self,
        start: int = 0,
        end: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve cached K/V for all layers at once.

        Args:
            start: Starting position
            end: Ending position

        Returns:
            Tuple of (all_keys, all_values)
            Shape: (num_layers, seq_len, num_heads, head_dim)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_all_layers")

    def clear(self) -> None:
        """Reset the cache to empty state."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement clear")

    def memory_usage_bytes(self) -> int:
        """
        Calculate total memory used by this cache.

        Returns:
            Memory usage in bytes
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement memory_usage_bytes")


def incremental_attention(
    query: np.ndarray,
    kv_cache: KVCache,
    layer_idx: int,
    new_key: np.ndarray,
    new_value: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention for a single new token using cached K/V.

    This demonstrates the efficiency of KV-caching: we only compute the
    query projection for the new token, but attend to ALL previous tokens
    using cached keys and values.

    Args:
        query: Query for new token, shape (num_heads, head_dim)
        kv_cache: KVCache containing previous K/V
        layer_idx: Which layer we're computing
        new_key: Key projection for new token, shape (num_heads, head_dim)
        new_value: Value projection for new token, shape (num_heads, head_dim)

    Returns:
        output: Attention output, shape (num_heads, head_dim)
        attention_weights: Weights over all positions including new token

    Steps:
        1. Append new K/V to cache
        2. Get all cached K/V (now including new token)
        3. Compute attention: output = softmax(Q @ K^T / sqrt(d)) @ V
        4. Return output and weights
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement incremental_attention")


def compute_attention_flops(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    use_cache: bool
) -> int:
    """
    Calculate FLOPs for attention with and without KV-cache.

    This illustrates why KV-caching is essential:
    - Without cache: Must recompute attention over entire sequence
    - With cache: Only compute attention for new token

    Args:
        seq_len: Current sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        use_cache: Whether using KV-cache

    Returns:
        Number of floating point operations

    The key operations in attention:
    - Q @ K^T: seq_len_q * seq_len_k * head_dim multiplications
    - Softmax: seq_len_q * seq_len_k operations (approximately)
    - Weights @ V: seq_len_q * seq_len_k * head_dim multiplications

    Without cache (full recompute):
        seq_len_q = seq_len_k = seq_len

    With cache (incremental):
        seq_len_q = 1 (just the new token)
        seq_len_k = seq_len (all cached + new)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_attention_flops")


def simulate_generation_memory(
    num_tokens: int,
    config: CacheConfig,
    batch_size: int = 1
) -> dict:
    """
    Simulate memory usage during token generation.

    This tracks how KV-cache memory grows as we generate more tokens.

    Args:
        num_tokens: Number of tokens to simulate generating
        config: Cache configuration
        batch_size: Number of sequences in batch

    Returns:
        Dictionary with:
        - 'memory_per_token': Memory in bytes added per token
        - 'total_memory': Total cache memory after generating all tokens
        - 'memory_timeline': List of memory usage after each token

    Example:
        >>> config = CacheConfig(num_layers=32, num_heads=32, head_dim=128, max_seq_len=2048)
        >>> result = simulate_generation_memory(100, config)
        >>> len(result['memory_timeline'])
        100
    """
    # YOUR CODE HERE
    #
    # For each token generated, the cache grows by:
    #   batch_size * num_layers * num_heads * head_dim * 2 (K and V) * dtype_size
    raise NotImplementedError("Implement simulate_generation_memory")


class SlidingWindowCache(KVCache):
    """
    KV-cache with sliding window for long sequences.

    For very long sequences, we can use a sliding window to limit
    memory usage. Only the most recent `window_size` tokens are kept.

    This trades memory for potential quality loss on long-range dependencies.
    """

    def __init__(self, config: CacheConfig, window_size: int):
        """
        Initialize sliding window cache.

        Args:
            config: Cache configuration
            window_size: Maximum number of tokens to keep in window

        Example:
            >>> config = CacheConfig(num_layers=32, num_heads=32, head_dim=128, max_seq_len=2048)
            >>> cache = SlidingWindowCache(config, window_size=512)
        """
        # YOUR CODE HERE
        # Override max_seq_len with window_size
        # Initialize tracking for total tokens seen vs tokens in window
        raise NotImplementedError("Implement __init__")

    def append(
        self,
        layer_idx: int,
        keys: np.ndarray,
        values: np.ndarray
    ) -> None:
        """
        Append new K/V, evicting oldest if window is full.

        When the window is full, this should:
        1. Remove the oldest position
        2. Shift remaining positions left (or use circular buffer)
        3. Add new K/V at the end
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement append")

    @property
    def total_tokens_seen(self) -> int:
        """Return total number of tokens processed (may exceed window_size)."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement total_tokens_seen")


# Helper function provided for convenience
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax with numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
