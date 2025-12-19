"""
Lab 02: KV-Cache Implementation

Implement KV-cache for efficient autoregressive generation.

Your task: Complete the classes and functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, Callable, List


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute scaled dot-product attention.

    Args:
        q: Query tensor, shape (..., seq_len_q, head_dim)
        k: Key tensor, shape (..., seq_len_k, head_dim)
        v: Value tensor, shape (..., seq_len_k, head_dim)
        mask: Optional boolean mask, shape (..., seq_len_q, seq_len_k)
              True values are masked (set to -inf)

    Returns:
        Attention output, shape (..., seq_len_q, head_dim)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Stores K and V tensors for all previous positions to avoid recomputation
    during autoregressive decoding.

    The cache is pre-allocated to max_seq_len to avoid dynamic memory allocation.

    Attributes:
        max_seq_len: Maximum sequence length the cache can hold
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type for cached values
    """

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize the KV cache.

        Pre-allocates buffers for K and V tensors.

        Args:
            max_seq_len: Maximum sequence length to cache
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            dtype: Data type for cached values

        The cache shape will be: (1, num_heads, max_seq_len, head_dim)
        We use batch=1 for simplicity; real implementations handle batches.

        Example:
            >>> cache = KVCache(max_seq_len=2048, num_heads=32, head_dim=128)
            >>> cache.k_cache.shape
            (1, 32, 2048, 128)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement KVCache.__init__")

    def update(
        self,
        k: np.ndarray,
        v: np.ndarray,
        position: int
    ) -> None:
        """
        Update the cache with new K and V values.

        Args:
            k: New key tensor, shape (1, num_heads, seq_len_new, head_dim)
            v: New value tensor, shape (1, num_heads, seq_len_new, head_dim)
            position: Starting position to insert the new values

        Example:
            >>> cache = KVCache(2048, 32, 128)
            >>> k = np.random.randn(1, 32, 10, 128)  # 10 new tokens
            >>> v = np.random.randn(1, 32, 10, 128)
            >>> cache.update(k, v, position=0)
            >>> cache.length
            10
            >>> cache.update(k[:, :, :1, :], v[:, :, :1, :], position=10)
            >>> cache.length
            11
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement KVCache.update")

    def get(self, end_position: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached K and V up to a given position.

        Args:
            end_position: Position to slice up to (exclusive).
                         If None, returns up to current length.

        Returns:
            k_cached: Cached keys, shape (1, num_heads, end_position, head_dim)
            v_cached: Cached values, shape (1, num_heads, end_position, head_dim)

        Example:
            >>> cache = KVCache(2048, 32, 128)
            >>> cache.update(np.ones((1, 32, 5, 128)), np.ones((1, 32, 5, 128)), 0)
            >>> k, v = cache.get()
            >>> k.shape
            (1, 32, 5, 128)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement KVCache.get")

    @property
    def length(self) -> int:
        """
        Current number of cached positions.

        Returns:
            Number of tokens currently in the cache

        Example:
            >>> cache = KVCache(2048, 32, 128)
            >>> cache.length
            0
            >>> cache.update(np.zeros((1, 32, 10, 128)), np.zeros((1, 32, 10, 128)), 0)
            >>> cache.length
            10
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement KVCache.length")

    def reset(self) -> None:
        """
        Clear the cache (reset length to 0).

        Does not reallocate memory, just resets the length counter.

        Example:
            >>> cache = KVCache(2048, 32, 128)
            >>> cache.update(np.zeros((1, 32, 10, 128)), np.zeros((1, 32, 10, 128)), 0)
            >>> cache.length
            10
            >>> cache.reset()
            >>> cache.length
            0
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement KVCache.reset")


def attention_with_kv_cache(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cache: Optional[KVCache] = None,
    position: int = 0
) -> Tuple[np.ndarray, Optional[KVCache]]:
    """
    Compute attention using KV-cache.

    This function handles both prefill (multiple tokens) and decode (single token).

    During prefill:
        - Process all prompt tokens at once
        - Cache their K, V values
        - Each query can only attend to positions <= its own (causal)

    During decode:
        - Process one new token
        - Append its K, V to cache
        - Query attends to all cached positions

    Args:
        q: Query tensor, shape (batch, num_heads, seq_len_q, head_dim)
        k: Key tensor for new tokens, shape (batch, num_heads, seq_len_new, head_dim)
        v: Value tensor for new tokens, shape (batch, num_heads, seq_len_new, head_dim)
        cache: KVCache instance, or None for no caching
        position: Starting position for the new tokens

    Returns:
        output: Attention output, shape (batch, num_heads, seq_len_q, head_dim)
        cache: Updated KVCache (or None if caching disabled)

    Example (Prefill):
        >>> cache = KVCache(2048, 32, 128)
        >>> q = np.random.randn(1, 32, 10, 128)  # 10 prompt tokens
        >>> k = np.random.randn(1, 32, 10, 128)
        >>> v = np.random.randn(1, 32, 10, 128)
        >>> output, cache = attention_with_kv_cache(q, k, v, cache, position=0)
        >>> output.shape
        (1, 32, 10, 128)
        >>> cache.length
        10

    Example (Decode):
        >>> # Continuing from above, generate one new token
        >>> new_q = np.random.randn(1, 32, 1, 128)
        >>> new_k = np.random.randn(1, 32, 1, 128)
        >>> new_v = np.random.randn(1, 32, 1, 128)
        >>> output, cache = attention_with_kv_cache(new_q, new_k, new_v, cache, position=10)
        >>> output.shape
        (1, 32, 1, 128)
        >>> cache.length
        11

    Note:
        - During prefill, apply causal masking so each position only attends to itself and before
        - During decode (seq_len_q=1), no mask needed since query is at the end
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement attention_with_kv_cache")


def count_attention_operations(seq_len: int, with_cache: bool) -> int:
    """
    Count total attention score computations for generating seq_len tokens.

    This demonstrates the O(N²) vs O(N) complexity difference.

    Without cache:
        - Step 1: compute 1 × 1 = 1 attention scores
        - Step 2: compute 2 × 2 = 4 attention scores
        - ...
        - Step N: compute N × N attention scores
        - Total: Σ(i²) for i=1 to N = N(N+1)(2N+1)/6 ≈ O(N³)

        Wait, that's for full self-attention each step. Actually for generation:
        - Step 1: 1 query attends to 1 key = 1
        - Step 2: 1 query attends to 2 keys = 2 (but we recompute all, so 2 queries × 2 keys = 4)

        Let's simplify: count "query-key pairs" in attention matrix

        Without cache (recompute full attention each step):
        - Step i: compute attention for i queries against i keys = i × i
        - Total: Σ(i²) from i=1 to N

    With cache:
        - Step 1: 1 query × 1 key = 1
        - Step 2: 1 query × 2 keys = 2
        - ...
        - Step N: 1 query × N keys = N
        - Total: Σ(i) from i=1 to N = N(N+1)/2 ≈ O(N²)

    Actually, let's use a simpler metric: total positions attended to.

    Without cache (each step recomputes for all tokens):
        - Step i: i tokens each attend to i positions = i² attention operations
        - Total: 1² + 2² + ... + N² = N(N+1)(2N+1)/6

    With cache (only new token needs attention):
        - Step i: 1 new token attends to i positions
        - Total: 1 + 2 + ... + N = N(N+1)/2

    For simplicity, let's just count "number of positions the new token attends to":
    - Without cache: we recompute everything, so N² for step N... that's confusing.

    Let's count differently: "total key positions attended across all steps"

    Without cache (recompute ALL attention each step):
        Each step i, we compute attention for ALL i tokens against ALL i keys.
        But for generation, we only NEED the last token's output.
        So inefficiency is: we computed i-1 extra rows of attention.
        Total unnecessary: Σ(i-1)×i = ... complicated.

    Simpler: count just the "useful" attention computations:
        - Step i: we need 1 query (new token) to attend to i keys
        - Minimum needed: Σi = N(N+1)/2

    With cache: we do exactly the minimum.
    Without cache: we redo everything, so Σ(i²) operations but Σi are useful.

    Let's just count "attention operations for the NEW token each step":
    - With cache: 1 + 2 + 3 + ... + N = N(N+1)/2 (this is the TOTAL)
    - Without cache: also needs 1 + 2 + 3 + ... + N for the new token, but ALSO
      recomputes 0 + 1 + 2 + ... + (N-1) redundant rows = (N-1)N/2 redundant

    Even simpler for pedagogy:
    - With cache: Total = N (one attention per step)
    - Without cache: Total = N(N+1)/2 (recompute increasing amount)

    Args:
        seq_len: Number of tokens to generate
        with_cache: Whether KV-cache is used

    Returns:
        Total number of attention score computations

    Examples:
        >>> count_attention_operations(100, with_cache=True)
        100  # One attention op per token
        >>> count_attention_operations(100, with_cache=False)
        5050  # 1 + 2 + ... + 100 = 100*101/2
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_attention_operations")


def generate_with_cache(
    prompt_length: int,
    max_new_tokens: int,
    num_heads: int = 32,
    head_dim: int = 128,
    attention_fn: Optional[Callable] = None
) -> Tuple[int, List[int]]:
    """
    Simulate generation with KV-cache.

    This function simulates autoregressive generation to demonstrate
    the efficiency of KV-cache. It tracks the number of attention
    operations at each step.

    Args:
        prompt_length: Number of tokens in the prompt
        max_new_tokens: Number of new tokens to generate
        num_heads: Number of attention heads
        head_dim: Dimension per head
        attention_fn: Optional custom attention function (for testing)

    Returns:
        total_ops: Total attention operations performed
        ops_per_step: List of operations per generation step

    Example:
        >>> total, per_step = generate_with_cache(10, 5, num_heads=32, head_dim=128)
        >>> len(per_step)
        6  # 1 prefill + 5 decode steps
        >>> per_step[0]
        55  # Prefill: 1+2+...+10 = 55 (causal attention)
        >>> per_step[1:6]
        [11, 12, 13, 14, 15]  # Decode: attend to 11, 12, 13, 14, 15 positions
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_with_cache")


def generate_without_cache(
    prompt_length: int,
    max_new_tokens: int,
    num_heads: int = 32,
    head_dim: int = 128,
    attention_fn: Optional[Callable] = None
) -> Tuple[int, List[int]]:
    """
    Simulate generation without KV-cache.

    This function simulates the inefficient approach where we recompute
    attention for all tokens at each step.

    Args:
        prompt_length: Number of tokens in the prompt
        max_new_tokens: Number of new tokens to generate
        num_heads: Number of attention heads
        head_dim: Dimension per head
        attention_fn: Optional custom attention function (for testing)

    Returns:
        total_ops: Total attention operations performed
        ops_per_step: List of operations per generation step

    Example:
        >>> total, per_step = generate_without_cache(10, 5, num_heads=32, head_dim=128)
        >>> len(per_step)
        6  # 1 prefill + 5 decode steps
        >>> per_step[0]
        55  # Prefill: 1+2+...+10 = 55 (same as with cache)
        >>> per_step[1:6]
        [66, 78, 91, 105, 120]  # Decode: 1+...+11, 1+...+12, etc. (recompute all!)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_without_cache")
