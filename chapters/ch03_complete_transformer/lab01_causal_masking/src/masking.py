"""
Lab 01: Causal Masking

Implement causal masking for autoregressive attention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (autoregressive) attention mask.

    In causal attention, position i can only attend to positions <= i.
    This prevents "looking into the future" during generation.

    The mask uses the convention: True = masked out, False = allowed.

    Args:
        seq_len: Length of the sequence

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True if position i should NOT attend to position j
        (i.e., j > i)

    Example:
        >>> create_causal_mask(4)
        array([[False,  True,  True,  True],
               [False, False,  True,  True],
               [False, False, False,  True],
               [False, False, False, False]])

    Visual representation (✓ = attend, ✗ = masked):
        Position 0: [✓, ✗, ✗, ✗]  - only sees itself
        Position 1: [✓, ✓, ✗, ✗]  - sees positions 0, 1
        Position 2: [✓, ✓, ✓, ✗]  - sees positions 0, 1, 2
        Position 3: [✓, ✓, ✓, ✓]  - sees all positions
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_causal_mask")


def apply_mask_to_scores(
    scores: np.ndarray,
    mask: np.ndarray,
    mask_value: float = -np.inf
) -> np.ndarray:
    """
    Apply a boolean mask to attention scores.

    Masked positions are set to mask_value (typically -inf).
    After softmax, -inf positions become 0.

    Args:
        scores: Attention scores of shape (..., seq_len_q, seq_len_k)
        mask: Boolean mask where True = masked out
              Shape should be broadcastable to scores
        mask_value: Value to use for masked positions (default: -inf)

    Returns:
        Masked scores with same shape as input scores

    Example:
        >>> scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> mask = np.array([[False, True], [False, False]])
        >>> apply_mask_to_scores(scores, mask)
        array([[ 1., -inf],
               [ 3.,  4.]])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement apply_mask_to_scores")


def create_padding_mask(
    seq_lengths: np.ndarray,
    max_len: int
) -> np.ndarray:
    """
    Create a padding mask for batched sequences of different lengths.

    Positions beyond each sequence's actual length should be masked.

    Args:
        seq_lengths: Array of actual sequence lengths, shape (batch_size,)
        max_len: Maximum sequence length (padded length)

    Returns:
        Boolean mask of shape (batch_size, max_len)
        mask[b, i] = True if position i is padding for sequence b

    Example:
        >>> seq_lengths = np.array([3, 2])  # Two sequences
        >>> create_padding_mask(seq_lengths, max_len=4)
        array([[False, False, False,  True],   # Seq 0: length 3, position 3 is padding
               [False, False,  True,  True]])  # Seq 1: length 2, positions 2,3 are padding

    Note:
        - Position i is padding if i >= seq_lengths[batch_idx]
        - This mask indicates which KEY positions to ignore
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_padding_mask")


def create_padding_mask_2d(
    seq_lengths: np.ndarray,
    max_len: int
) -> np.ndarray:
    """
    Create a 2D padding mask for attention (query x key).

    This is used when both queries and keys have padding.
    A query should not attend to padding positions in keys.

    Args:
        seq_lengths: Array of actual sequence lengths, shape (batch_size,)
        max_len: Maximum sequence length

    Returns:
        Boolean mask of shape (batch_size, max_len, max_len)
        mask[b, i, j] = True if position j is padding for sequence b
        (same mask repeated for each query position)

    Example:
        >>> seq_lengths = np.array([2])  # One sequence of length 2
        >>> create_padding_mask_2d(seq_lengths, max_len=3)
        array([[[False, False,  True],
                [False, False,  True],
                [False, False,  True]]])  # Column 2 is all padding
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_padding_mask_2d")


def combine_masks(
    mask1: np.ndarray,
    mask2: np.ndarray
) -> np.ndarray:
    """
    Combine two masks using logical OR.

    A position is masked if EITHER mask says to mask it.

    Args:
        mask1: First boolean mask
        mask2: Second boolean mask (must be broadcastable with mask1)

    Returns:
        Combined boolean mask

    Example:
        >>> causal = np.array([[False, True], [False, False]])
        >>> padding = np.array([[False, False], [False, True]])
        >>> combine_masks(causal, padding)
        array([[False,  True],
               [False,  True]])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement combine_masks")


def create_full_mask(
    seq_lengths: np.ndarray,
    max_len: int,
    causal: bool = True
) -> np.ndarray:
    """
    Create a complete attention mask combining causal and padding masks.

    This is the mask you'd use in practice for a decoder transformer.

    Args:
        seq_lengths: Array of actual sequence lengths, shape (batch_size,)
        max_len: Maximum sequence length
        causal: Whether to apply causal masking

    Returns:
        Boolean mask of shape (batch_size, max_len, max_len)
        Combines both causal (can't see future) and padding (can't see padding) constraints

    Example:
        >>> seq_lengths = np.array([3])  # One sequence of length 3
        >>> create_full_mask(seq_lengths, max_len=4, causal=True)
        # Combines:
        # - Causal: can't see future tokens
        # - Padding: can't see position 3 (padding)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_full_mask")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax along the specified axis.

    Provided for convenience - handles -inf correctly.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    # Replace -inf max with 0 to avoid nan
    x_max = np.where(np.isinf(x_max), 0, x_max)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def masked_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention with masking.

    This combines everything: scores, masking, softmax, and output.

    Args:
        Q: Queries of shape (..., seq_len_q, d_k)
        K: Keys of shape (..., seq_len_k, d_k)
        V: Values of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask, True = masked out

    Returns:
        output: Attention output of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)

    Example:
        >>> Q = np.random.randn(4, 8)  # 4 queries
        >>> K = np.random.randn(4, 8)  # 4 keys
        >>> V = np.random.randn(4, 8)  # 4 values
        >>> mask = create_causal_mask(4)
        >>> output, weights = masked_attention(Q, K, V, mask)
        >>> # weights[i, j] = 0 for j > i (masked positions)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement masked_attention")
