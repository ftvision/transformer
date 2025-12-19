"""
Lab 01: Sparse Attention Patterns

Implement various sparse attention mask patterns.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (autoregressive) attention mask.

    In causal attention, each position can only attend to itself and
    previous positions (not future positions).

    Args:
        seq_len: Length of the sequence

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True means position i CANNOT attend to position j
        (i.e., j > i for causal masking)

    Example:
        >>> create_causal_mask(4)
        array([[False,  True,  True,  True],
               [False, False,  True,  True],
               [False, False, False,  True],
               [False, False, False, False]])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_causal_mask")


def create_local_mask(seq_len: int, window_size: int) -> np.ndarray:
    """
    Create a local (sliding window) attention mask with causal constraint.

    Each position can attend to at most `window_size` previous positions
    (including itself). This is combined with causal masking.

    Args:
        seq_len: Length of the sequence
        window_size: Number of positions each token can attend to
                    (including itself and previous positions)

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True means position i CANNOT attend to position j

    Example (seq_len=6, window_size=3):
        Position 0: attends to [0]           (window clips at start)
        Position 1: attends to [0, 1]        (window clips at start)
        Position 2: attends to [0, 1, 2]     (full window)
        Position 3: attends to [1, 2, 3]     (slides forward)
        Position 4: attends to [2, 3, 4]     (slides forward)
        Position 5: attends to [3, 4, 5]     (slides forward)

    The resulting mask (False=can attend, True=blocked):
        [[F T T T T T]
         [F F T T T T]
         [F F F T T T]
         [T F F F T T]
         [T T F F F T]
         [T T T F F F]]
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create base causal mask
    # 2. For each position i, also mask positions j where (i - j) >= window_size
    # 3. Combine both constraints
    raise NotImplementedError("Implement create_local_mask")


def create_strided_mask(seq_len: int, stride: int) -> np.ndarray:
    """
    Create a strided (dilated) attention mask with causal constraint.

    Each position attends to every `stride`-th previous position.
    This allows attending to distant positions efficiently.

    Args:
        seq_len: Length of the sequence
        stride: Attend to every stride-th position

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True means position i CANNOT attend to position j

    Example (seq_len=8, stride=2):
        Position 6 attends to: [0, 2, 4, 6] (every 2nd position up to itself)
        Position 7 attends to: [1, 3, 5, 7] (every 2nd position up to itself)

    Note: Position i attends to position j if:
        - j <= i (causal)
        - (i - j) % stride == 0

    Example mask (stride=2):
        Position 0: [0]
        Position 1: [1]
        Position 2: [0, 2]
        Position 3: [1, 3]
        Position 4: [0, 2, 4]
        Position 5: [1, 3, 5]
        ...
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create indices: i, j where i is query position, j is key position
    # 2. Attend where j <= i (causal) AND (i - j) % stride == 0
    # 3. Invert to get mask (True = blocked)
    raise NotImplementedError("Implement create_strided_mask")


def create_block_mask(seq_len: int, block_size: int) -> np.ndarray:
    """
    Create a block-sparse attention mask with causal constraint.

    The sequence is divided into blocks of size `block_size`.
    Positions can only attend to positions within the same block
    (with causal constraint within the block).

    Args:
        seq_len: Length of the sequence
        block_size: Size of each attention block

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True means position i CANNOT attend to position j

    Example (seq_len=8, block_size=4):
        Block 0: positions [0, 1, 2, 3] - attend within block (causally)
        Block 1: positions [4, 5, 6, 7] - attend within block (causally)

        Position 5 attends to: [4, 5] (same block, causal)
        Position 5 does NOT attend to: [0, 1, 2, 3] (different block)

    Note: If seq_len is not divisible by block_size, the last block
          may be smaller.
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Determine which block each position belongs to: block_id = pos // block_size
    # 2. Positions can attend if same block AND causal (j <= i)
    # 3. Invert to get mask
    raise NotImplementedError("Implement create_block_mask")


def create_combined_mask(
    seq_len: int,
    window_size: int,
    stride: int
) -> np.ndarray:
    """
    Create a combined local + strided attention mask.

    This pattern allows each position to:
    1. Attend locally within a window (for nearby context)
    2. Attend to strided positions (for long-range dependencies)

    Both are combined with causal masking.

    Args:
        seq_len: Length of the sequence
        window_size: Size of local attention window
        stride: Stride for long-range attention

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True means position i CANNOT attend to position j

    Example (seq_len=16, window_size=4, stride=4):
        Position 10 attends to:
        - Local: [7, 8, 9, 10] (window of 4)
        - Strided: [2, 6, 10] (every 4th position)
        - Combined: [2, 6, 7, 8, 9, 10]

    This is similar to patterns used in Sparse Transformer (OpenAI).
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create local mask
    # 2. Create strided mask
    # 3. A position can attend if EITHER local OR strided allows it
    # 4. Combined mask: blocked only if BOTH patterns block it
    raise NotImplementedError("Implement create_combined_mask")


def apply_sparse_mask(
    attention_scores: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply a sparse attention mask to attention scores.

    Masked positions (where mask=True) are set to -inf, which
    causes them to become 0 after softmax.

    Args:
        attention_scores: Raw attention scores of shape (..., seq_len, seq_len)
        mask: Boolean mask of shape (seq_len, seq_len) or broadcastable

    Returns:
        Masked attention scores with same shape as input
        Masked positions have value -inf

    Example:
        >>> scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> mask = np.array([[False, True], [False, False]])
        >>> apply_sparse_mask(scores, mask)
        array([[ 1., -inf],
               [ 3.,  4.]])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement apply_sparse_mask")


def compute_sparsity(mask: np.ndarray) -> float:
    """
    Compute the sparsity of an attention mask.

    Sparsity is the fraction of positions that are masked (blocked).
    Higher sparsity = more efficient attention (fewer computations needed).

    Args:
        mask: Boolean attention mask of shape (seq_len, seq_len)

    Returns:
        Fraction of positions that are masked (between 0 and 1)

    Example:
        >>> mask = np.array([[False, True], [False, False]])
        >>> compute_sparsity(mask)
        0.25  # 1 out of 4 positions masked
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_sparsity")


def visualize_mask_pattern(mask: np.ndarray) -> str:
    """
    Create an ASCII visualization of an attention mask.

    Useful for debugging and understanding patterns.

    Args:
        mask: Boolean attention mask of shape (seq_len, seq_len)

    Returns:
        String representation where:
        - '.' represents positions that CAN be attended to (False in mask)
        - 'X' represents blocked positions (True in mask)

    Example:
        >>> mask = create_causal_mask(4)
        >>> print(visualize_mask_pattern(mask))
        . X X X
        . . X X
        . . . X
        . . . .
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement visualize_mask_pattern")
