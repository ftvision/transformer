"""
Lab 02: Online Softmax

Implement online (incremental) softmax computation.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import List, Tuple


def safe_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable softmax.

    Standard softmax can overflow for large values:
        exp(1000) = inf!

    Safe softmax subtracts the maximum first:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    This is mathematically equivalent but numerically stable.

    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax

    Returns:
        Softmax output, same shape as x

    Example:
        >>> x = np.array([1000, 1001, 1002])
        >>> safe_softmax(x)
        array([0.09003057, 0.24472847, 0.66524096])  # No overflow!
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Find max along axis (keep dims for broadcasting)
    # 2. Subtract max from x
    # 3. Compute exp
    # 4. Divide by sum along axis
    raise NotImplementedError("Implement safe_softmax")


def online_softmax_stats(x_blocks: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute softmax statistics (max and sum) in one pass over blocks.

    This is the core of online softmax. We track:
    - m: running maximum
    - l: running sum of exp(x - m)

    The key insight: when max changes, we rescale the running sum!

    l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))

    Args:
        x_blocks: List of 1D arrays representing blocks of the full vector

    Returns:
        Tuple of (m, l) where:
        - m: global maximum across all blocks
        - l: sum of exp(x - m) across all elements

    Example:
        >>> blocks = [np.array([1, 2, 3]), np.array([4, 5])]
        >>> m, l = online_softmax_stats(blocks)
        >>> m
        5.0
        >>> # l = sum(exp([1,2,3,4,5] - 5))
    """
    # YOUR CODE HERE
    #
    # Initialize:
    #   m = -inf (any real number is larger)
    #   l = 0 (no sum yet)
    #
    # For each block:
    #   1. Find block max: m_block = max(block)
    #   2. Update global max: m_new = max(m, m_block)
    #   3. Rescale running sum: l = l * exp(m - m_new)
    #   4. Add block contribution: l = l + sum(exp(block - m_new))
    #   5. Update m = m_new
    #
    # Return (m, l)
    raise NotImplementedError("Implement online_softmax_stats")


def online_softmax(x_blocks: List[np.ndarray]) -> np.ndarray:
    """
    Compute softmax incrementally over blocks.

    Uses online_softmax_stats to get (m, l), then computes:
        softmax(x) = exp(x - m) / l

    Args:
        x_blocks: List of 1D arrays representing blocks

    Returns:
        Concatenated softmax output (same total length as input)

    Example:
        >>> blocks = [np.array([1, 2]), np.array([3, 4])]
        >>> online_softmax(blocks)
        # Should equal safe_softmax(np.array([1, 2, 3, 4]))
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Call online_softmax_stats to get (m, l)
    # 2. For each block, compute exp(block - m) / l
    # 3. Concatenate results
    raise NotImplementedError("Implement online_softmax")


def online_attention_accumulator(
    Q_row: np.ndarray,
    K_blocks: List[np.ndarray],
    V_blocks: List[np.ndarray],
    d_k: int
) -> np.ndarray:
    """
    Compute attention output for one query row, incrementally.

    This is the heart of Flash Attention! We compute:
        output = softmax(Q @ K^T / sqrt(d_k)) @ V

    But incrementally:
    - Process K, V blocks one at a time
    - Track running max (m), sum (l), and output accumulator (o)
    - Rescale when max changes

    Args:
        Q_row: Single query row of shape (d_k,)
        K_blocks: List of K blocks, each shape (block_size, d_k)
        V_blocks: List of V blocks, each shape (block_size, d_v)
        d_k: Key dimension (for scaling)

    Returns:
        Attention output of shape (d_v,)

    The algorithm:
        Initialize: m = -inf, l = 0, o = zeros(d_v)

        For each (K_block, V_block):
            # Compute scores for this block
            scores = Q_row @ K_block.T / sqrt(d_k)

            # Update running max
            m_block = max(scores)
            m_new = max(m, m_block)

            # Rescale previous accumulator
            scale = exp(m - m_new)
            l = l * scale
            o = o * scale

            # Add this block's contribution
            p = exp(scores - m_new)
            l = l + sum(p)
            o = o + p @ V_block

            m = m_new

        # Final normalization
        return o / l
    """
    # YOUR CODE HERE
    #
    # Follow the algorithm above step by step
    raise NotImplementedError("Implement online_attention_accumulator")


def compare_online_vs_standard(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compare online attention vs standard attention.

    Useful for verifying correctness.

    Args:
        Q: Query matrix (seq_len_q, d_k)
        K: Key matrix (seq_len_k, d_k)
        V: Value matrix (seq_len_k, d_v)
        block_size: Block size for online computation

    Returns:
        Tuple of:
        - online_output: Output from online algorithm
        - standard_output: Output from standard algorithm
        - max_diff: Maximum absolute difference
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute standard attention output
    # 2. Split K, V into blocks
    # 3. Compute online attention for each query row
    # 4. Compare outputs
    raise NotImplementedError("Implement compare_online_vs_standard")
