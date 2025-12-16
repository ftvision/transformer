"""
Lab 01: Dot-Product Attention

Implement scaled dot-product attention from scratch.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax values along the specified axis.

    The softmax function converts a vector of values into a probability
    distribution. Each output value is between 0 and 1, and the outputs
    sum to 1 along the specified axis.

    Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)

    IMPORTANT: For numerical stability, subtract the maximum value before
    computing exponentials. This prevents overflow for large values.
    softmax(x) = softmax(x - max(x))

    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax (default: -1)

    Returns:
        Array of same shape as x with softmax applied along axis
        Each slice along axis sums to 1

    Examples:
        >>> softmax(np.array([1.0, 2.0, 3.0]))
        array([0.09003057, 0.24472847, 0.66524096])

        >>> softmax(np.array([[1, 2], [3, 4]]), axis=-1)
        # Each row sums to 1
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement softmax")


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Steps:
    1. Compute attention scores: scores = Q @ K^T
    2. Scale by sqrt(d_k): scores = scores / sqrt(d_k)
    3. Apply mask if provided: scores[mask] = -inf
    4. Apply softmax: attention_weights = softmax(scores)
    5. Compute output: output = attention_weights @ V

    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k)
              True values will be masked (set to -inf before softmax)

    Returns:
        output: Attention output of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)

    Note:
        - d_k is the dimension of queries/keys (last dim of Q, K)
        - d_v is the dimension of values (last dim of V)
        - seq_len_q and seq_len_k can be different (cross-attention)
        - For self-attention, seq_len_q == seq_len_k

    Examples:
        >>> Q = np.random.randn(4, 8)  # 4 queries, dimension 8
        >>> K = np.random.randn(6, 8)  # 6 keys, dimension 8
        >>> V = np.random.randn(6, 10) # 6 values, dimension 10
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
        >>> output.shape
        (4, 10)
        >>> weights.shape
        (4, 6)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement scaled_dot_product_attention")
