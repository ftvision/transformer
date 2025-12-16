"""
Lab 01: Dot-Product Attention - SOLUTION

Reference implementation for scaled dot-product attention.
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax values along the specified axis.

    Uses the numerically stable version: softmax(x) = softmax(x - max(x))
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)

    # Normalize to get probabilities
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    # Get the dimension of keys (for scaling)
    d_k = K.shape[-1]

    # Step 1: Compute attention scores
    # Q: (..., seq_len_q, d_k)
    # K: (..., seq_len_k, d_k)
    # scores: (..., seq_len_q, seq_len_k)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1))

    # Step 2: Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask if provided
    # Masked positions get -inf so softmax gives them 0 weight
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    # Step 4: Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Step 5: Compute weighted sum of values
    # attention_weights: (..., seq_len_q, seq_len_k)
    # V: (..., seq_len_k, d_v)
    # output: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, V)

    return output, attention_weights
