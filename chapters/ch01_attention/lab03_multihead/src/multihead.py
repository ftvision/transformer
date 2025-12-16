"""
Lab 03: Multi-Head Attention

Implement multi-head attention from scratch.

Your task: Complete the MultiHeadAttention class to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax along the specified axis.

    Provided for convenience - same as Lab 01.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Provided for convenience - same as Lab 01.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V

    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

    Attributes:
        d_model: The model dimension
        num_heads: Number of attention heads
        d_k: Dimension of keys/queries per head (d_model // num_heads)
        d_v: Dimension of values per head (d_model // num_heads)
        W_Q: Query projection matrix (d_model, d_model)
        W_K: Key projection matrix (d_model, d_model)
        W_V: Value projection matrix (d_model, d_model)
        W_O: Output projection matrix (d_model, d_model)
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension. Must be divisible by num_heads.
            num_heads: Number of attention heads.

        Raises:
            ValueError: If d_model is not divisible by num_heads.

        Example:
            >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
            >>> mha.d_k
            64
        """
        # YOUR CODE HERE
        # 1. Validate that d_model is divisible by num_heads
        # 2. Store d_model, num_heads, d_k, d_v
        # 3. Initialize W_Q, W_K, W_V, W_O with small random values
        #    Shape: (d_model, d_model) for each
        #    Use: np.random.randn(...) * 0.02
        raise NotImplementedError("Implement __init__")

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model)

        Returns:
            Tensor of shape (batch, num_heads, seq_len, d_k)
            or (num_heads, seq_len, d_k)

        This reshapes from:
            (batch, seq_len, d_model)
        to:
            (batch, seq_len, num_heads, d_k)
        then transposes to:
            (batch, num_heads, seq_len, d_k)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _split_heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back into d_model dimension.

        Args:
            x: Tensor of shape (batch, num_heads, seq_len, d_k)
               or (num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch, seq_len, d_model)
            or (seq_len, d_model)

        This is the inverse of _split_heads:
            (batch, num_heads, seq_len, d_k)
        transpose to:
            (batch, seq_len, num_heads, d_k)
        then reshape to:
            (batch, seq_len, d_model)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _combine_heads")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute multi-head self-attention.

        For self-attention, Q, K, V are all derived from the same input x.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model) for unbatched input
            mask: Optional attention mask of shape (..., seq_len, seq_len)
                  True values will be masked (set to -inf before softmax)

        Returns:
            Output tensor of same shape as input x

        Steps:
            1. Linear projections: Q = x @ W_Q, K = x @ W_K, V = x @ W_V
            2. Split into heads: reshape and transpose
            3. Apply scaled dot-product attention to each head
            4. Combine heads: transpose and reshape back
            5. Output projection: output = combined @ W_O
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement forward")

    def get_attention_weights(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get attention weights for visualization.

        Same as forward(), but returns attention weights instead of output.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
            or (num_heads, seq_len, seq_len) for unbatched input
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_attention_weights")

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x, mask)
