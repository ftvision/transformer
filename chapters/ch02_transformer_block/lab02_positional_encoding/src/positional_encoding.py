"""
Lab 02: Positional Encodings

Implement various positional encoding schemes used in transformers.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple, Optional


def sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    This is the original positional encoding from "Attention Is All You Need".
    Each position gets a unique encoding based on sine and cosine functions
    at different frequencies.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Number of positions (sequence length)
        d_model: Dimension of the encoding (must be even)

    Returns:
        Positional encoding matrix of shape (seq_len, d_model)
        Values are bounded between -1 and 1

    Example:
        >>> pe = sinusoidal_encoding(100, 512)
        >>> pe.shape
        (100, 512)
        >>> # Each row is a unique position encoding
        >>> # Values alternate between sin and cos
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create position array: [0, 1, 2, ..., seq_len-1] shape (seq_len, 1)
    # 2. Create dimension array for computing frequencies: [0, 2, 4, ..., d_model-2]
    # 3. Compute frequencies: 1 / (10000 ^ (dim / d_model))
    # 4. Compute angles: position * frequency
    # 5. Apply sin to even indices, cos to odd indices
    raise NotImplementedError("Implement sinusoidal_encoding")


class LearnedPositionalEmbedding:
    """
    Learned positional embeddings.

    Instead of fixed sinusoidal patterns, positions are embedded using
    a learnable embedding table. This is the approach used in GPT and BERT.

    Attributes:
        max_seq_len: Maximum sequence length supported
        d_model: Embedding dimension
        embedding: Learnable embedding matrix of shape (max_seq_len, d_model)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialize learned positional embeddings.

        Args:
            max_seq_len: Maximum sequence length to support
            d_model: Dimension of the embeddings

        Example:
            >>> pe = LearnedPositionalEmbedding(1024, 512)
            >>> pe.embedding.shape
            (1024, 512)
        """
        # YOUR CODE HERE
        # Initialize:
        # - self.max_seq_len
        # - self.d_model
        # - self.embedding: shape (max_seq_len, d_model)
        #   Initialize with small random values (np.random.randn(...) * 0.02)
        raise NotImplementedError("Implement LearnedPositionalEmbedding.__init__")

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional embeddings for a sequence.

        Args:
            seq_len: Length of the sequence (must be <= max_seq_len)

        Returns:
            Positional embeddings of shape (seq_len, d_model)

        Raises:
            ValueError: If seq_len > max_seq_len
        """
        # YOUR CODE HERE
        # Return the first seq_len rows of self.embedding
        raise NotImplementedError("Implement LearnedPositionalEmbedding.forward")

    def __call__(self, seq_len: int) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(seq_len)


def precompute_freqs_cis(
    d_model: int,
    max_seq_len: int,
    base: float = 10000.0
) -> np.ndarray:
    """
    Precompute frequency values for Rotary Position Embeddings (RoPE).

    RoPE encodes position by rotating pairs of dimensions. This function
    precomputes the rotation angles for all positions.

    The rotation for position m and dimension pair i is:
        theta_i = base^(-2i/d_model)
        angle = m * theta_i
        rotation = exp(i * angle) = cos(angle) + i*sin(angle)

    Args:
        d_model: Model dimension (must be even, as we pair dimensions)
        max_seq_len: Maximum sequence length to precompute for
        base: Base for the frequency computation (default: 10000)

    Returns:
        Complex array of shape (max_seq_len, d_model // 2)
        Each entry is exp(i * pos * theta) for that position and dimension pair

    Example:
        >>> freqs = precompute_freqs_cis(64, 100)
        >>> freqs.shape
        (100, 32)
        >>> freqs.dtype
        dtype('complex128')
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute theta for each dimension pair: base^(-2i/d_model) for i in [0, d_model/2)
    # 2. Create position array: [0, 1, ..., max_seq_len-1]
    # 3. Compute angles: outer product of positions and thetas
    # 4. Return complex exponentials: exp(1j * angles)
    raise NotImplementedError("Implement precompute_freqs_cis")


def apply_rotary_emb(
    x: np.ndarray,
    freqs_cis: np.ndarray
) -> np.ndarray:
    """
    Apply rotary position embeddings to input tensor.

    RoPE works by treating consecutive pairs of dimensions as 2D vectors
    and rotating them by position-dependent angles.

    For input [..., d_k] where d_k is even:
    - Pair dimensions (0,1), (2,3), ..., (d_k-2, d_k-1)
    - Rotate each pair by the corresponding angle from freqs_cis

    Args:
        x: Input tensor of shape (..., seq_len, d_k) where d_k is even
           Typically (batch, seq_len, num_heads, d_k) or (batch, num_heads, seq_len, d_k)
        freqs_cis: Precomputed frequencies of shape (seq_len, d_k // 2)
                   Complex numbers representing rotations

    Returns:
        Rotated tensor of same shape as x

    Example:
        >>> x = np.random.randn(2, 10, 8, 64)  # (batch, seq, heads, d_k)
        >>> freqs = precompute_freqs_cis(64, 10)
        >>> rotated = apply_rotary_emb(x, freqs)
        >>> rotated.shape
        (2, 10, 8, 64)

    Note:
        The rotation is applied by:
        1. Viewing x as complex: x_complex[..., i] = x[..., 2i] + 1j * x[..., 2i+1]
        2. Multiplying by freqs_cis: rotated = x_complex * freqs_cis
        3. Converting back to real: separate real and imaginary parts
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Reshape x to pair consecutive dimensions: (..., seq_len, d_k//2, 2)
    # 2. View as complex numbers: x_complex = x[..., 0] + 1j * x[..., 1]
    # 3. Reshape freqs_cis to broadcast correctly
    # 4. Multiply: rotated = x_complex * freqs_cis
    # 5. Convert back: stack real and imaginary parts
    # 6. Reshape to original shape
    raise NotImplementedError("Implement apply_rotary_emb")


def rotate_half(x: np.ndarray) -> np.ndarray:
    """
    Rotate half the hidden dims of the input.

    This is an alternative implementation of RoPE using real arithmetic
    instead of complex numbers.

    For input [..., d], returns [..., d] where:
        output[..., :d//2] = -x[..., d//2:]
        output[..., d//2:] = x[..., :d//2]

    Args:
        x: Input tensor of shape (..., d) where d is even

    Returns:
        Rotated tensor of same shape

    Example:
        >>> x = np.array([1, 2, 3, 4])
        >>> rotate_half(x)
        array([-3, -4, 1, 2])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement rotate_half")


def apply_rotary_emb_real(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray
) -> np.ndarray:
    """
    Apply rotary embeddings using real arithmetic (alternative to complex).

    This is equivalent to apply_rotary_emb but uses the formula:
        rotated = x * cos + rotate_half(x) * sin

    Args:
        x: Input tensor of shape (..., seq_len, d_k)
        cos: Cosine values of shape (seq_len, d_k)
        sin: Sine values of shape (seq_len, d_k)

    Returns:
        Rotated tensor of same shape as x

    Note:
        cos and sin should be precomputed as:
            angles = positions[:, None] * freqs[None, :]
            cos = np.cos(angles)
            sin = np.sin(angles)
        where freqs = 1 / (base ** (2 * np.arange(d_k//2) / d_k))
        and then interleaved to shape (seq_len, d_k)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement apply_rotary_emb_real")
