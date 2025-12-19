"""
Lab 02: Token Embeddings

Implement token and positional embeddings for transformers.

Your task: Complete the classes below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional


class TokenEmbedding:
    """
    Token embedding layer.

    Maps discrete token IDs to dense vectors. This is essentially a
    learnable lookup table.

    Attributes:
        vocab_size: Number of unique tokens in vocabulary
        d_model: Dimension of embedding vectors
        weight: Embedding matrix of shape (vocab_size, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize token embedding.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            d_model: Dimension of embedding vectors

        The weight matrix should be initialized with small random values.
        Use: np.random.randn(vocab_size, d_model) * 0.02
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TokenEmbedding.__init__")

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for token IDs.

        This is a simple lookup operation: for each token ID, return
        the corresponding row from the weight matrix.

        Args:
            token_ids: Integer array of token IDs
                      Shape: (seq_len,) or (batch, seq_len)
                      Values should be in range [0, vocab_size)

        Returns:
            Embeddings array
            Shape: (seq_len, d_model) or (batch, seq_len, d_model)

        Example:
            >>> embed = TokenEmbedding(vocab_size=100, d_model=64)
            >>> ids = np.array([[1, 2, 3], [4, 5, 6]])
            >>> output = embed.forward(ids)
            >>> output.shape
            (2, 3, 64)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TokenEmbedding.forward")

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(token_ids)


class PositionalEmbedding:
    """
    Learned positional embedding layer.

    Maps position indices to dense vectors. Unlike sinusoidal encodings,
    these are learned during training (but we initialize randomly here).

    Used by GPT-2 and many modern models.

    Attributes:
        max_seq_len: Maximum sequence length supported
        d_model: Dimension of embedding vectors
        weight: Embedding matrix of shape (max_seq_len, d_model)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialize positional embedding.

        Args:
            max_seq_len: Maximum sequence length this embedding supports
            d_model: Dimension of embedding vectors

        Initialize weight with small random values:
        np.random.randn(max_seq_len, d_model) * 0.02
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement PositionalEmbedding.__init__")

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional embeddings for a sequence.

        Returns the first seq_len rows of the weight matrix.

        Args:
            seq_len: Length of sequence (must be <= max_seq_len)

        Returns:
            Positional embeddings of shape (seq_len, d_model)

        Raises:
            ValueError: If seq_len > max_seq_len

        Example:
            >>> pos_embed = PositionalEmbedding(max_seq_len=512, d_model=64)
            >>> positions = pos_embed.forward(10)
            >>> positions.shape
            (10, 64)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement PositionalEmbedding.forward")

    def __call__(self, seq_len: int) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(seq_len)


class SinusoidalPositionalEncoding:
    """
    Sinusoidal positional encoding (fixed, not learned).

    Uses sine and cosine functions of different frequencies to encode
    position information. This is the original positional encoding from
    "Attention Is All You Need".

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
        - Each position has a unique encoding
        - Nearby positions have similar encodings
        - Can extrapolate to longer sequences than seen during training

    Attributes:
        max_seq_len: Maximum sequence length (for precomputation)
        d_model: Dimension of encoding vectors
        encoding: Precomputed encodings of shape (max_seq_len, d_model)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialize and precompute sinusoidal encodings.

        Args:
            max_seq_len: Maximum sequence length to precompute
            d_model: Dimension of encoding vectors (should be even)

        The encoding is computed as:
            positions = [0, 1, 2, ..., max_seq_len-1]
            div_term = 10000^(2i/d_model) for i in [0, 1, ..., d_model/2-1]

            encoding[:, 0::2] = sin(positions / div_term)  # Even indices
            encoding[:, 1::2] = cos(positions / div_term)  # Odd indices
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement SinusoidalPositionalEncoding.__init__")

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get sinusoidal encodings for a sequence.

        Args:
            seq_len: Length of sequence

        Returns:
            Positional encodings of shape (seq_len, d_model)

        Example:
            >>> pe = SinusoidalPositionalEncoding(max_seq_len=512, d_model=64)
            >>> encodings = pe.forward(10)
            >>> encodings.shape
            (10, 64)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement SinusoidalPositionalEncoding.forward")

    def __call__(self, seq_len: int) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(seq_len)


class TransformerEmbedding:
    """
    Complete transformer embedding layer.

    Combines token embeddings with positional embeddings (or encodings)
    to produce the input representation for transformer layers.

    Output = TokenEmbed(tokens) + PositionalEmbed(positions)

    Optionally applies dropout for regularization.

    Attributes:
        token_embedding: TokenEmbedding layer
        positional_embedding: PositionalEmbedding or SinusoidalPositionalEncoding
        dropout_rate: Dropout probability (0 = no dropout)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_sinusoidal: bool = False
    ):
        """
        Initialize transformer embedding.

        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout rate (0 to disable)
            use_sinusoidal: If True, use sinusoidal positional encoding
                           If False, use learned positional embedding
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TransformerEmbedding.__init__")

    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Convert token IDs to embeddings with positional information.

        Args:
            token_ids: Integer array of token IDs
                      Shape: (batch, seq_len) or (seq_len,)
            training: If True and dropout > 0, apply dropout

        Returns:
            Embeddings of shape (batch, seq_len, d_model) or (seq_len, d_model)

        Process:
            1. Look up token embeddings
            2. Get positional embeddings for sequence length
            3. Add token + positional embeddings
            4. Apply dropout if training

        Example:
            >>> embed = TransformerEmbedding(vocab_size=50000, d_model=768, max_seq_len=1024)
            >>> ids = np.array([[101, 2054, 2003, 102]])
            >>> output = embed.forward(ids)
            >>> output.shape
            (1, 4, 768)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TransformerEmbedding.forward")

    def __call__(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(token_ids, training)


def dropout(x: np.ndarray, rate: float, training: bool = True) -> np.ndarray:
    """
    Apply dropout to input array.

    During training, randomly sets elements to 0 with probability `rate`,
    and scales remaining elements by 1/(1-rate).

    During inference (training=False), returns input unchanged.

    Args:
        x: Input array
        rate: Dropout probability (0 to 1)
        training: Whether in training mode

    Returns:
        Array with dropout applied (if training) or unchanged (if not)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement dropout")


class OutputProjection:
    """
    Output projection layer (language model head).

    Maps hidden states back to vocabulary logits.
    Can optionally share weights with token embedding (tied embeddings).

    Output: hidden_states @ weight.T  (shape: batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tied_weight: Optional[np.ndarray] = None
    ):
        """
        Initialize output projection.

        Args:
            d_model: Dimension of input hidden states
            vocab_size: Size of vocabulary (output dimension)
            tied_weight: If provided, use this as the projection weight
                        (for tied embeddings). Shape: (vocab_size, d_model)
                        If None, initialize a new weight matrix.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement OutputProjection.__init__")

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden_states: Shape (batch, seq_len, d_model) or (seq_len, d_model)

        Returns:
            Logits of shape (batch, seq_len, vocab_size) or (seq_len, vocab_size)

        Note: For tied embeddings, this is: hidden_states @ token_embedding.weight.T
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement OutputProjection.forward")

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(hidden_states)
