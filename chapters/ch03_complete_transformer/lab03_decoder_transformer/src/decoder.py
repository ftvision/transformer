"""
Lab 03: Decoder-Only Transformer

Build a complete GPT-style decoder transformer.

Your task: Complete the classes below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, List


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isinf(x_max), 0, x_max)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.

    Approximation used in GPT-2:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def dropout(x: np.ndarray, rate: float, training: bool = True) -> np.ndarray:
    """Apply dropout with scaling."""
    if not training or rate == 0:
        return x
    mask = np.random.binomial(1, 1 - rate, x.shape)
    return x * mask / (1 - rate)


class LayerNorm:
    """
    Layer Normalization.

    Normalizes inputs to have zero mean and unit variance over the last
    dimension, then applies a learnable scale (gamma) and shift (beta).

    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta

    Unlike BatchNorm, LayerNorm normalizes over features (not batch),
    making it suitable for variable-length sequences.

    Attributes:
        d_model: Dimension of input features
        eps: Small constant for numerical stability
        gamma: Learnable scale parameter (initialized to 1)
        beta: Learnable shift parameter (initialized to 0)
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize layer normalization.

        Args:
            d_model: Dimension of input features (last dimension)
            eps: Small value for numerical stability in division
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement LayerNorm.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape

        Steps:
            1. Compute mean over last dimension
            2. Compute variance over last dimension
            3. Normalize: (x - mean) / sqrt(var + eps)
            4. Scale and shift: gamma * normalized + beta
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement LayerNorm.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class FeedForward:
    """
    Position-wise Feed-Forward Network.

    A two-layer MLP applied independently to each position.
    First expands the dimension (d_model → d_ff), applies activation,
    then contracts back (d_ff → d_model).

    FFN(x) = Linear2(activation(Linear1(x)))

    Typically d_ff = 4 * d_model.

    Attributes:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        W1: First linear layer weights (d_model, d_ff)
        b1: First linear layer bias (d_ff,)
        W2: Second linear layer weights (d_ff, d_model)
        b2: Second linear layer bias (d_model,)
        dropout_rate: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Input and output dimension
            d_ff: Hidden layer dimension
            dropout: Dropout probability after activation
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FeedForward.__init__")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply feed-forward network.

        Args:
            x: Input tensor of shape (..., d_model)
            training: Whether to apply dropout

        Returns:
            Output tensor of same shape as input

        Steps:
            1. Linear projection: x @ W1 + b1
            2. GELU activation
            3. Dropout (if training)
            4. Linear projection: hidden @ W2 + b2
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FeedForward.forward")

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(x, training)


class MultiHeadAttention:
    """
    Multi-Head Self-Attention.

    Same as Chapter 1, Lab 03 - provided here for convenience.
    You can copy your implementation or use this one.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MultiHeadAttention.__init__")

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split last dimension into (num_heads, d_k)."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement _split_heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine heads back to d_model dimension."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement _combine_heads")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask (True = masked)
            training: Whether to apply dropout

        Returns:
            Output tensor of same shape as input
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MultiHeadAttention.forward")

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        return self.forward(x, mask, training)


class TransformerBlock:
    """
    Transformer Decoder Block.

    A single transformer block with:
    - Multi-head self-attention (with causal masking)
    - Position-wise feed-forward network
    - Layer normalization (pre-norm architecture)
    - Residual connections

    Pre-norm architecture (GPT-2 style):
        x → LayerNorm → Attention → + x → LayerNorm → FFN → + x

    This differs from the original post-norm architecture:
        x → Attention → + x → LayerNorm → FFN → + x → LayerNorm

    Pre-norm tends to train more stably for deep models.

    Attributes:
        ln1: First layer normalization
        attn: Multi-head attention
        ln2: Second layer normalization
        ffn: Feed-forward network
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TransformerBlock.__init__")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            training: Whether to apply dropout

        Returns:
            Output tensor of same shape

        Pre-norm architecture:
            1. residual = x
            2. x = LayerNorm(x)
            3. x = Attention(x, mask)
            4. x = Dropout(x)
            5. x = residual + x

            6. residual = x
            7. x = LayerNorm(x)
            8. x = FFN(x)
            9. x = residual + x
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TransformerBlock.forward")

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        return self.forward(x, mask, training)


class TokenEmbedding:
    """Token embedding layer (from Lab 02)."""

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.weight[token_ids]

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)


class PositionalEmbedding:
    """Positional embedding layer (from Lab 02)."""

    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.weight = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02

    def forward(self, seq_len: int) -> np.ndarray:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        return self.weight[:seq_len]

    def __call__(self, seq_len: int) -> np.ndarray:
        return self.forward(seq_len)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create causal attention mask (from Lab 01)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


class GPTModel:
    """
    GPT-style Decoder-Only Transformer.

    A complete language model that:
    1. Embeds tokens and adds positional information
    2. Processes through N transformer blocks
    3. Applies final layer normalization
    4. Projects to vocabulary logits

    Architecture:
        Token IDs → Token Embedding + Positional Embedding
                  → Transformer Block × N
                  → LayerNorm
                  → Output Projection → Logits

    Attributes:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        token_embedding: Token embedding layer
        pos_embedding: Positional embedding layer
        blocks: List of transformer blocks
        ln_f: Final layer normalization
        lm_head: Output projection (can be tied with token embedding)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        """
        Initialize GPT model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length supported
            dropout: Dropout probability
            tie_weights: If True, tie input and output embeddings
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement GPTModel.__init__")

    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            token_ids: Integer array of shape (batch, seq_len)
            training: Whether to apply dropout

        Returns:
            Logits of shape (batch, seq_len, vocab_size)

        Steps:
            1. Get token embeddings
            2. Add positional embeddings
            3. Apply dropout (if training)
            4. Create causal mask
            5. Pass through each transformer block
            6. Apply final layer normalization
            7. Project to vocabulary logits
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement GPTModel.forward")

    def __call__(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        return self.forward(token_ids, training)

    def get_num_params(self) -> int:
        """Count total number of parameters."""
        # This is useful for verifying model size
        total = 0

        # Token embeddings
        total += self.token_embedding.weight.size

        # Position embeddings
        total += self.pos_embedding.weight.size

        # Transformer blocks
        for block in self.blocks:
            # Layer norms
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size

            # Attention weights
            total += block.attn.W_Q.size
            total += block.attn.W_K.size
            total += block.attn.W_V.size
            total += block.attn.W_O.size

            # FFN weights
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size

        # Final layer norm
        total += self.ln_f.gamma.size + self.ln_f.beta.size

        # Output projection (if not tied)
        if not self.tie_weights:
            total += self.lm_head.size

        return total


class OutputProjection:
    """Output projection layer for language modeling head."""

    def __init__(self, d_model: int, vocab_size: int, weight: Optional[np.ndarray] = None):
        self.d_model = d_model
        self.vocab_size = vocab_size
        if weight is not None:
            self.weight = weight
        else:
            self.weight = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
