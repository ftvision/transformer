"""
Lab 04: Transformer Block

Assemble a complete transformer block from the components you've built.

Your task: Complete the TransformerBlock class to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Helper functions (provided - same as previous labs)
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """Apply layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# =============================================================================
# Layer Normalization class (copy from Lab 01 or implement here)
# =============================================================================

class LayerNorm:
    """
    Layer Normalization module.

    You can copy your implementation from Lab 01 or implement here.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize LayerNorm.

        Args:
            d_model: Size of the feature dimension
            eps: Numerical stability constant
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement LayerNorm.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement LayerNorm.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# =============================================================================
# Multi-Head Attention (copy from Chapter 1 Lab 03 or implement here)
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    You can copy your implementation from Chapter 1 Lab 03 or implement here.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MultiHeadAttention.__init__")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute multi-head self-attention.

        Args:
            x: Input of shape (..., seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output of same shape as x
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MultiHeadAttention.forward")

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.forward(x, mask)


# =============================================================================
# Feed-Forward Network (copy from Lab 03 or implement here)
# =============================================================================

class FeedForward:
    """
    Feed-forward network.

    You can copy your implementation from Lab 03 or implement here.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        use_bias: bool = True
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension (default: 4 * d_model)
            use_bias: Whether to use bias terms
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FeedForward.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward network."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement FeedForward.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# =============================================================================
# Transformer Block - THE MAIN IMPLEMENTATION
# =============================================================================

class TransformerBlock:
    """
    A complete transformer block combining:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections

    This is the fundamental unit that gets stacked to build transformers.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        pre_norm: Whether to use pre-norm (True) or post-norm (False)
        norm1: First layer normalization
        attn: Multi-head attention
        norm2: Second layer normalization
        ffn: Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        pre_norm: bool = True
    ):
        """
        Initialize a transformer block.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout probability (stored but not implemented in NumPy)
            pre_norm: If True, use pre-norm style; if False, use post-norm

        Example:
            >>> block = TransformerBlock(d_model=512, num_heads=8)
            >>> x = np.random.randn(2, 10, 512)
            >>> out = block(x)
            >>> out.shape
            (2, 10, 512)
        """
        # YOUR CODE HERE
        #
        # Initialize:
        # - self.d_model
        # - self.num_heads
        # - self.d_ff (default: 4 * d_model)
        # - self.pre_norm
        # - self.norm1 = LayerNorm(d_model)
        # - self.attn = MultiHeadAttention(d_model, num_heads)
        # - self.norm2 = LayerNorm(d_model)
        # - self.ffn = FeedForward(d_model, d_ff)
        raise NotImplementedError("Implement TransformerBlock.__init__")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through the transformer block.

        For pre-norm (self.pre_norm = True):
            x = x + attn(norm1(x))
            x = x + ffn(norm2(x))

        For post-norm (self.pre_norm = False):
            x = norm1(x + attn(x))
            x = norm2(x + ffn(x))

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model) for unbatched input
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input

        Note:
            The mask is passed to the attention layer for causal masking
            or padding mask support.
        """
        # YOUR CODE HERE
        #
        # Implement both pre-norm and post-norm variants based on self.pre_norm
        raise NotImplementedError("Implement TransformerBlock.forward")

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x, mask)


# =============================================================================
# Stacked Transformer (bonus - for building full models)
# =============================================================================

class Transformer:
    """
    Stack of transformer blocks.

    This is the backbone of models like GPT, BERT, etc.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        pre_norm: bool = True
    ):
        """
        Initialize a stack of transformer blocks.

        Args:
            num_layers: Number of transformer blocks to stack
            d_model: Model dimension
            num_heads: Number of attention heads per block
            d_ff: FFN hidden dimension
            pre_norm: Whether to use pre-norm style
        """
        # YOUR CODE HERE
        #
        # Initialize:
        # - self.layers: list of TransformerBlock instances
        # - self.final_norm: LayerNorm(d_model) if pre_norm else None
        raise NotImplementedError("Implement Transformer.__init__")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through all transformer blocks.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # YOUR CODE HERE
        #
        # 1. Pass x through each layer
        # 2. Apply final_norm if using pre-norm
        raise NotImplementedError("Implement Transformer.forward")

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.forward(x, mask)
