"""
Lab 03: Feed-Forward Network

Implement the FFN component of transformer blocks.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Literal


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation.

    ReLU(x) = max(0, x)

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with ReLU applied element-wise
    """
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation.

    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with sigmoid applied element-wise
    """
    return 1 / (1 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * Φ(x) where Φ is the Gaussian CDF

    Use the approximation:
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    This approximation is used in GPT-2 and many other models.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with GELU applied element-wise

    Example:
        >>> gelu(np.array([-1, 0, 1]))
        array([-0.158..., 0., 0.841...])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement gelu")


def silu(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Linear Unit (SiLU) activation, also called Swish.

    SiLU(x) = x * sigmoid(x)

    This is the activation used in SwiGLU feed-forward networks.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with SiLU applied element-wise

    Example:
        >>> silu(np.array([-1, 0, 1]))
        array([-0.268..., 0., 0.731...])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement silu")


class FeedForward:
    """
    Standard feed-forward network used in transformers.

    Architecture:
        x -> Linear1 -> Activation -> Linear2 -> output

    This is the "MLP" part of each transformer block.
    It processes each position independently.

    Attributes:
        d_model: Input and output dimension
        d_ff: Hidden layer dimension (typically 4 * d_model)
        activation: Activation function name ('relu' or 'gelu')
        W1: First linear layer weights (d_model, d_ff)
        b1: First linear layer bias (d_ff,)
        W2: Second linear layer weights (d_ff, d_model)
        b2: Second linear layer bias (d_model,)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: Literal['relu', 'gelu'] = 'gelu',
        use_bias: bool = True
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension (default: 4 * d_model)
            activation: Activation function ('relu' or 'gelu')
            use_bias: Whether to use bias terms

        Example:
            >>> ffn = FeedForward(512, 2048)
            >>> x = np.random.randn(2, 10, 512)
            >>> out = ffn(x)
            >>> out.shape
            (2, 10, 512)
        """
        # YOUR CODE HERE
        #
        # Initialize:
        # - self.d_model
        # - self.d_ff (default to 4 * d_model if not specified)
        # - self.activation_name
        # - self.W1: shape (d_model, d_ff), random init * 0.02
        # - self.b1: shape (d_ff,), zeros (or None if not use_bias)
        # - self.W2: shape (d_ff, d_model), random init * 0.02
        # - self.b2: shape (d_model,), zeros (or None if not use_bias)
        raise NotImplementedError("Implement FeedForward.__init__")

    def _get_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured activation function."""
        if self.activation_name == 'relu':
            return relu(x)
        elif self.activation_name == 'gelu':
            return gelu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply feed-forward network.

        Computation:
            hidden = activation(x @ W1 + b1)
            output = hidden @ W2 + b2

        Args:
            x: Input of shape (..., d_model)

        Returns:
            Output of shape (..., d_model)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FeedForward.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x)


class SwiGLUFeedForward:
    """
    SwiGLU feed-forward network used in modern LLMs.

    Architecture:
        x -> [W1 (gate), W2 (value)] -> SiLU(W1 @ x) * (W2 @ x) -> W3 -> output

    The gating mechanism allows the network to learn to selectively
    pass or block information.

    Used in LLaMA, Mistral, and other modern architectures.

    Attributes:
        d_model: Input and output dimension
        d_ff: Hidden layer dimension
        W1: Gate projection (d_model, d_ff)
        W2: Value projection (d_model, d_ff)
        W3: Output projection (d_ff, d_model)
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        """
        Initialize SwiGLU feed-forward network.

        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension. Default: (2/3) * 4 * d_model
                  This matches parameter count with standard FFN

        Example:
            >>> ffn = SwiGLUFeedForward(512)
            >>> x = np.random.randn(2, 10, 512)
            >>> out = ffn(x)
            >>> out.shape
            (2, 10, 512)
        """
        # YOUR CODE HERE
        #
        # Initialize:
        # - self.d_model
        # - self.d_ff: if not specified, compute as int((2/3) * 4 * d_model)
        #   Optionally round to multiple of 256: d_ff = 256 * ((d_ff + 255) // 256)
        # - self.W1: shape (d_model, d_ff), random init * 0.02 (gate)
        # - self.W2: shape (d_model, d_ff), random init * 0.02 (value)
        # - self.W3: shape (d_ff, d_model), random init * 0.02 (output)
        # Note: No biases in SwiGLU
        raise NotImplementedError("Implement SwiGLUFeedForward.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply SwiGLU feed-forward network.

        Computation:
            gate = silu(x @ W1)
            value = x @ W2
            hidden = gate * value  # element-wise multiplication
            output = hidden @ W3

        Args:
            x: Input of shape (..., d_model)

        Returns:
            Output of shape (..., d_model)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement SwiGLUFeedForward.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x)


def count_parameters(model) -> int:
    """
    Count total parameters in a feed-forward network.

    Args:
        model: FeedForward or SwiGLUFeedForward instance

    Returns:
        Total number of parameters
    """
    total = 0
    if hasattr(model, 'W1'):
        total += model.W1.size
    if hasattr(model, 'W2'):
        total += model.W2.size
    if hasattr(model, 'W3'):
        total += model.W3.size
    if hasattr(model, 'b1') and model.b1 is not None:
        total += model.b1.size
    if hasattr(model, 'b2') and model.b2 is not None:
        total += model.b2.size
    return total
