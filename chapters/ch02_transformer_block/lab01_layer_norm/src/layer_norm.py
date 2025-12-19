"""
Lab 01: Layer Normalization

Implement layer normalization from scratch.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Apply layer normalization to input.

    Layer normalization normalizes each sample independently across features.
    Unlike batch normalization, it works the same during training and inference.

    Formula:
        mean = mean(x, axis=-1)
        var = var(x, axis=-1)
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    Args:
        x: Input array of shape (..., d_model)
           The last dimension is the feature dimension to normalize over
        gamma: Scale parameter of shape (d_model,)
               Learnable, typically initialized to ones
        beta: Shift parameter of shape (d_model,)
              Learnable, typically initialized to zeros
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized output of same shape as x

    Example:
        >>> x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> gamma = np.ones(3)
        >>> beta = np.zeros(3)
        >>> out = layer_norm(x, gamma, beta)
        >>> # Each row now has mean~0, var~1
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement layer_norm")


def rms_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Apply RMS (Root Mean Square) normalization to input.

    RMSNorm is a simplified version of LayerNorm that skips the mean centering.
    It's used in modern models like LLaMA for its efficiency.

    Formula:
        rms = sqrt(mean(x^2, axis=-1))
        output = gamma * (x / (rms + eps))

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized output of same shape as x

    Note:
        RMSNorm has no beta parameter (no shift), only gamma (scale).
        This makes it slightly more efficient than LayerNorm.

    Example:
        >>> x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> gamma = np.ones(3)
        >>> out = rms_norm(x, gamma)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement rms_norm")


class LayerNorm:
    """
    Layer Normalization module.

    Normalizes inputs across the feature dimension with learnable
    scale (gamma) and shift (beta) parameters.

    Attributes:
        d_model: Feature dimension size
        eps: Numerical stability constant
        gamma: Scale parameter of shape (d_model,), initialized to ones
        beta: Shift parameter of shape (d_model,), initialized to zeros
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize LayerNorm.

        Args:
            d_model: Size of the feature dimension to normalize
            eps: Small constant for numerical stability

        Example:
            >>> ln = LayerNorm(512)
            >>> x = np.random.randn(2, 10, 512)
            >>> out = ln(x)
            >>> out.shape
            (2, 10, 512)
        """
        # YOUR CODE HERE
        # Initialize:
        # - self.d_model
        # - self.eps
        # - self.gamma: shape (d_model,), initialized to ones
        # - self.beta: shape (d_model,), initialized to zeros
        raise NotImplementedError("Implement LayerNorm.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.

        Args:
            x: Input of shape (..., d_model)

        Returns:
            Normalized output of same shape as x
        """
        # YOUR CODE HERE
        # Use the layer_norm function you implemented above
        raise NotImplementedError("Implement LayerNorm.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x)


class RMSNorm:
    """
    Root Mean Square Normalization module.

    A simplified normalization that's more efficient than LayerNorm.
    Used in modern architectures like LLaMA.

    Attributes:
        d_model: Feature dimension size
        eps: Numerical stability constant
        gamma: Scale parameter of shape (d_model,), initialized to ones
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize RMSNorm.

        Args:
            d_model: Size of the feature dimension to normalize
            eps: Small constant for numerical stability

        Example:
            >>> rms = RMSNorm(512)
            >>> x = np.random.randn(2, 10, 512)
            >>> out = rms(x)
        """
        # YOUR CODE HERE
        # Initialize:
        # - self.d_model
        # - self.eps
        # - self.gamma: shape (d_model,), initialized to ones
        # Note: RMSNorm has no beta parameter
        raise NotImplementedError("Implement RMSNorm.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMS normalization.

        Args:
            x: Input of shape (..., d_model)

        Returns:
            Normalized output of same shape as x
        """
        # YOUR CODE HERE
        # Use the rms_norm function you implemented above
        raise NotImplementedError("Implement RMSNorm.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x)
