"""Common utility functions for the transformer learning course.

These utilities are provided so you can focus on the core concepts
rather than reimplementing basic operations in every lab.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for array x along specified axis.

    Args:
        x: Input array
        axis: Axis along which to compute softmax (default: -1, last axis)

    Returns:
        Softmax probabilities (same shape as x, sums to 1 along axis)

    Note:
        Uses the numerically stable version: softmax(x) = softmax(x - max(x))
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation.

    Args:
        x: Input array

    Returns:
        max(0, x) element-wise
    """
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation.

    The GELU activation is defined as:
        GELU(x) = x * Φ(x)

    where Φ(x) is the cumulative distribution function of the standard
    normal distribution.

    This uses the approximate form:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Args:
        x: Input array

    Returns:
        GELU activation applied element-wise
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization.

    Normalizes across the last dimension (features).

    Args:
        x: Input array of shape (..., features)
        gamma: Scale parameter of shape (features,)
        beta: Shift parameter of shape (features,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
